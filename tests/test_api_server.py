"""The real server: `python -m uvicorn --factory docextract.api.app:create_app` in a subprocess,
configured only through the environment, against the fake LLM. It runs from an EMPTY working
directory with a scrubbed environment, so the repository's .env (and its key) can never be read.
Subprocesses, so this module is slow."""

import io
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from PIL import Image

from fakes.fake_llm import FakeLLMServer

pytestmark = pytest.mark.slow

TOKEN = "server-test-token"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), (200, 100, 50)).save(buffer, "PNG")
    return buffer.getvalue()


def scrubbed(**extra: str) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("LLM_", "PRELABEL_", "MAX_RUN_COST", "API_", "CALL_LOG", "INGEST_"))
    }
    env.update(extra)
    return env


@pytest.fixture
def server(tmp_path: Path, fake_llm: FakeLLMServer) -> Iterator[tuple[str, Path]]:
    port = free_port()
    workdir = tmp_path / "cwd"
    workdir.mkdir()  # empty: no .env here
    log = tmp_path / "calls.jsonl"
    env = scrubbed(
        LLM_BASE_URL=fake_llm.base_url,
        LLM_MODEL="fake-extractor",
        LLM_API_KEY="sk-server-test",
        LLM_CACHE="off",
        LLM_MAX_RETRIES="0",
        API_CONCURRENCY="2",
        API_TOKEN=TOKEN,
        CALL_LOG_PATH=str(log),
    )
    stderr = (tmp_path / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "--factory",
            "docextract.api.app:create_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=workdir,
        env=env,
        stdout=stderr,
        stderr=stderr,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if process.poll() is not None or time.monotonic() > deadline:
                stderr.flush()
                raise AssertionError(
                    "the server did not come up:\n" + (tmp_path / "server.log").read_text("utf-8")
                )
            time.sleep(0.2)
        yield base, log
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        stderr.close()


def test_the_server_starts_from_the_environment_and_answers_health(
    server: tuple[str, Path],
) -> None:
    base, _ = server
    health = httpx.get(f"{base}/health").json()
    assert health["status"] == "ok" and health["model"] == "fake-extractor"
    assert health["concurrency"] == 2
    assert "sk-server-test" not in json.dumps(health) and TOKEN not in json.dumps(health)


def test_an_upload_is_extracted_and_logged(
    server: tuple[str, Path], fake_llm: FakeLLMServer
) -> None:
    base, log = server
    files = [("files", ("invoice.png", png(), "image/png"))]
    refused = httpx.post(f"{base}/extract", files=files, timeout=60)
    assert refused.status_code == 401  # the token came from the environment
    assert fake_llm.requests == []

    answer = httpx.post(
        f"{base}/extract", files=files, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=60
    )
    assert answer.status_code == 200, answer.text
    record = answer.json()
    assert record["status"] == "ok"
    assert record["validated"]["vendor_name"]["value"] == "CORNER CAFE DEMO"
    assert len(fake_llm.requests) == 1

    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["request_id"] == answer.headers["x-request-id"]
    assert lines[0]["route"] == "/extract" and lines[0]["input_tokens"] == 1500


def test_a_batch_over_the_real_server(server: tuple[str, Path], fake_llm: FakeLLMServer) -> None:
    base, log = server
    files = [("files", (f"d{i}.png", png(), "image/png")) for i in range(3)]
    answer = httpx.post(
        f"{base}/extract/batch",
        files=files,
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=60,
    )
    body = answer.json()
    assert answer.status_code == 200 and body["run"]["ok"] == 3
    assert [d["doc_id"] for d in body["documents"]] == ["d0", "d1", "d2"]
    assert len(log.read_text(encoding="utf-8").splitlines()) == 3
