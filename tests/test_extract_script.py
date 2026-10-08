"""scripts/extract.py (and smoke.py's evidence contract) end to end: real subprocesses against the
fake server, with an EMPTY env file and a scrubbed environment, so the real .env (and its key) can
never be read."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from docextract.extract.prompt import PROMPT_SHA256, prompt_sha256
from docextract.ingest.manifest import ManifestRow
from fakes.fake_llm import FakeLLMServer, canned_answer, chat_reply
from files import A4, make_sized_pdf
from ingest_support import ROOT, Env, gray_png, settings, write_manifest_file

pytestmark = pytest.mark.slow

KEY = "sk-script-secret-key"


def scrubbed(**extra: str) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("LLM_", "PRELABEL_", "MAX_RUN_COST", "INGEST_"))
    }
    env.update(extra)
    return env


def row(doc_id: str, **overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "doc_id": doc_id,
        "raw_files": f"{doc_id}.png",
        "content_group": doc_id,
        "doc_type": "invoice",
        "capture": "born_digital",
        "split": "dev",
        "needs_redaction": "n",
    }
    values.update(overrides)
    return ManifestRow.model_validate(values)


class Workspace:
    def __init__(self, tmp_path: Path, env: Env) -> None:
        self.tmp = tmp_path
        self.env = env
        self.labels = tmp_path / "ground_truth"
        self.labels.mkdir()
        self.runs = tmp_path / "runs"
        self.empty_env = tmp_path / "empty.env"
        self.empty_env.write_text("")
        self.rows: list[ManifestRow] = []

    def add_png(self, doc_id: str, **overrides: Any) -> None:
        r = row(doc_id, **overrides)
        gray_png(self.env, f"{doc_id}.png")
        self.env.ingest(r)
        self.rows.append(r)

    def add_pdf(self, doc_id: str, pages: int) -> None:
        r = row(doc_id, raw_files=f"{doc_id}.pdf")
        make_sized_pdf(self.env.raw / f"{doc_id}.pdf", [A4] * pages)
        self.env.ingest(r, settings=settings(dpi=40))
        self.rows.append(r)

    def extract(
        self, server: FakeLLMServer | None, *args: str, **env_extra: str
    ) -> subprocess.CompletedProcess[str]:
        manifest = write_manifest_file(self.tmp, self.rows)
        extra = (
            {}
            if server is None
            else {
                "LLM_BASE_URL": server.base_url,
                "LLM_MODEL": "fake-extractor",
                "LLM_API_KEY": KEY,
                "LLM_MAX_RETRIES": "0",
                "LLM_CACHE": "off",
            }
        )
        extra.update(env_extra)
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "extract.py"),
                *args,
                "--env-file",
                str(self.empty_env),
                "--manifest",
                str(manifest),
                "--pages-dir",
                str(self.env.pages),
                "--ground-truth-dir",
                str(self.labels),
                "--runs-dir",
                str(self.runs),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=scrubbed(**extra),
            cwd=ROOT,
            timeout=120,
        )


@pytest.fixture
def ws(tmp_path: Path, env: Env) -> Workspace:
    return Workspace(tmp_path, env)


def uncited() -> dict[str, Any]:
    answer = canned_answer()
    del answer["vendor_name"]["evidence"]
    return answer


def test_a_dev_split_runs_end_to_end_with_a_repair(ws: Workspace, fake_llm: FakeLLMServer) -> None:
    ws.add_pdf("doc_a", 3)
    ws.add_png("doc_b")
    ws.add_png("doc_c")
    ws.rows.append(row("doc_t", split="test"))  # not in the dev split: never touched
    fake_llm.enqueue(
        chat_reply(canned_answer()),  # doc_a
        chat_reply(uncited()),  # doc_b: unusable, then repaired
        chat_reply(canned_answer()),
    )  # doc_c: the fake's default valid answer
    done = ws.extract(fake_llm, "--split", "dev", "--run-id", "e2e")

    assert done.returncode == 0, done.stdout + done.stderr
    assert any(line.startswith("done: 3 ok, 0 failed") for line in done.stdout.splitlines())
    assert "1 repair(s)" in done.stdout
    run = json.loads((ws.runs / "e2e" / "run.json").read_text(encoding="utf-8"))
    assert run["totals"]["ok"] == 3 and run["totals"]["repairs"] == 1
    assert run["selection"] == {"split": "dev"} and run["stopped"] is None
    assert sorted(p.name for p in (ws.runs / "e2e" / "docs").iterdir()) == [
        "doc_a.json",
        "doc_b.json",
        "doc_c.json",
    ]
    doc_b = json.loads((ws.runs / "e2e" / "docs" / "doc_b.json").read_text(encoding="utf-8"))
    assert doc_b["status"] == "ok" and doc_b["repairs"] == 1
    assert "extracted without evidence" in doc_b["repair_history"][0]["problem"]
    assert len(fake_llm.requests) == 4
    image_counts = [
        sum(part["type"] == "image_url" for part in r.body["messages"][1]["content"])
        for r in fake_llm.requests
    ]
    assert image_counts == [3, 1, 1, 1]  # doc_a's three pages in one call; b, b's repair, c
    for path in (ws.runs / "e2e").rglob("*.json"):
        assert KEY not in path.read_text(encoding="utf-8")
    assert KEY not in done.stdout + done.stderr


def test_an_empty_split_is_refused_before_any_configuration(ws: Workspace) -> None:
    ws.rows.append(row("doc_t", split="test"))
    done = ws.extract(None, "--split", "dev")  # no LLM_* at all: it must not get that far
    assert done.returncode == 2
    assert "no documents in split 'dev'" in done.stderr
    assert not ws.runs.exists()


def test_a_refused_document_makes_the_run_exit_1(ws: Workspace, fake_llm: FakeLLMServer) -> None:
    ws.add_png("doc_a")
    ws.add_png("doc_x", label_mode="blind")
    done = ws.extract(fake_llm, "--doc", "doc_x", "--doc", "doc_a", "--run-id", "r")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "REFUSED   doc_x: doc_x is BLIND" in done.stdout
    assert done.stdout.index("doc_x") < done.stdout.index("ok        doc_a")  # the order given
    assert len(fake_llm.requests) == 1


def test_the_cost_cap_makes_the_run_exit_3(ws: Workspace, fake_llm: FakeLLMServer) -> None:
    ws.add_png("doc_a")
    ws.add_png("doc_b")
    done = ws.extract(
        fake_llm,
        "--split",
        "dev",
        "--run-id",
        "capped",
        LLM_PRICE_INPUT_PER_MTOK="1000",  # 1500 prompt tokens = $1.50 a call
        MAX_RUN_COST_USD="1.00",
    )
    assert done.returncode == 3, done.stdout + done.stderr
    assert "STOPPED: the run cost cap (MAX_RUN_COST_USD) was exceeded: $1.5000 spent" in done.stdout
    assert "NOT_RUN   doc_b" in done.stdout
    assert (ws.runs / "capped" / "run.json").is_file()


def test_a_dry_run_calls_nothing(ws: Workspace, fake_llm: FakeLLMServer) -> None:
    ws.add_pdf("doc_a", 2)
    ws.add_png("doc_x", label_mode="blind")
    ws.rows.append(row("doc_n"))  # never ingested
    done = ws.extract(fake_llm, "--split", "dev", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "would extract doc_a (2 page(s))" in done.stdout
    assert "would refuse  doc_x: doc_x is BLIND" in done.stdout
    assert "would refuse  doc_n: doc_n: not ingested yet" in done.stdout
    assert fake_llm.requests == [] and not ws.runs.exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--doc", "doc_zz"], "not in the manifest: doc_zz"),
        (["--doc", "doc_a", "--run-id", "../escape"], "--run-id '../escape'"),
    ],
)
def test_bad_arguments_exit_2(ws: Workspace, args: list[str], message: str) -> None:
    ws.add_png("doc_a")
    done = ws.extract(None, *args)
    assert done.returncode == 2 and message in done.stderr


def test_an_existing_run_folder_is_never_overwritten(
    ws: Workspace, fake_llm: FakeLLMServer
) -> None:
    ws.add_png("doc_a")
    (ws.runs / "taken").mkdir(parents=True)
    done = ws.extract(fake_llm, "--doc", "doc_a", "--run-id", "taken")
    assert done.returncode == 2 and "already exists" in done.stderr
    assert fake_llm.requests == []


def test_smoke_holds_the_single_image_to_page_1(fake_llm: FakeLLMServer, tmp_path: Path) -> None:
    on_page_two = canned_answer()
    on_page_two["total"]["page"] = 2
    fake_llm.enqueue(chat_reply(on_page_two), chat_reply(canned_answer()))
    empty = tmp_path / "empty.env"
    empty.write_text("")
    artifact = tmp_path / "smoke.json"
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "smoke.py"),
            str(ROOT / "tests" / "fixtures" / "sample_receipt.png"),
            "--env-file",
            str(empty),
            "--output",
            str(artifact),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=scrubbed(LLM_BASE_URL=fake_llm.base_url, LLM_MODEL="fake", LLM_API_KEY=KEY),
        cwd=ROOT,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["repairs"] == 1
    assert (
        "page outside 1..1 (the document has 1 page): total"
        in (saved["repair_history"][0]["problem"])
    )


def test_a_prompt_file_is_sent_and_its_hash_recorded(
    ws: Workspace, fake_llm: FakeLLMServer, tmp_path: Path
) -> None:
    ws.add_png("doc_a")
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(b"Be brief.\r\nSecond line.\r\n")  # CRLF, as a Windows checkout
    done = ws.extract(
        fake_llm, "--doc", "doc_a", "--run-id", "p", "--system-prompt-file", str(prompt)
    )
    assert done.returncode == 0, done.stdout + done.stderr
    text = "Be brief.\nSecond line.\n"
    assert fake_llm.requests[0].body["messages"][0] == {"role": "system", "content": text}
    run = json.loads((ws.runs / "p" / "run.json").read_text(encoding="utf-8"))
    assert run["prompt_sha256"] == prompt_sha256(text)
    assert f"prompt: {prompt_sha256(text)[:12]} (from {prompt})" in done.stdout


def test_without_a_prompt_file_the_production_prompt_is_used(
    ws: Workspace, fake_llm: FakeLLMServer
) -> None:
    ws.add_png("doc_a")
    done = ws.extract(fake_llm, "--doc", "doc_a", "--run-id", "d")
    assert done.returncode == 0, done.stdout + done.stderr
    run = json.loads((ws.runs / "d" / "run.json").read_text(encoding="utf-8"))
    assert run["prompt_sha256"] == PROMPT_SHA256
    assert f"prompt: {PROMPT_SHA256[:12]} (production)" in done.stdout


@pytest.mark.parametrize("content", [None, "", "  \n\t\n"], ids=["missing", "empty", "blank"])
def test_a_missing_or_blank_prompt_file_exits_2_before_any_request(
    ws: Workspace, fake_llm: FakeLLMServer, tmp_path: Path, content: str | None
) -> None:
    ws.add_png("doc_a")
    prompt = tmp_path / "prompt.txt"
    if content is not None:
        prompt.write_text(content, encoding="utf-8")
    done = ws.extract(
        fake_llm, "--doc", "doc_a", "--run-id", "x", "--system-prompt-file", str(prompt)
    )
    assert done.returncode == 2
    assert "--system-prompt-file" in done.stderr
    assert fake_llm.requests == [] and not ws.runs.exists()


def test_concurrency_runs_documents_in_parallel_and_records_it(
    ws: Workspace, fake_llm: FakeLLMServer
) -> None:
    for doc_id in ("doc_a", "doc_b", "doc_c"):
        ws.add_png(doc_id)
    done = ws.extract(fake_llm, "--split", "dev", "--run-id", "par", "--concurrency", "2")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "concurrency 2" in done.stdout
    run = json.loads((ws.runs / "par" / "run.json").read_text(encoding="utf-8"))
    assert run["concurrency"] == 2 and list(run["statuses"]) == ["doc_a", "doc_b", "doc_c"]
    assert run["totals"]["ok"] == 3 and len(fake_llm.requests) == 3


def test_the_default_concurrency_is_one(ws: Workspace, fake_llm: FakeLLMServer) -> None:
    ws.add_png("doc_a")
    done = ws.extract(fake_llm, "--doc", "doc_a", "--run-id", "seq")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "concurrency 1" in done.stdout
    assert (
        json.loads((ws.runs / "seq" / "run.json").read_text(encoding="utf-8"))["concurrency"] == 1
    )


@pytest.mark.parametrize(("value", "message"), [("0", "at least 1"), ("abc", "not a whole number")])
def test_a_bad_concurrency_exits_2_before_any_request(
    ws: Workspace, fake_llm: FakeLLMServer, value: str, message: str
) -> None:
    ws.add_png("doc_a")
    done = ws.extract(fake_llm, "--doc", "doc_a", "--concurrency", value)
    assert done.returncode == 2 and message in done.stderr
    assert fake_llm.requests == [] and not ws.runs.exists()
