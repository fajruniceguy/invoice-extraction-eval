"""scripts/check_run.py end to end: an extract.py run against the fake server, then the DoD check
over the real run folder it wrote. Subprocesses, so this module is slow."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fakes.fake_llm import FakeLLMServer, canned_answer, chat_reply
from ingest_support import ROOT, Env
from test_extract_script import Workspace, scrubbed

pytestmark = pytest.mark.slow


@pytest.fixture
def ws(tmp_path: Path, env: Env) -> Workspace:
    """The same harness test_extract_script.py uses (fixtures are not shared across modules)."""
    return Workspace(tmp_path, env)


def check(run_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run scripts/check_run.py over a run folder, with the real environment scrubbed."""
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_run.py"), str(run_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=scrubbed(),
        cwd=ROOT,
        timeout=120,
    )


def test_a_real_run_passes_the_definition_of_done(
    ws: Workspace, fake_llm: FakeLLMServer
) -> None:
    ws.add_pdf("doc_a", 3)
    ws.add_png("doc_b")
    fake_llm.enqueue(chat_reply(canned_answer()), chat_reply(canned_answer()))
    done = ws.extract(fake_llm, "--split", "dev", "--run-id", "e2e")
    assert done.returncode == 0, done.stdout + done.stderr

    checked = check(ws.runs / "e2e")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "2 document(s) checked" in checked.stdout
    assert "all fields carry value/confidence/evidence/status/reasons" in checked.stdout


def test_a_doctored_record_fails_the_check(ws: Workspace, fake_llm: FakeLLMServer) -> None:
    ws.add_png("doc_a")
    fake_llm.enqueue(chat_reply(canned_answer()))
    assert ws.extract(fake_llm, "--doc", "doc_a", "--run-id", "damaged").returncode == 0

    path = ws.runs / "damaged" / "docs" / "doc_a.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["validated"]["total"]["evidence"] = None
    path.write_text(json.dumps(data), encoding="utf-8")

    checked = check(ws.runs / "damaged")
    assert checked.returncode == 1
    assert "FAIL doc_a total: extracted without evidence" in checked.stdout
