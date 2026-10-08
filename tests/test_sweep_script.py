"""scripts/sweep.py end to end: real subprocesses against the fake server, an EMPTY env file and a
scrubbed environment (the real .env can never be read). Subprocesses, so this module is slow."""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from evalkit.groundtruth import GroundTruth
from evalkit.gt_store import write_label
from fakes.fake_llm import FakeLLMServer, Reply, canned_answer, chat_reply
from ingest_support import ROOT, write_manifest_file
from test_extract_script import KEY, Workspace, scrubbed
from test_scorecard import NOW

pytestmark = pytest.mark.slow

PRICES = {"LLM_PRICE_INPUT_PER_MTOK": "1000", "LLM_PRICE_OUTPUT_PER_MTOK": "2000"}
DOCS = ("doc_a", "doc_b", "doc_c")


def wrong() -> dict[str, Any]:
    answer = canned_answer()
    answer["total"]["value"] = 99.0
    answer["total"]["evidence"] = "TOTAL $99.00"
    return answer


def cheap(tokens: int = 600, answer: dict[str, Any] | None = None) -> Reply:
    return chat_reply(
        answer or canned_answer(), prompt_tokens=tokens, completion_tokens=tokens // 2
    )


@pytest.fixture
def ws(tmp_path: Path, env: Any) -> Workspace:
    workspace = Workspace(tmp_path, env)
    for doc_id in DOCS:
        workspace.add_png(doc_id)
        write_label(
            workspace.labels,
            GroundTruth(
                doc_id=doc_id,
                label_mode="prelabel",
                labeled_at=NOW,
                vendor_name="CORNER CAFE DEMO",
                total=Decimal("17.28"),
            ),
        )
    return workspace


def baseline(ws: Workspace, server: FakeLLMServer) -> None:
    done = ws.extract(server, "--split", "dev", "--run-id", "base", "--concurrency", "1", **PRICES)
    assert done.returncode == 0, done.stdout + done.stderr
    server.requests.clear()


def sweep(
    ws: Workspace, server: FakeLLMServer | None, *args: str, **env_extra: str
) -> subprocess.CompletedProcess[str]:
    manifest = write_manifest_file(ws.tmp, ws.rows)
    extra: dict[str, str] = dict(PRICES)
    if server is not None:
        extra.update(
            LLM_BASE_URL=server.base_url,
            LLM_MODEL="fake-extractor",
            LLM_API_KEY=KEY,
            LLM_MAX_RETRIES="0",
            LLM_CACHE="off",
        )
    extra.update(env_extra)
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sweep.py"),
            "--baseline-run",
            str(ws.runs / "base"),
            "--sweep-id",
            "t",
            "--concurrency",
            "1",
            *args,
            "--env-file",
            str(ws.empty_env),
            "--manifest",
            str(manifest),
            "--pages-dir",
            str(ws.env.pages),
            "--ground-truth-dir",
            str(ws.labels),
            "--runs-dir",
            str(ws.runs),
            "--sweeps-dir",
            str(ws.tmp / "sweeps"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=scrubbed(**extra),
        cwd=ROOT,
        timeout=180,
    )


def test_a_full_sweep_runs_every_point_and_chooses_by_the_rule(
    ws: Workspace, fake_llm: FakeLLMServer
) -> None:
    baseline(ws, fake_llm)
    fake_llm.enqueue(
        *[cheap(600)] * 3,  # res-1024: as accurate, cheaper
        *[chat_reply(canned_answer(), prompt_tokens=3000, completion_tokens=1400)] * 3,  # dearer
        *[cheap(300, wrong())] * 3,  # effort-off: cheap but wrong
        cheap(300),
        Reply(500, {"error": {"message": "upstream exploded"}}),
        cheap(300),  # effort-low: cheap but a document failed
        *[cheap(200)] * 3,  # combined (res-1024 alone won): cheaper still
    )
    done = sweep(ws, fake_llm, "--resolution", "1024,1568", "--effort", "off,low", "--confirm")
    assert done.returncode == 1, done.stdout + done.stderr  # one document failed in effort-low
    report = (ws.tmp / "sweeps" / "t" / "sweep.md").read_text(encoding="utf-8")
    assert "Chosen: **combined**." in report
    for label in ("baseline", "res-1024", "res-1568", "effort-off", "effort-low", "combined"):
        assert f"| {label} |" in report, label
    assert "- res-1568: not cheaper than the baseline" in report
    assert "- effort-off: accuracy fell" in report
    assert "- effort-low: 1 document(s) failed or were refused" in report

    data = json.loads((ws.tmp / "sweeps" / "t" / "sweep.json").read_text(encoding="utf-8"))
    assert data["chosen"] == "combined" and len(data["rows"]) == 6
    by_label = {row["label"]: row for row in data["rows"]}
    assert by_label["effort-off"]["accuracy_delta"] == pytest.approx(-0.5, abs=0.2)
    assert by_label["effort-low"]["failed"] == 1

    # every point has its own run folder, and what the point varied reached the request and run.json
    run = lambda name: json.loads(  # noqa: E731
        (ws.runs / f"sweep-t-{name}" / "run.json").read_text(encoding="utf-8")
    )
    assert run("res-1024")["settings"]["image_max_long_edge"] == 1024
    assert run("res-1568")["settings"]["image_max_long_edge"] == 1568
    assert run("effort-off")["settings"]["reasoning_effort"] == "off"
    assert run("combined")["settings"]["image_max_long_edge"] == 1024
    assert run("combined")["settings"]["reasoning_effort"] is None
    assert run("res-1024")["selection"] == {"split": "dev", "sweep": "t", "point": "res-1024"}
    bodies = [request.body for request in fake_llm.requests]
    assert len(bodies) == 15  # 5 runs x 3 documents
    assert [b.get("reasoning_effort") for b in bodies[6:9]] == ["off"] * 3
    assert all("reasoning_effort" not in b for b in bodies[:6] + bodies[12:])


def test_a_dry_run_prints_the_plan_and_calls_nothing(
    ws: Workspace, fake_llm: FakeLLMServer
) -> None:
    baseline(ws, fake_llm)
    done = sweep(ws, fake_llm, "--confirm", "--dry-run")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "res-1024" in done.stdout and "effort-low" in done.stdout and "combined" in done.stdout
    assert "planned requests: 15 (3 documents x 5 runs)" in done.stdout
    assert "dry run: nothing was called" in done.stdout
    assert fake_llm.requests == []
    assert not any(path.name.startswith("sweep-") for path in ws.runs.iterdir())
    assert not (ws.tmp / "sweeps").exists()


def test_resume_skips_finished_points_and_a_repeat_without_it_is_refused(
    ws: Workspace, fake_llm: FakeLLMServer
) -> None:
    baseline(ws, fake_llm)
    args = ("--resolution", "1024", "--effort", "low")
    first = sweep(ws, fake_llm, *args)
    assert first.returncode == 0, first.stdout + first.stderr
    assert len(fake_llm.requests) == 6
    again = sweep(ws, fake_llm, *args)
    assert again.returncode == 2 and "use --resume or choose another --sweep-id" in again.stderr
    assert len(fake_llm.requests) == 6
    resumed = sweep(ws, fake_llm, *args, "--resume")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert resumed.stdout.count("already done, skipped (--resume)") == 2
    assert len(fake_llm.requests) == 6  # nothing new was asked
    assert (ws.tmp / "sweeps" / "t" / "sweep.md").is_file()


def test_resume_runs_only_the_points_that_are_missing(
    ws: Workspace, fake_llm: FakeLLMServer
) -> None:
    baseline(ws, fake_llm)
    assert sweep(ws, fake_llm, "--resolution", "1024", "--effort", "").returncode == 0
    assert len(fake_llm.requests) == 3
    more = sweep(ws, fake_llm, "--resolution", "1024", "--effort", "low", "--resume")
    assert more.returncode == 0, more.stdout + more.stderr
    assert len(fake_llm.requests) == 6  # only effort-low ran
    assert more.stdout.count("already done, skipped (--resume)") == 1


def test_a_half_finished_run_folder_is_never_overwritten(
    ws: Workspace, fake_llm: FakeLLMServer
) -> None:
    baseline(ws, fake_llm)
    (ws.runs / "sweep-t-res-1024" / "docs").mkdir(parents=True)
    done = sweep(ws, fake_llm, "--resolution", "1024", "--effort", "")
    assert done.returncode == 2 and "holds a half-finished run: delete it first" in done.stderr
    assert fake_llm.requests == []


def test_zero_prices_are_refused_because_cost_is_an_axis(
    ws: Workspace, fake_llm: FakeLLMServer
) -> None:
    baseline(ws, fake_llm)
    done = sweep(ws, fake_llm, LLM_PRICE_INPUT_PER_MTOK="0", LLM_PRICE_OUTPUT_PER_MTOK="0")
    assert done.returncode == 2 and "both 0" in done.stderr
    assert fake_llm.requests == []


def test_a_missing_baseline_is_refused(ws: Workspace, fake_llm: FakeLLMServer) -> None:
    done = sweep(ws, fake_llm)  # no baseline run was made
    assert done.returncode == 2 and "no run.json" in done.stderr
    assert fake_llm.requests == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--resolution", "0"], "sizes are whole numbers"),
        (["--resolution", "1024,1024"], "listed twice"),
        (["--resolution", "big"], "cannot read every level"),
        (["--resolution", "", "--effort", ""], "nothing to sweep"),
        (["--tolerance", "2"], "--tolerance"),
        (["--concurrency", "0"], "--concurrency"),
    ],
)
def test_bad_arguments_exit_2_before_any_request(
    ws: Workspace, fake_llm: FakeLLMServer, args: list[str], message: str
) -> None:
    baseline(ws, fake_llm)
    done = sweep(ws, fake_llm, *args)
    assert done.returncode == 2 and message in done.stderr
    assert fake_llm.requests == []


def test_a_sweep_id_with_odd_characters_is_refused(ws: Workspace, fake_llm: FakeLLMServer) -> None:
    baseline(ws, fake_llm)
    manifest = write_manifest_file(ws.tmp, ws.rows)
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sweep.py"),
            "--baseline-run",
            str(ws.runs / "base"),
            "--sweep-id",
            "../escape",
            "--env-file",
            str(ws.empty_env),
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=scrubbed(**PRICES),
        cwd=ROOT,
        timeout=60,
    )
    assert done.returncode == 2 and "letters, digits" in done.stderr
