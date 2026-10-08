"""scripts/compare.py end to end: run folders on disk, compared by the real CLI in a subprocess.
Subprocesses, so this module is slow."""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from ingest_support import ROOT
from test_compare import Scenario, manifest_row, many_documents, ok
from test_extract_script import scrubbed
from test_scorecard import dataset, extracted, label, run_dir

pytestmark = pytest.mark.slow


def compare(
    scenario: Scenario,
    baseline: Path,
    candidate: Path,
    out: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare.py"),
            str(baseline),
            str(candidate),
            "--manifest",
            str(scenario.manifest),
            "--gt-dir",
            str(scenario.labels),
            "--pages-dir",
            str(scenario.pages),
            "--out",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=scrubbed(),
        cwd=ROOT,
        timeout=120,
    )


def test_identical_runs_exit_0_and_write_the_report(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path)
    out = tmp_path / "report"
    done = compare(scenario, scenario.baseline, scenario.baseline, out)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines()[0].startswith("VERDICT: OK")
    assert "differs: nothing" in done.stdout
    assert {path.name for path in out.iterdir()} == {"comparison.json", "comparison.md"}
    assert json.loads((out / "comparison.json").read_text(encoding="utf-8"))["verdict"] == "ok"
    assert "# Comparison:" in (out / "comparison.md").read_text(encoding="utf-8")
    assert f"report: {out / 'comparison.md'}" in done.stdout


def test_a_degraded_run_exits_1_with_the_verdict_first(tmp_path: Path) -> None:
    scenario = many_documents(tmp_path, count=12, wrong=7)
    done = compare(scenario, scenario.baseline, scenario.candidate, tmp_path / "report")
    assert done.returncode == 1, done.stdout + done.stderr
    assert done.stdout.splitlines()[0].startswith("VERDICT: REGRESSION")
    assert "regressions (7):" in done.stdout
    data = json.loads((tmp_path / "report" / "comparison.json").read_text(encoding="utf-8"))
    assert data["verdict"] == "regression" and len(data["flips"]["regression"]) == 7


def test_the_thresholds_are_flags(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path)
    args = (scenario, scenario.baseline, scenario.candidate)
    assert compare(*args, tmp_path / "a").returncode == 0  # 3 regressions of 9: not significant
    loose = compare(*args, tmp_path / "b", "--alpha", "0.5", "--max-drop", "0")
    assert loose.returncode == 1 and loose.stdout.startswith("VERDICT: REGRESSION")
    strict = compare(*args, tmp_path / "c", "--alpha", "0.01", "--max-drop", "0.9")
    assert strict.returncode == 0


def test_the_report_is_byte_identical_on_a_second_run(tmp_path: Path) -> None:
    scenario = many_documents(tmp_path, count=12, wrong=7)
    for name in ("one", "two"):
        compare(scenario, scenario.baseline, scenario.candidate, tmp_path / name)
    for artifact in ("comparison.json", "comparison.md"):
        first = (tmp_path / "one" / artifact).read_bytes()
        assert first == (tmp_path / "two" / artifact).read_bytes()


def test_a_missing_run_folder_exits_2(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path)
    done = compare(scenario, scenario.baseline, tmp_path / "nope", tmp_path / "report")
    assert done.returncode == 2 and "not a directory" in done.stderr
    assert not (tmp_path / "report").exists()


def test_no_labeled_document_in_common_exits_2(tmp_path: Path) -> None:
    rows = [manifest_row("doc_001"), manifest_row("doc_002")]
    labels = [label(row.doc_id, total=Decimal("10")) for row in rows]
    manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)
    scenario = Scenario.__new__(Scenario)
    scenario.manifest, scenario.labels, scenario.pages = manifest, labels_dir, pages_dir
    one = run_dir(tmp_path, [ok("doc_001", total=extracted(Decimal("10")))], name="a")
    two = run_dir(tmp_path, [ok("doc_002", total=extracted(Decimal("10")))], name="b")
    done = compare(scenario, one, two, tmp_path / "report")
    assert done.returncode == 2
    assert "cannot compare" in done.stderr and "nothing to compare" in done.stderr


def test_a_run_folder_without_its_run_record_exits_2(tmp_path: Path) -> None:
    scenario = Scenario(tmp_path)
    (scenario.candidate / "run.json").unlink()
    done = compare(scenario, scenario.baseline, scenario.candidate, tmp_path / "report")
    assert done.returncode == 2 and "cannot be read" in done.stderr


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--alpha", "1.5"], "alpha"),
        (["--alpha", "0"], "alpha"),
        (["--max-drop", "-1"], "max_drop"),
        (["--resamples", "0"], "resamples"),
    ],
)
def test_bad_arguments_exit_2(tmp_path: Path, flags: list[str], message: str) -> None:
    scenario = Scenario(tmp_path)
    done = compare(scenario, scenario.baseline, scenario.candidate, tmp_path / "r", *flags)
    assert done.returncode == 2 and message in done.stderr
