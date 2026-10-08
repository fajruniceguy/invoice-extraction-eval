"""scripts/score.py end to end: a run folder on disk, scored by the real CLI in a subprocess.
Subprocesses, so this module is slow."""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from docextract.extract.runner import DocRecord
from docextract.ingest.manifest import ManifestRow
from ingest_support import ROOT
from test_extract_script import scrubbed
from test_scorecard import dataset, document, extracted, label, run_dir

pytestmark = pytest.mark.slow


def score(
    folder: Path, manifest: Path, labels_dir: Path, pages_dir: Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "score.py"),
            str(folder),
            "--manifest",
            str(manifest),
            "--gt-dir",
            str(labels_dir),
            "--pages-dir",
            str(pages_dir),
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=scrubbed(),
        cwd=ROOT,
        timeout=120,
    )


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rows = [
        ManifestRow(
            doc_id="doc_001",
            raw_files=("doc_001.pdf",),
            content_group="doc_001",
            doc_type="invoice",
            capture="born_digital",
            split="dev",
        ),
        ManifestRow(
            doc_id="doc_002",
            raw_files=("doc_002.pdf",),
            content_group="doc_002",
            doc_type="faktur",
            capture="born_digital",
            split="test",
        ),
    ]
    labels = [
        label("doc_001", total=Decimal("10"), vendor_name="Acme"),
        label("doc_002", total=Decimal("20")),
    ]
    manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)
    folder = run_dir(
        tmp_path,
        [
            DocRecord(
                doc_id="doc_001",
                status="ok",
                validated=document(total=extracted(Decimal("10")), vendor_name=extracted("Acme")),
            ),
            DocRecord(
                doc_id="doc_002", status="ok", validated=document(total=extracted(Decimal("25")))
            ),
        ],
    )
    return folder, manifest, labels_dir, pages_dir


def test_scoring_a_run_writes_the_artifacts(fixture: tuple[Path, Path, Path, Path]) -> None:
    folder, manifest, labels_dir, pages_dir = fixture
    done = score(folder, manifest, labels_dir, pages_dir)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "2 document(s) scored, 0 without a ground-truth label" in done.stdout
    assert "accuracy 0.667" in done.stdout
    assert "numeric_mismatch 1" in done.stdout
    out = folder / "score"
    assert {path.name for path in out.iterdir()} == {
        "manifest.json",
        "scorecard.json",
        "scorecard.md",
        "docs",
    }
    assert (out / "docs" / "doc_002.json").is_file()
    assert "# Scorecard: run test-run" in (out / "scorecard.md").read_text(encoding="utf-8")


def test_out_puts_the_scorecard_somewhere_else(
    fixture: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    folder, manifest, labels_dir, pages_dir = fixture
    elsewhere = tmp_path / "reports" / "m5"
    done = score(folder, manifest, labels_dir, pages_dir, "--out", str(elsewhere))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (elsewhere / "scorecard.json").is_file()
    assert not (folder / "score").exists()


def test_a_run_folder_without_its_run_record_exits_1(
    fixture: tuple[Path, Path, Path, Path],
) -> None:
    folder, manifest, labels_dir, pages_dir = fixture
    (folder / "run.json").unlink()
    done = score(folder, manifest, labels_dir, pages_dir)
    assert done.returncode == 1
    assert "cannot be read" in done.stderr


def test_a_run_without_labels_exits_1_and_names_the_unscored_documents(
    fixture: tuple[Path, Path, Path, Path],
) -> None:
    folder, manifest, labels_dir, pages_dir = fixture
    for path in labels_dir.glob("*.json"):
        path.unlink()
    done = score(folder, manifest, labels_dir, pages_dir)
    assert done.returncode == 1
    assert "0 document(s) scored" in done.stdout
    assert "unscored: doc_001, doc_002" in done.stdout
    assert "no document has a ground-truth label yet" in done.stderr


def test_a_missing_run_folder_exits_2(
    fixture: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    _, manifest, labels_dir, pages_dir = fixture
    done = score(tmp_path / "nope", manifest, labels_dir, pages_dir)
    assert done.returncode == 2
    assert "not a directory" in done.stderr


def test_the_summary_has_a_cost_and_latency_line(fixture: tuple[Path, Path, Path, Path]) -> None:
    folder, manifest, labels_dir, pages_dir = fixture
    done = score(folder, manifest, labels_dir, pages_dir)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (
        "cost: $0.0000 paid ($0.0000 at list price, $0.0000/doc) over 2 document(s)" in done.stdout
    )
    assert "latency p50 0.0 s / p95 0.0 s over 2 live call(s)" in done.stdout
    scorecard = (folder / "score" / "scorecard.md").read_text(encoding="utf-8")
    assert "## Cost and latency" in scorecard


def other_run(tmp_path: Path) -> Path:
    """The fixture's two documents again, both right (its own run misses one)."""
    return run_dir(
        tmp_path,
        [
            DocRecord(
                doc_id="doc_001",
                status="ok",
                validated=document(total=extracted(Decimal("10")), vendor_name=extracted("Acme")),
            ),
            DocRecord(
                doc_id="doc_002", status="ok", validated=document(total=extracted(Decimal("20")))
            ),
        ],
        name="other",
    )


def test_baseline_adds_a_before_after_section_and_a_summary_line(
    fixture: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    folder, manifest, labels_dir, pages_dir = fixture
    baseline = other_run(tmp_path)
    done = score(folder, manifest, labels_dir, pages_dir, "--baseline", str(baseline))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "vs baseline other: accuracy 1.000 -> 0.667 (-33.3 pts)" in done.stdout
    assert "McNemar p_worse" in done.stdout and "verdict" in done.stdout
    card = (folder / "score" / "scorecard.md").read_text(encoding="utf-8")
    assert "## Before / after" in card
    assert "Baseline `other` against this run `test-run`: 2 paired document(s)" in card
    payload = json.loads((folder / "score" / "scorecard.json").read_text(encoding="utf-8"))
    assert payload["before_after"]["baseline"] == "other"
    assert payload["before_after"]["metrics"]["accuracy"]["delta"] == pytest.approx(-1 / 3)


def test_without_baseline_nothing_about_a_baseline_is_written(
    fixture: tuple[Path, Path, Path, Path],
) -> None:
    folder, manifest, labels_dir, pages_dir = fixture
    done = score(folder, manifest, labels_dir, pages_dir)
    assert "vs baseline" not in done.stdout
    assert "Before / after" not in (folder / "score" / "scorecard.md").read_text(encoding="utf-8")


def test_a_baseline_that_cannot_be_read_exits_1_and_says_so(
    fixture: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    folder, manifest, labels_dir, pages_dir = fixture
    done = score(
        folder, manifest, labels_dir, pages_dir, "--baseline", str(tmp_path / "no-such-run")
    )
    assert done.returncode == 1
    assert "cannot compare with the baseline" in done.stderr


def test_baseline_with_several_run_folders_is_a_usage_error(
    fixture: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    folder, manifest, labels_dir, pages_dir = fixture
    baseline = other_run(tmp_path)
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "score.py"),
            str(folder),
            str(baseline),
            "--baseline",
            str(baseline),
            "--manifest",
            str(manifest),
            "--gt-dir",
            str(labels_dir),
            "--pages-dir",
            str(pages_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=scrubbed(),
        cwd=ROOT,
        timeout=120,
    )
    assert done.returncode == 2
    assert "--baseline takes a single run folder" in done.stderr
