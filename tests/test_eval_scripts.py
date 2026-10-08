"""The three evaluation scripts as real subprocesses over hand-built run folders, with a scrubbed
environment (no .env can be read: prices and the threshold are passed explicitly)."""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ingest_support import ROOT
from test_extract_script import scrubbed
from test_failures import build
from test_noise import build as build_repeats

pytestmark = pytest.mark.slow


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=scrubbed(),
        cwd=ROOT,
        timeout=120,
    )


def data_args(tmp_path: Path) -> list[str]:
    return [
        "--manifest",
        str(tmp_path / "data" / "manifest.csv"),
        "--gt-dir",
        str(tmp_path / "ground_truth"),
    ]


class TestAnalyzeFailures:
    def test_it_prints_the_report_or_writes_it(self, tmp_path: Path) -> None:
        run = build(tmp_path)
        printed = run_script("analyze_failures.py", str(run), *data_args(tmp_path))
        assert printed.returncode == 0, printed.stderr
        assert "# Failure analysis: `r1`" in printed.stdout
        assert "5 of 8 fields were returned as `abstained`" in printed.stdout

        out = tmp_path / "docs" / "eval" / "f.md"
        notes = tmp_path / "notes.md"
        notes.write_text("Hand-written.\n", encoding="utf-8")
        written = run_script(
            "analyze_failures.py",
            str(run),
            *data_args(tmp_path),
            "--out",
            str(out),
            "--notes",
            str(notes),
        )
        assert written.returncode == 0 and "8 fields analysed" in written.stdout
        assert (
            out.read_text(encoding="utf-8")
            == printed.stdout.replace("\n\n", "\n\n") + "\n## Reading\n\nHand-written.\n"
        )

    def test_bad_input_exits_2(self, tmp_path: Path) -> None:
        run = build(tmp_path)
        missing = run_script("analyze_failures.py", str(tmp_path / "nope"), *data_args(tmp_path))
        assert missing.returncode == 2 and "not a directory" in missing.stderr
        no_notes = run_script(
            "analyze_failures.py", str(run), *data_args(tmp_path), "--notes", str(tmp_path / "x.md")
        )
        assert no_notes.returncode == 2 and "cannot be read" in no_notes.stderr
        (run / "docs" / "doc_b.json").unlink()
        broken = run_script("analyze_failures.py", str(run), *data_args(tmp_path))
        assert broken.returncode == 2 and "cannot be read" in broken.stderr


class TestNoiseReport:
    def test_it_reports_the_spread_of_repeated_runs(self, tmp_path: Path) -> None:
        runs = [str(run) for run in build_repeats(tmp_path)]
        out = tmp_path / "noise.md"
        done = run_script(
            "noise_report.py",
            *runs,
            *data_args(tmp_path),
            "--price-in",
            "1",
            "--price-out",
            "2",
            "--out",
            str(out),
        )
        assert done.returncode == 0, done.stderr
        assert "3 runs, 4 fields" in done.stdout
        text = out.read_text(encoding="utf-8")
        assert "Between two runs 1 to 2 of 4 fields flip" in text and "$1/$2 per million" in text

    def test_one_run_is_refused(self, tmp_path: Path) -> None:
        (run,) = build_repeats(tmp_path)[:1]
        done = run_script(
            "noise_report.py", str(run), *data_args(tmp_path), "--price-in", "1", "--price-out", "2"
        )
        assert done.returncode == 2 and "at least two runs" in done.stderr


class TestCalibrate:
    def test_it_writes_the_report_and_both_diagrams(self, tmp_path: Path) -> None:
        run = build(tmp_path)
        out, svgs = tmp_path / "c.md", tmp_path / "svg"
        done = run_script(
            "calibrate.py",
            str(run),
            *data_args(tmp_path),
            "--abstain-below",
            "0.3",
            "--out",
            str(out),
            "--svg-dir",
            str(svgs),
        )
        assert done.returncode == 0, done.stderr
        text = out.read_text(encoding="utf-8")
        assert "| in force (`ABSTAIN_BELOW`) | 0.30 |" in text
        assert "| chosen by the rule | 0.95 |" in text
        for name in ("reliability.svg", "risk_coverage.svg"):
            assert ET.fromstring((svgs / name).read_text(encoding="utf-8")).tag.endswith("svg")

    def test_bad_input_exits_2(self, tmp_path: Path) -> None:
        build(tmp_path)
        done = run_script(
            "calibrate.py",
            str(tmp_path / "missing"),
            *data_args(tmp_path),
            "--abstain-below",
            "0.3",
        )
        assert done.returncode == 2 and "not a directory" in done.stderr
