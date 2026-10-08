"""scripts/check_results.py: a committed result is scored again offline and must reproduce, byte for
byte, every file of its score/ folder."""

import importlib.util
import json
import shutil
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from evalkit.groundtruth import GroundTruth
from evalkit.gt_store import write_label
from evalkit.scorecard import score_run
from ingest_support import ROOT
from test_promote_run import SCANNER, promote_run, scored  # noqa: F401  (scored is a fixture)

spec = importlib.util.spec_from_file_location(
    "check_results", ROOT / "scripts" / "check_results.py"
)
assert spec is not None and spec.loader is not None
check_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_results)


@pytest.fixture
def result(scored: Path, tmp_path: Path) -> Path:  # noqa: F811
    """The scored run, promoted to tmp_path/results/test-run, its score made on 'another commit
    long ago' so that nothing a re-score would produce today (git SHA, time) matches by luck."""
    shutil.rmtree(scored / "score")
    score_run(
        scored,
        manifest_path=tmp_path / "data" / "manifest.csv",
        labels_dir=tmp_path / "ground_truth",
        pages_dir=tmp_path / "data" / "pages",
        now=datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC),
        git_sha="0123456789abcdef0123456789abcdef01234567",
        git_dirty=True,
    ).write(scored / "score")
    return promote_run.promote(scored, tmp_path / "results", SCANNER)


def args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--results-dir",
        str(tmp_path / "results"),
        "--manifest",
        str(tmp_path / "data" / "manifest.csv"),
        "--pages-dir",
        str(tmp_path / "data" / "pages"),
        "--gt-dir",
        str(tmp_path / "ground_truth"),
        *extra,
    ]


def test_a_result_reproduces(
    result: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check_results.main(args(tmp_path)) == 0
    assert "test-run: reproduces" in capsys.readouterr().out


def test_the_git_sha_and_time_of_the_committed_score_are_kept(result: Path, tmp_path: Path) -> None:
    """A re-score on another commit, another day, still reproduces the committed files."""
    path = result / "score" / "manifest.json"
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed["git_sha"] == "0123456789abcdef0123456789abcdef01234567"
    # the committed values are what a re-score is run with; none of them is "now"
    sha, dirty, when = check_results.committed_facts(result)
    assert (sha, dirty, when.isoformat()) == (
        committed["git_sha"],
        committed["git_dirty"],
        committed["scored_at"],
    )
    assert (
        check_results.differences(
            result,
            tmp_path / "data" / "manifest.csv",
            tmp_path / "data" / "pages",
            tmp_path / "ground_truth",
        )
        == []
    )


def test_an_edited_record_that_moves_a_score_is_caught(
    result: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = result / "docs" / "doc_002.json"
    text = record.read_text(encoding="utf-8")
    assert '"value": "25"' in text
    record.write_text(text.replace('"value": "25"', '"value": "20"', 1), encoding="utf-8")
    assert check_results.main(args(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "DOES NOT REPRODUCE" in out and "differs: scorecard.json" in out


def test_an_edited_scorecard_is_caught(result: Path, tmp_path: Path) -> None:
    card = result / "score" / "scorecard.md"
    card.write_text(card.read_text(encoding="utf-8") + "\nedited\n", encoding="utf-8")
    assert check_results.main(args(tmp_path)) == 1


def test_a_changed_label_is_caught(result: Path, tmp_path: Path) -> None:
    write_label(
        tmp_path / "ground_truth",
        GroundTruth(
            doc_id="doc_002",
            label_mode="prelabel",
            labeled_at=datetime(2026, 10, 9, tzinfo=UTC),
            total=Decimal("25"),
        ),
    )
    assert check_results.main(args(tmp_path)) == 1


def test_a_stray_or_missing_score_file_is_caught(
    result: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (result / "score" / "extra.json").write_text("{}", encoding="utf-8")
    assert check_results.main(args(tmp_path)) == 1
    assert "not produced by a fresh score: extra.json" in capsys.readouterr().out
    (result / "score" / "extra.json").unlink()
    shutil.rmtree(result / "score" / "docs")
    assert check_results.main(args(tmp_path)) == 1
    assert "missing from the committed score: docs/doc_001.json" in capsys.readouterr().out


def test_nothing_to_check_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert check_results.main(["--results-dir", str(tmp_path / "none")]) == 2
    assert "no result to check" in capsys.readouterr().err


def test_a_result_without_a_score_manifest_is_exit_2(result: Path, tmp_path: Path) -> None:
    (result / "score" / "manifest.json").unlink()
    assert check_results.main(args(tmp_path)) == 2


def test_a_named_folder_is_checked_alone(result: Path, tmp_path: Path) -> None:
    assert check_results.main(args(tmp_path, str(result))) == 0
