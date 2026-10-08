"""scripts/promote_run.py: a finished, scored run is copied byte for byte to results/<run_id>/, and
anything that makes it not a result worth keeping is refused with nothing written."""

import importlib.util
import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from docextract.extract.runner import DocRecord
from docextract.ingest.manifest import ManifestRow
from evalkit.scorecard import score_run
from ingest_support import ROOT
from test_scorecard import dataset, document, extracted, label, run_dir

spec = importlib.util.spec_from_file_location("promote_run", ROOT / "scripts" / "promote_run.py")
assert spec is not None and spec.loader is not None
promote_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promote_run)

SCANNER = promote_run.load_scanner(ROOT)


def row(doc_id: str) -> ManifestRow:
    return ManifestRow(
        doc_id=doc_id,
        raw_files=(f"{doc_id}.pdf",),
        content_group=doc_id,
        doc_type="invoice",
        capture="born_digital",
        split="test",
    )


@pytest.fixture
def scored(tmp_path: Path) -> Path:
    """outputs/runs/test-run with two ok documents and its score/ folder."""
    manifest, labels_dir, pages_dir = dataset(
        tmp_path,
        [row("doc_001"), row("doc_002")],
        [label("doc_001", total=Decimal("10")), label("doc_002", total=Decimal("20"))],
    )
    folder = run_dir(
        tmp_path,
        [
            DocRecord(
                doc_id="doc_001", status="ok", validated=document(total=extracted(Decimal("10")))
            ),
            DocRecord(
                doc_id="doc_002", status="ok", validated=document(total=extracted(Decimal("25")))
            ),
        ],
    )
    score_run(folder, manifest_path=manifest, labels_dir=labels_dir, pages_dir=pages_dir).write(
        folder / "score"
    )
    return folder


def edit_run(folder: Path, **changes: object) -> None:
    path = folder / "run.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def tree(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def promote(folder: Path, tmp_path: Path, **kwargs: bool) -> Path:
    return promote_run.promote(folder, tmp_path / "results", SCANNER, **kwargs)


class TestPromote:
    def test_it_copies_run_docs_and_score_byte_for_byte(self, scored: Path, tmp_path: Path) -> None:
        target = promote(scored, tmp_path)
        assert target == tmp_path / "results" / "test-run"
        copied = tree(target)
        assert copied == tree(scored)
        assert {"run.json", "docs/doc_001.json", "docs/doc_002.json", "score/scorecard.md"} <= set(
            copied
        )
        assert "score/docs/doc_002.json" in copied

    def test_stray_files_in_the_run_folder_are_not_copied(
        self, scored: Path, tmp_path: Path
    ) -> None:
        (scored / "notes.txt").write_text("scratch", encoding="utf-8")
        (scored / "docs" / "x.tmp").write_text("x", encoding="utf-8")
        copied = tree(promote(scored, tmp_path))
        assert "notes.txt" not in copied and "docs/x.tmp" not in copied

    def test_the_source_is_left_alone(self, scored: Path, tmp_path: Path) -> None:
        before = tree(scored)
        promote(scored, tmp_path)
        assert tree(scored) == before


class TestRefusals:
    def refused(self, scored: Path, tmp_path: Path, match: str, **kwargs: bool) -> None:
        with pytest.raises(promote_run.Refused, match=match):
            promote(scored, tmp_path, **kwargs)
        assert not (tmp_path / "results").exists()  # nothing was written

    def test_a_run_that_was_not_scored(self, scored: Path, tmp_path: Path) -> None:
        shutil.rmtree(scored / "score")
        self.refused(scored, tmp_path, "not scored yet")

    def test_a_score_folder_made_for_another_run(self, scored: Path, tmp_path: Path) -> None:
        path = scored / "score" / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["run_id"] = "another"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.refused(scored, tmp_path, "another run")

    def test_a_run_stopped_by_the_cost_cap(self, scored: Path, tmp_path: Path) -> None:
        edit_run(scored, stopped="cost_cap")
        self.refused(scored, tmp_path, "stopped")
        self.refused(scored, tmp_path, "stopped", allow_incomplete=True)  # never a result

    def test_a_document_that_is_not_ok(self, scored: Path, tmp_path: Path) -> None:
        data = json.loads((scored / "run.json").read_text(encoding="utf-8"))
        edit_run(scored, statuses={**data["statuses"], "doc_002": "failed"})
        self.refused(scored, tmp_path, "doc_002 failed")

    def test_allow_incomplete_keeps_a_run_with_a_failed_document(
        self, scored: Path, tmp_path: Path
    ) -> None:
        data = json.loads((scored / "run.json").read_text(encoding="utf-8"))
        edit_run(scored, statuses={**data["statuses"], "doc_002": "failed"})
        assert promote(scored, tmp_path, allow_incomplete=True).is_dir()

    def test_records_that_do_not_match_the_statuses(self, scored: Path, tmp_path: Path) -> None:
        (scored / "docs" / "doc_002.json").unlink()
        self.refused(scored, tmp_path, "holds 1 record")

    def test_a_folder_whose_name_is_not_the_run_id(self, scored: Path, tmp_path: Path) -> None:
        renamed = scored.with_name("other-name")
        scored.rename(renamed)
        self.refused(renamed, tmp_path, "names the run")

    def test_an_existing_result_is_never_overwritten(self, scored: Path, tmp_path: Path) -> None:
        promote(scored, tmp_path)
        with pytest.raises(promote_run.Refused, match="already exists"):
            promote(scored, tmp_path)

    def test_a_file_holding_something_shaped_like_a_key(self, scored: Path, tmp_path: Path) -> None:
        record = scored / "docs" / "doc_001.json"
        key = "sk-" + "a1B2" * 8  # built at runtime: this file stays clean for the scanner
        record.write_text(record.read_text(encoding="utf-8") + f'\n"raw_text": "{key}"\n', "utf-8")
        with pytest.raises(promote_run.Refused, match=r"docs/doc_001\.json:\d+: .*key") as caught:
            promote(scored, tmp_path)
        assert key not in str(caught.value)  # the report never repeats the value
        assert not (tmp_path / "results").exists()

    def test_run_json_that_is_not_a_run_record(self, scored: Path, tmp_path: Path) -> None:
        (scored / "run.json").write_text("{}", encoding="utf-8")
        self.refused(scored, tmp_path, "not a run record")


class TestMain:
    def args(self, folder: Path, tmp_path: Path, *extra: str) -> list[str]:
        return [str(folder), "--results-dir", str(tmp_path / "results"), *extra]

    def test_exit_codes(
        self, scored: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert promote_run.main(self.args(scored, tmp_path)) == 0
        assert "promoted test-run: " in capsys.readouterr().out
        assert promote_run.main(self.args(scored, tmp_path)) == 1  # already there
        assert "refused:" in capsys.readouterr().err
        assert promote_run.main(self.args(tmp_path / "nope", tmp_path)) == 2
