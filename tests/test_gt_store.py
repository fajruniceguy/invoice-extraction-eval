import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from docextract.ingest.manifest import ManifestRow
from evalkit.groundtruth import SCALAR_FIELD_NAMES, UNLABELED, GroundTruth, GTLineItem
from evalkit.gt_store import (
    GroundTruthError,
    MissingGroundTruth,
    check_labels,
    dumps_label,
    format_report,
    label_path,
    load_labels,
    parse_label,
    read_label,
    resolve_label,
    to_json_dict,
    write_label,
)


def row(doc_id: str, **overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "doc_id": doc_id,
        "raw_files": f"{doc_id}.pdf",
        "content_group": doc_id,
    }
    values.update(overrides)
    if values.get("variant"):
        values.update(raw_files="", capture="synthetic_degraded")
    return ManifestRow.model_validate(values)


def label(doc_id: str = "doc_001", **overrides: Any) -> GroundTruth:
    values: dict[str, Any] = {
        "doc_id": doc_id,
        "label_mode": "blind",
        "labeled_at": "2026-10-06T09:15:00Z",
        "vendor_name": "PT Contoh",
        "document_date": "2024-03-05",
        "total": "1250000.00",
        "due_date": None,
        "line_items": [
            {"description": "Kertas", "quantity": "10", "unit_price": "45000", "amount": "450000"}
        ],
        "field_notes": {"due_date": "not printed"},
        "notes": "page 1 only",
    }
    values.update(overrides)
    return GroundTruth.model_validate(values)


@pytest.fixture
def gt(tmp_path: Path) -> Path:
    return tmp_path / "ground_truth"


# ------------------------------------------------------------------ the JSON form


class TestJson:
    def test_every_field_is_written_explicitly_in_a_fixed_order(self) -> None:
        data = to_json_dict(label())
        assert list(data) == [
            "doc_id",
            "label_mode",
            "labeled_at",
            *SCALAR_FIELD_NAMES,
            "line_items",
            "field_notes",
            "notes",
        ]
        assert data["discount"] == UNLABELED  # a decision, written out rather than left default
        assert data["due_date"] is None  # labeled absent
        assert data["total"] == "1250000.00"  # Decimal as the string it was labeled with
        assert data["document_date"] == "2024-03-05"
        assert data["labeled_at"] == "2026-10-06T09:15:00Z"
        assert data["line_items"] == [
            {"description": "Kertas", "quantity": "10", "unit_price": "45000", "amount": "450000"}
        ]

    def test_empty_provenance_is_left_out(self) -> None:
        data = to_json_dict(GroundTruth.model_validate({"doc_id": "d"}))
        assert "label_mode" not in data and "labeled_at" not in data
        assert "field_notes" not in data and "notes" not in data and "source_file" not in data
        assert data["line_items"] == UNLABELED

    def test_the_source_file_is_kept_when_there_is_one(self) -> None:
        assert (
            to_json_dict(GroundTruth.model_validate({"doc_id": "d", "source_file": "x"}))[
                "source_file"
            ]
            == "x"
        )

    @pytest.mark.parametrize("keep", ["all", "bare", "no_items", "null_items"])
    def test_it_round_trips(self, keep: str) -> None:
        original = {
            "all": label(),
            "bare": GroundTruth.model_validate({"doc_id": "doc_001"}),
            "no_items": label(line_items=UNLABELED),
            "null_items": label(line_items=None),
        }[keep]
        again = parse_label(dumps_label(original), source="x")
        assert again == original

    def test_money_keeps_its_printed_scale(self) -> None:
        text = dumps_label(label(total="1250000.00", subtotal="1250000", tax="0.10"))
        data = json.loads(text)
        assert (data["total"], data["subtotal"], data["tax"]) == ("1250000.00", "1250000", "0.10")
        assert parse_label(text, source="x").total == Decimal("1250000.00")

    def test_the_text_is_indented_utf8_with_a_trailing_newline(self) -> None:
        text = dumps_label(label(vendor_name="Toko Bapak Ñandú"))
        assert text.endswith("}\n") and "\n  " in text
        assert "Toko Bapak Ñandú" in text  # not escaped

    def test_line_item_cells_keep_their_three_states(self) -> None:
        item = GTLineItem.model_validate({"description": "x", "quantity": None})
        data = to_json_dict(label(line_items=[item.model_dump(mode="json")]))
        assert data["line_items"] == [
            {"description": "x", "quantity": None, "unit_price": UNLABELED, "amount": UNLABELED}
        ]


# ------------------------------------------------------------------ reading and writing


class TestFiles:
    def test_write_then_read(self, gt: Path) -> None:
        path = write_label(gt, label())
        assert path == label_path(gt, "doc_001") == gt / "doc_001.json"
        assert read_label(path) == label()

    def test_writing_replaces_atomically_and_leaves_no_temporary_files(self, gt: Path) -> None:
        write_label(gt, label())
        write_label(gt, label(vendor_name="Changed"))
        assert read_label(gt / "doc_001.json").vendor_name == "Changed"
        assert [p.name for p in gt.iterdir()] == ["doc_001.json"]

    def test_a_failed_write_leaves_the_old_file_alone(
        self, gt: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_label(gt, label())
        before = (gt / "doc_001.json").read_bytes()

        def boom(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("evalkit.gt_store.os.replace", boom)
        with pytest.raises(OSError, match="disk full"):
            write_label(gt, label(vendor_name="Changed"))
        assert (gt / "doc_001.json").read_bytes() == before
        assert [p.name for p in gt.iterdir()] == ["doc_001.json"]  # temp file cleaned up

    def test_read_names_the_file_in_every_error(self, gt: Path) -> None:
        gt.mkdir()
        (gt / "a.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(GroundTruthError, match=r"^a.json: not valid JSON"):
            read_label(gt / "a.json")
        (gt / "b.json").write_text(json.dumps({"doc_id": "b", "vendor_nme": "x"}), encoding="utf-8")
        with pytest.raises(GroundTruthError, match=r"^b.json: vendor_nme"):
            read_label(gt / "b.json")
        (gt / "c.json").write_bytes(b"\xff\xfe")
        with pytest.raises(GroundTruthError, match=r"^c.json: cannot be read"):
            read_label(gt / "c.json")
        with pytest.raises(GroundTruthError, match=r"^nope.json: cannot be read"):
            read_label(gt / "nope.json")

    def test_the_doc_id_inside_must_match_the_file_name(self, gt: Path) -> None:
        gt.mkdir()
        (gt / "doc_002.json").write_text(json.dumps({"doc_id": "doc_001"}), encoding="utf-8")
        with pytest.raises(GroundTruthError, match="holds doc_id 'doc_001', not 'doc_002'"):
            read_label(gt / "doc_002.json")

    def test_bad_values_are_reported_with_their_location(self, gt: Path) -> None:
        gt.mkdir()
        bad = {"doc_id": "d", "total": "12,5", "labeled_at": "2026-10-06T09:15:00"}
        (gt / "d.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(GroundTruthError) as caught:
            read_label(gt / "d.json")
        assert "total" in str(caught.value) and "labeled_at" in str(caught.value)


# ------------------------------------------------------------------ borrowed labels


class TestResolve:
    def rows(self) -> list[ManifestRow]:
        return [
            row("doc_001"),
            row("doc_002", gt_from="doc_001"),  # a second photo of the same content
            row("doc_001-deg-light", variant="light", gt_from="doc_001"),
            row("doc_003"),
        ]

    def test_a_document_resolves_to_its_own_label(self, gt: Path) -> None:
        write_label(gt, label())
        resolved = resolve_label("doc_001", self.rows(), gt)
        assert (resolved.doc_id, resolved.owner, resolved.borrowed) == ("doc_001", "doc_001", False)
        assert resolved.label == label()

    def test_a_borrower_gets_the_owners_label_under_its_own_id(self, gt: Path) -> None:
        write_label(gt, label())
        for borrower in ("doc_002", "doc_001-deg-light"):
            resolved = resolve_label(borrower, self.rows(), gt)
            assert (resolved.doc_id, resolved.owner, resolved.borrowed) == (
                borrower,
                "doc_001",
                True,
            )
            assert resolved.label.doc_id == borrower
            assert resolved.label.model_dump(exclude={"doc_id"}) == label().model_dump(
                exclude={"doc_id"}
            )
        assert read_label(gt / "doc_001.json").doc_id == "doc_001"  # the file is untouched

    def test_no_label_yet(self, gt: Path) -> None:
        with pytest.raises(MissingGroundTruth, match="doc_003: no ground truth yet$"):
            resolve_label("doc_003", self.rows(), gt)
        with pytest.raises(MissingGroundTruth, match=r"takes its labels from doc_001\)$"):
            resolve_label("doc_002", self.rows(), gt)

    def test_a_missing_label_is_a_ground_truth_error_too(self, gt: Path) -> None:
        with pytest.raises(GroundTruthError):
            resolve_label("doc_003", self.rows(), gt)

    def test_an_unknown_document(self, gt: Path) -> None:
        with pytest.raises(GroundTruthError, match="doc_999: not in the manifest") as caught:
            resolve_label("doc_999", self.rows(), gt)
        assert not isinstance(
            caught.value, MissingGroundTruth
        )  # a caller error, not "no label yet"

    def test_a_borrower_with_a_file_of_its_own_is_ambiguous(self, gt: Path) -> None:
        write_label(gt, label())
        write_label(gt, label("doc_002"))
        with pytest.raises(GroundTruthError, match="has a label file of its own but also takes"):
            resolve_label("doc_002", self.rows(), gt)

    def test_load_labels_splits_labeled_from_missing(self, gt: Path) -> None:
        write_label(gt, label())
        labels, missing = load_labels(self.rows(), gt)
        assert sorted(labels) == ["doc_001", "doc_001-deg-light", "doc_002"]
        assert missing == ["doc_003"]
        assert labels["doc_002"].borrowed and not labels["doc_001"].borrowed

    def test_load_labels_does_not_hide_a_broken_file(self, gt: Path) -> None:
        gt.mkdir()
        (gt / "doc_001.json").write_text("{", encoding="utf-8")
        with pytest.raises(GroundTruthError, match="not valid JSON"):
            load_labels(self.rows(), gt)


# ------------------------------------------------------------------ integrity


class TestCheck:
    def test_a_consistent_directory_has_no_problems(self, gt: Path) -> None:
        rows = [
            row("doc_001", label_mode="blind"),
            row("doc_002", gt_from="doc_001"),
            row("cons_001", capture="constructed"),
        ]
        write_label(gt, label())
        write_label(gt, label("cons_001", label_mode=None, labeled_at=None))
        assert check_labels(rows, gt, gt / "drafts") == []

    def test_no_directory_is_fine(self, tmp_path: Path) -> None:
        assert check_labels([row("doc_001")], tmp_path / "nope", tmp_path / "nope" / "drafts") == []

    def test_a_label_for_a_document_that_is_not_in_the_manifest(self, gt: Path) -> None:
        write_label(gt, label("doc_009"))
        assert check_labels([row("doc_001")], gt) == [
            "doc_009.json: no document 'doc_009' in the manifest"
        ]

    def test_a_borrower_must_not_have_its_own_file(self, gt: Path) -> None:
        rows = [row("doc_001"), row("doc_002", gt_from="doc_001")]
        write_label(gt, label("doc_002"))
        (problem,) = check_labels(rows, gt)
        assert "doc_002 takes its labels from doc_001" in problem

    def test_the_label_mode_in_the_file_must_match_the_manifest(self, gt: Path) -> None:
        write_label(gt, label(label_mode="prelabel"))
        (problem,) = check_labels([row("doc_001", label_mode="blind")], gt)
        assert "label_mode is 'prelabel' but the manifest says 'blind'" in problem

    def test_a_label_without_a_mode_for_a_document_that_has_one(self, gt: Path) -> None:
        write_label(gt, label(label_mode=None))
        (problem,) = check_labels([row("doc_001", label_mode="prelabel")], gt)
        assert "label_mode is None but the manifest says 'prelabel'" in problem

    def test_a_mode_in_the_file_for_a_document_that_has_none(self, gt: Path) -> None:
        write_label(gt, label(label_mode="blind"))
        (problem,) = check_labels([row("doc_001")], gt)
        assert "label_mode is 'blind' but the manifest says None" in problem

    def test_a_broken_file_is_reported_not_raised(self, gt: Path) -> None:
        gt.mkdir()
        (gt / "doc_001.json").write_text("{", encoding="utf-8")
        write_label(gt, label("doc_002", label_mode=None, labeled_at=None))
        problems = check_labels([row("doc_001"), row("doc_002")], gt)
        assert len(problems) == 1 and problems[0].startswith("doc_001.json: not valid JSON")

    def test_a_blind_document_must_never_have_a_draft(self, gt: Path) -> None:
        rows = [row("doc_001", label_mode="blind"), row("doc_002", label_mode="prelabel")]
        drafts = gt / "drafts"
        drafts.mkdir(parents=True)
        (drafts / "doc_002.json").write_text("{}", encoding="utf-8")  # fine: it is prelabel
        assert check_labels(rows, gt, drafts) == []
        (drafts / "doc_001.json").write_text("{}", encoding="utf-8")
        (problem,) = check_labels(rows, gt, drafts)
        assert problem.startswith("drafts/doc_001.json: doc_001 is a BLIND document")

    def test_drafts_are_only_checked_when_a_directory_is_given(self, gt: Path) -> None:
        drafts = gt / "drafts"
        drafts.mkdir(parents=True)
        (drafts / "doc_001.json").write_text("{}", encoding="utf-8")
        assert check_labels([row("doc_001", label_mode="blind")], gt) == []

    def test_every_problem_is_listed(self, gt: Path) -> None:
        rows = [row("doc_001", label_mode="blind"), row("doc_002", gt_from="doc_001")]
        write_label(gt, label("doc_009"))
        write_label(gt, label("doc_002"))
        write_label(gt, label("doc_001", label_mode="prelabel"))
        drafts = gt / "drafts"
        drafts.mkdir()
        (drafts / "doc_001.json").write_text("{}", encoding="utf-8")
        assert len(check_labels(rows, gt, drafts)) == 4


# ------------------------------------------------------------------ the report


class TestReport:
    def rows(self) -> list[ManifestRow]:
        return [
            row("doc_001", split="dev", label_mode="blind"),
            row("doc_002", split="dev", gt_from="doc_001"),
            row("doc_003", split="test", label_mode="prelabel"),
            row("doc_004", split="test", label_mode="prelabel"),
            row("cons_001", capture="constructed", split="test"),
        ]

    def test_it_summarizes_coverage_per_split(self, gt: Path) -> None:
        write_label(gt, label())
        write_label(gt, label("doc_003", label_mode="prelabel"))
        write_label(gt, label("cons_001", label_mode=None, labeled_at=None))
        text, problems = format_report(self.rows(), gt)
        assert problems == []
        lines = text.splitlines()
        assert lines[0] == (
            "ground truth: 3 label file(s) -> 4 document(s) labeled (1 by borrowing), "
            "1 without a label yet"
        )
        assert "  dev            2 / 2    labeled" in lines
        assert "  test           2 / 3    labeled" in lines
        assert "  hand/generated labels: blind 1, generated 1, prelabel 1" in lines
        assert lines[-1] == "problems: none"

    def test_problems_are_listed_and_counted(self, gt: Path) -> None:
        write_label(gt, label("doc_009"))
        write_label(gt, label("doc_002"))
        text, problems = format_report(self.rows(), gt)
        assert len(problems) == 2
        assert "problems: 2" in text
        assert all(f"  - {problem}" in text for problem in problems)

    def test_a_broken_file_does_not_crash_the_report(self, gt: Path) -> None:
        gt.mkdir()
        (gt / "doc_001.json").write_text("{", encoding="utf-8")
        text, problems = format_report(self.rows(), gt)
        assert len(problems) == 1 and "not valid JSON" in problems[0]
        assert "0 document(s) labeled" in text

    def test_an_empty_directory(self, tmp_path: Path) -> None:
        text, problems = format_report(self.rows(), tmp_path / "none")
        assert problems == []
        assert text.splitlines()[0] == (
            "ground truth: 0 label file(s) -> 0 document(s) labeled (0 by borrowing), "
            "5 without a label yet"
        )

    def test_rows_without_a_split_are_counted_apart(self, gt: Path) -> None:
        text, _ = format_report([row("doc_001"), row("doc_002", split="dev")], gt)
        assert "  (no split)     0 / 1    labeled" in text.splitlines()


@pytest.mark.slow
class TestCheckScript:
    def run(self, tmp_path: Path, rows: list[ManifestRow]) -> tuple[int, str, str]:
        from ingest_support import run_script, write_manifest_file

        manifest = write_manifest_file(tmp_path, rows)
        return run_script(
            "check_ground_truth.py",
            "--manifest",
            str(manifest),
            "--ground-truth-dir",
            str(tmp_path / "gt"),
        )

    def test_a_consistent_directory_exits_0(self, tmp_path: Path) -> None:
        write_label(tmp_path / "gt", label())
        code, out, _ = self.run(tmp_path, [row("doc_001", label_mode="blind")])
        assert code == 0 and "problems: none" in out

    def test_problems_exit_1(self, tmp_path: Path) -> None:
        write_label(tmp_path / "gt", label("doc_009"))
        code, out, _ = self.run(tmp_path, [row("doc_001")])
        assert code == 1 and "no document 'doc_009' in the manifest" in out

    def test_a_blind_draft_is_found_in_the_default_drafts_directory(self, tmp_path: Path) -> None:
        drafts = tmp_path / "gt" / "drafts"
        drafts.mkdir(parents=True)
        (drafts / "doc_001.json").write_text("{}", encoding="utf-8")
        code, out, _ = self.run(tmp_path, [row("doc_001", label_mode="blind")])
        assert code == 1 and "BLIND document" in out

    def test_an_unreadable_manifest_exits_2(self, tmp_path: Path) -> None:
        from ingest_support import run_script

        bad = tmp_path / "m.csv"
        bad.write_text("nonsense\n", encoding="utf-8")
        code, _, err = run_script("check_ground_truth.py", "--manifest", str(bad))
        assert code == 2 and "cannot be read" in err
