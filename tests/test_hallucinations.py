"""Hallucinated values, taken apart, on a hand-built world. Every number is worked out here.

Three documents; `own` owns a label, `cp_bad` and `cp_ok` borrow it (gt_from).

  label of `own`: vendor Acme, number INV-1, subtotal 9, total 10 (4 labeled non-null fields);
                  tax and payment_method are ABSENT (2 labeled-absent fields); one row Tea with
                  no quantity (a null cell) and unit_price/amount 5.
  own     returns vendor Acme, INV-1, 9, 10 (all right); tax 1 (a HALLUCINATION); payment_method
          abstained (not a hallucination); row Tea quantity 1 (a CELL hallucination, conf 0.4).
  cp_bad  returns vendor Other, X-1, 98, 99 (0 of 4 right); tax 5 (a HALLUCINATION); payment_method
          absent (right); no rows.
  cp_ok   returns Acme, INV-1, 9, and total 11 (3 of 4 right); tax absent; payment_method absent;
          row Tea with no quantity (a null cell compared and right).

  scalar hallucinations: own.tax, cp_bad.tax = 2;   labeled-absent scalar fields: 3 docs x 2 = 6
  cell hallucinations:   own.quantity = 1;          null cells compared: own 1 + cp_ok 1 = 2
  lineage: owner 4/4 = 1.00; cp_bad 0/4 = 0.00 (gap 1.00: suspect); cp_ok 3/4 = 0.75 (gap 0.25).
"""

import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest

from docextract.extract.runner import DocRecord
from docextract.extract.schema import LineItem
from evalkit.groundtruth import GTLineItem
from evalkit.hallucinations import (
    LINEAGE_GAP,
    HallucinationError,
    Member,
    band_of,
    collect,
    is_suspect,
    review_markdown,
    to_markdown,
)
from evalkit.scorecard import score_run
from ingest_support import ROOT
from test_compare import manifest_row
from test_failures import abst, ext, none
from test_scorecard import dataset, document, label, run_dir

spec = importlib.util.spec_from_file_location(
    "analyze_hallucinations", ROOT / "scripts" / "analyze_hallucinations.py"
)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

QUOTE = "SECRET-QUOTE-FROM-A-PAGE"
D = Decimal


def tea(quantity: object, confidence: float = 0.9, **extra: object) -> LineItem:
    return LineItem(
        description="Tea",
        quantity=quantity,
        unit_price=D(5),
        amount=D(5),
        status="extracted",
        confidence=confidence,
        evidence=QUOTE,
        page=1,
        **extra,
    )


def quoted(value: object, confidence: float = 0.9, **extra: object):  # type: ignore[no-untyped-def]
    field = ext(value, confidence)
    return field.model_copy(update={"evidence": QUOTE, **extra})


def world(tmp_path: Path, *, extra: list[DocRecord] | None = None) -> Path:
    rows = [
        manifest_row("own", capture="phone_photo", doc_type="bill"),
        manifest_row("cp_bad", gt_from="own", content_group="own", capture="phone_photo"),
        manifest_row("cp_ok", gt_from="own", content_group="own", capture="phone_photo"),
    ]
    labels = [
        label(
            "own",
            vendor_name="Acme",
            document_number="INV-1",
            subtotal=D(9),
            total=D(10),
            tax=None,
            payment_method=None,
            line_items=[GTLineItem(description="Tea", quantity=None, unit_price=D(5), amount=D(5))],
        )
    ]
    dataset(tmp_path, rows, labels)
    own = document(
        vendor_name=ext("Acme"),
        document_number=ext("INV-1"),
        subtotal=ext(D(9)),
        total=ext(D(10)),
        tax=quoted(D(1), 0.8, reasons=["components_ne_total", "the model's own words"]),
        payment_method=abst(0.5, "evidence_not_grounded"),
        line_items=[tea(D(1), 0.4)],
    )
    cp_bad = document(
        vendor_name=ext("Other"),
        document_number=ext("X-1"),
        subtotal=ext(D(98)),
        total=ext(D(99)),
        tax=ext(D(5), 0.95),
        payment_method=none(),
    )
    cp_ok = document(
        vendor_name=ext("Acme"),
        document_number=ext("INV-1"),
        subtotal=ext(D(9)),
        total=ext(D(11)),
        tax=none(),
        payment_method=none(),
        line_items=[tea(None)],
    )
    records = [
        DocRecord(doc_id="own", status="ok", parsed=own, validated=own),
        DocRecord(doc_id="cp_bad", status="ok", parsed=cp_bad, validated=cp_bad),
        DocRecord(doc_id="cp_ok", status="ok", parsed=cp_ok, validated=cp_ok),
        *(extra or []),
    ]
    return run_dir(tmp_path, records, name="r1")


def report_of(tmp_path: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    run = world(tmp_path, **kwargs)  # type: ignore[arg-type]
    return collect(
        run, manifest_path=tmp_path / "data" / "manifest.csv", labels_dir=tmp_path / "ground_truth"
    )


class TestCollect:
    def test_it_finds_the_scalar_and_the_cell_cases(self, tmp_path: Path) -> None:
        report = report_of(tmp_path)
        assert [(i.doc_id, i.name) for i in report.fields] == [
            ("cp_bad", "tax"),
            ("own", "tax"),
        ]  # run order: sorted by doc id
        assert [(i.doc_id, i.name, i.row) for i in report.cells] == [("own", "quantity", 0)]

    def test_the_total_equals_the_scorers_hallucinated_value_count(self, tmp_path: Path) -> None:
        report = report_of(tmp_path)
        run = tmp_path / "outputs" / "runs" / "r1"
        card = score_run(
            run,
            manifest_path=tmp_path / "data" / "manifest.csv",
            labels_dir=tmp_path / "ground_truth",
            pages_dir=tmp_path / "data" / "pages",
        )
        assert card.errors["hallucinated_value"] == len(report.items) == 3

    def test_a_wrong_value_an_abstention_and_a_correct_absence_are_not_counted(
        self, tmp_path: Path
    ) -> None:
        report = report_of(tmp_path)
        names = {(i.doc_id, i.name) for i in report.items}
        assert ("own", "payment_method") not in names  # abstained
        assert ("cp_bad", "payment_method") not in names  # correctly absent
        assert ("cp_bad", "vendor_name") not in names  # wrong, but the label has a value
        assert ("cp_ok", "total") not in names  # wrong value
        assert ("cp_ok", "quantity") not in {(i.doc_id, i.name) for i in report.cells}

    def test_the_denominators(self, tmp_path: Path) -> None:
        report = report_of(tmp_path)
        assert report.absent_fields == 6
        assert report.null_cells == 2
        assert report.documents == 3

    def test_what_each_case_carries(self, tmp_path: Path) -> None:
        report = report_of(tmp_path)
        own_tax = next(i for i in report.fields if i.doc_id == "own")
        assert own_tax.value == "1" and own_tax.confidence == 0.8 and own_tax.page == 1
        assert own_tax.owner == "own" and own_tax.capture == "phone_photo"
        assert own_tax.doc_type == "bill" and own_tax.label_mode == "prelabel"
        assert own_tax.codes == ("components_ne_total",)
        assert own_tax.note == "the model's own words"  # the model's words, never the codes
        assert own_tax.quote == QUOTE
        cp_tax = next(i for i in report.fields if i.doc_id == "cp_bad")
        assert cp_tax.owner == "own" and cp_tax.doc_id == "cp_bad"  # a borrower reports its owner
        cell = report.cells[0]
        assert cell.value == "1" and cell.confidence == 0.4 and cell.level == "cell"

    def test_the_models_own_answer_is_compared(self, tmp_path: Path) -> None:
        assert all(i.raw_also for i in report_of(tmp_path).items)
        run = world(tmp_path / "x")
        # a record whose raw answer had NO value where the validated one has: raw_also is False
        record = DocRecord.model_validate_json(
            (run / "docs" / "cp_bad.json").read_text(encoding="utf-8")
        )
        record = record.model_copy(
            update={"parsed": record.parsed.model_copy(update={"tax": none()})}  # type: ignore[union-attr]
        )
        (run / "docs" / "cp_bad.json").write_text(record.model_dump_json(), encoding="utf-8")
        report = collect(
            run,
            manifest_path=tmp_path / "x" / "data" / "manifest.csv",
            labels_dir=tmp_path / "x" / "ground_truth",
        )
        by_doc = {i.doc_id: i.raw_also for i in report.fields}
        assert by_doc == {"own": True, "cp_bad": False}

    def test_a_document_that_did_not_run_is_skipped(self, tmp_path: Path) -> None:
        failed = DocRecord(doc_id="zzz", status="failed", error="boom")
        report = report_of(tmp_path, extra=[failed])
        assert report.documents == 3

    def test_an_unreadable_run_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(HallucinationError, match="cannot be read"):
            collect(tmp_path, manifest_path=tmp_path / "m.csv", labels_dir=tmp_path)


class TestLineage:
    def test_the_copy_that_reads_differently_is_suspect_and_the_owner_is_not(
        self, tmp_path: Path
    ) -> None:
        (group,) = report_of(tmp_path).lineage
        assert group.owner.doc_id == "own" and not group.owner.suspect
        by_id = {m.doc_id: m for m in group.members}
        assert (by_id["cp_bad"].correct, by_id["cp_bad"].labeled) == (0, 4)
        assert by_id["cp_bad"].suspect
        assert (by_id["cp_ok"].correct, by_id["cp_ok"].labeled) == (3, 4)
        assert not by_id["cp_ok"].suspect
        assert [m.doc_id for m in group.suspects] == ["cp_bad"]

    def test_what_the_run_read_for_the_identifiers(self, tmp_path: Path) -> None:
        (group,) = report_of(tmp_path).lineage
        # own, then members in id order (cp_bad, cp_ok); a field the model left out reads (absent)
        assert group.reads["document_number"] == ("INV-1", "X-1", "INV-1")
        assert group.reads["total"] == ("10", "99", "11")
        assert group.reads["document_date"] == ("(absent)", "(absent)", "(absent)")

    def test_the_rule_at_its_boundary(self) -> None:
        assert LINEAGE_GAP == 0.5
        owner = Member("o", 4, 4)
        assert is_suspect(owner, Member("m", 4, 2))  # a gap of exactly 0.5
        assert not is_suspect(owner, Member("m", 4, 3))  # 0.25
        big = Member("o", 100, 100)
        assert not is_suspect(big, Member("m", 100, 51))  # a gap of 0.49
        assert is_suspect(big, Member("m", 100, 50))

    def test_fewer_than_three_fields_is_never_suspect(self) -> None:
        assert not is_suspect(Member("o", 2, 2), Member("m", 2, 0))
        assert not is_suspect(Member("o", 4, 4), Member("m", 2, 0))
        assert not is_suspect(Member("o", 2, 2), Member("m", 4, 0))
        assert is_suspect(Member("o", 3, 3), Member("m", 3, 0))  # exactly 3 is enough

    def test_a_member_that_is_better_than_its_owner_is_not_suspect(self) -> None:
        assert not is_suspect(Member("o", 4, 0), Member("m", 4, 4))

    def test_no_group_when_nothing_borrows_a_label(self, tmp_path: Path) -> None:
        rows = [manifest_row("solo")]
        dataset(tmp_path, rows, [label("solo", tax=None)])
        doc = document(tax=ext(D(1)))
        run = run_dir(tmp_path, [DocRecord(doc_id="solo", status="ok", parsed=doc, validated=doc)])
        report = collect(
            run,
            manifest_path=tmp_path / "data" / "manifest.csv",
            labels_dir=tmp_path / "ground_truth",
        )
        assert report.lineage == () and len(report.items) == 1
        assert "No document borrows a label" in to_markdown(report)


class TestReports:
    def test_the_committed_report_has_the_numbers_and_no_quote(self, tmp_path: Path) -> None:
        text = to_markdown(report_of(tmp_path))
        assert "| scalar fields | 2 | 6 | 33.3% |" in text
        assert "| row cells (matched rows) | 1 | 2 | 50.0% |" in text
        assert "| **all** | **3** |" in text
        assert "| own | own, cp_bad | 2 |" not in text  # own: own.tax + own.quantity = 2, cp_bad 1
        assert "| own | cp_bad, own | 3 | 100.0% |" in text
        assert "**suspect**" in text
        assert "the model's own words" in text
        assert QUOTE not in text

    def test_the_confidence_bands(self) -> None:
        assert band_of(0.9) == ">= 0.90" and band_of(1.0) == ">= 0.90"
        assert band_of(0.89) == "0.70 - 0.89" and band_of(0.7) == "0.70 - 0.89"
        assert band_of(0.69) == "< 0.70" and band_of(0.0) == "< 0.70"

    def test_the_review_list_has_the_quotes_and_the_lineage_question(self, tmp_path: Path) -> None:
        text = review_markdown(report_of(tmp_path))
        assert QUOTE in text
        assert "cp_bad is scored against own's label" in text
        assert "cp_ok is scored" not in text
        assert text.count("- verdict:") == 3

    def test_notes_are_appended_as_the_reading(self, tmp_path: Path) -> None:
        text = to_markdown(report_of(tmp_path), notes="  Hand written.  ")
        assert text.rstrip().endswith("## Reading\n\nHand written.")

    def test_a_run_with_none_says_so(self, tmp_path: Path) -> None:
        rows = [manifest_row("solo")]
        dataset(tmp_path, rows, [label("solo", tax=None)])
        doc = document(tax=none())
        run = run_dir(tmp_path, [DocRecord(doc_id="solo", status="ok", parsed=doc, validated=doc)])
        report = collect(
            run,
            manifest_path=tmp_path / "data" / "manifest.csv",
            labels_dir=tmp_path / "ground_truth",
        )
        assert "No value was returned where the label says absent." in to_markdown(report)

    def test_the_report_is_reproducible(self, tmp_path: Path) -> None:
        first = to_markdown(report_of(tmp_path / "a"))
        second = to_markdown(report_of(tmp_path / "b"))
        assert first == second


class TestScript:
    def args(self, tmp_path: Path, *extra: str) -> list[str]:
        return [
            str(tmp_path / "outputs" / "runs" / "r1"),
            "--manifest",
            str(tmp_path / "data" / "manifest.csv"),
            "--gt-dir",
            str(tmp_path / "ground_truth"),
            *extra,
        ]

    def test_it_writes_the_report_and_the_review_list(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        world(tmp_path)
        out, review = tmp_path / "out" / "h.md", tmp_path / "scratch" / "review.md"
        code = script.main(self.args(tmp_path, "--out", str(out), "--review", str(review)))
        assert code == 0
        assert "2 scalar + 1 cell" in capsys.readouterr().out
        assert out.read_text(encoding="utf-8").startswith("# Hallucinated values: `r1`")
        assert QUOTE in review.read_text(encoding="utf-8")
        assert QUOTE not in out.read_text(encoding="utf-8")

    @pytest.mark.parametrize("folder", ["docs", "results", "ground_truth", "data"])
    def test_a_review_list_is_refused_inside_a_tracked_folder(
        self, tmp_path: Path, folder: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        world(tmp_path)
        target = ROOT / folder / "never-written.md"
        code = script.main(self.args(tmp_path, "--review", str(target)))
        assert code == 2 and not target.exists()
        assert "must not go inside" in capsys.readouterr().err

    def test_a_missing_run_is_exit_2(self, tmp_path: Path) -> None:
        assert script.main([str(tmp_path / "nope")]) == 2

    def test_the_notes_file_is_added(self, tmp_path: Path) -> None:
        world(tmp_path)
        notes = tmp_path / "notes.md"
        notes.write_text("My reading.", encoding="utf-8")
        out = tmp_path / "h.md"
        assert script.main(self.args(tmp_path, "--out", str(out), "--notes", str(notes))) == 0
        assert out.read_text(encoding="utf-8").rstrip().endswith("My reading.")
