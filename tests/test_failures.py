"""Failure analysis on hand-built runs: every number below is worked out by hand in the comment
above the world that produces it."""

from decimal import Decimal
from pathlib import Path

import pytest

from docextract.extract.runner import DocRecord
from docextract.extract.schema import Field, LineItem
from evalkit.failures import (
    COST,
    NO_ANSWER,
    SAVED,
    FailureError,
    FieldCase,
    collect,
    field_table,
    headline,
    outcome_table,
    reason_table,
    to_markdown,
    withheld_cases,
    wrong_cases,
)
from evalkit.groundtruth import GTLineItem
from test_compare import manifest_row
from test_scorecard import dataset, document, label, run_dir


def ext(value: object, confidence: float = 0.9) -> Field:
    return Field(value=value, confidence=confidence, status="extracted", evidence="q", page=1)


def abst(confidence: float, *reasons: str) -> Field:
    return Field(value=None, confidence=confidence, status="abstained", reasons=list(reasons))


def none(confidence: float = 0.9) -> Field:
    return Field(value=None, confidence=confidence, status="absent")


def row(description: str, amount: int, *, status: str = "extracted") -> LineItem:
    if status == "abstained":
        return LineItem(status="abstained", confidence=0.5, reasons=["evidence_not_grounded"])
    return LineItem(
        description=description,
        quantity=Decimal(1),
        unit_price=Decimal(amount),
        amount=Decimal(amount),
        status="extracted",
        confidence=0.9,
    )


def gt_row(description: str, amount: int) -> GTLineItem:
    return GTLineItem(
        description=description,
        quantity=Decimal(1),
        unit_price=Decimal(amount),
        amount=Decimal(amount),
    )


GROUNDING = "evidence_not_grounded"
THRESHOLD = "confidence_below_threshold"

# doc_a labels: vendor_name Acme, total 10, subtotal 9, tax absent, document_number INV-1
#   field          raw says           returned            returned / raw outcome
#   vendor_name    Acme               Acme                correct / correct
#   total          10                 abstained [ground]  abstained / correct        (cost)
#   subtotal       9                  abstained [thresh]  abstained / correct        (cost)
#   tax            absent             absent              correct / correct
#   document_no    INV-9              INV-9               value_wrong / value_wrong
# doc_b labels: vendor_name Beta, total absent, subtotal 7
#   vendor_name    Gamma              abstained [ground]  abstained / value_wrong    (saved)
#   total          5 (should be none) abstained [ground]  abstained / should_be_absent (saved)
#   subtotal       model abstained    abstained           abstained / abstained      (no answer)
# rows: doc_a labels 2 rows; the raw rows are both right; the second is returned abstained, so
#   returned: matched 1, missed 1, spurious 1, 1 abstained row; raw: matched 2, missed 0, spurious 0
# 8 labeled fields; 5 withheld; 2 returned right; 4 right had the two `cost` fields been kept.
LABELED = 8


def build(tmp_path: Path) -> Path:
    rows = [manifest_row("doc_a"), manifest_row("doc_b")]
    labels = [
        label(
            "doc_a",
            vendor_name="Acme",
            total=Decimal("10"),
            subtotal=Decimal("9"),
            tax=None,
            document_number="INV-1",
            line_items=[gt_row("Tea", 5), gt_row("Cake", 5)],
        ),
        label("doc_b", vendor_name="Beta", total=None, subtotal=Decimal("7")),
    ]
    dataset(tmp_path, rows, labels)
    doc_a_raw = document(
        vendor_name=ext("Acme", 0.95),
        total=ext(Decimal("10"), 0.9),
        subtotal=ext(Decimal("9"), 0.8),
        tax=none(),
        document_number=ext("INV-9", 0.85),
        line_items=[row("Tea", 5), row("Cake", 5)],
    )
    doc_a_returned = document(
        vendor_name=ext("Acme", 0.95),
        total=abst(0.5, GROUNDING),
        subtotal=abst(0.5, THRESHOLD),
        tax=none(),
        document_number=ext("INV-9", 0.85),
        line_items=[row("Tea", 5), row("", 0, status="abstained")],
    )
    doc_b_raw = document(
        vendor_name=ext("Gamma", 0.9),
        total=ext(Decimal("5"), 0.7),
        subtotal=abst(0.4, "the model could not read it"),
    )
    doc_b_returned = document(
        vendor_name=abst(0.5, GROUNDING),
        total=abst(0.5, GROUNDING),
        subtotal=abst(0.4, "the model could not read it"),
    )
    records = [
        DocRecord(doc_id="doc_a", status="ok", parsed=doc_a_raw, validated=doc_a_returned),
        DocRecord(doc_id="doc_b", status="ok", parsed=doc_b_raw, validated=doc_b_returned),
    ]
    return run_dir(tmp_path, records, name="r1")


def cases_of(tmp_path: Path) -> list[FieldCase]:
    run = build(tmp_path)
    return collect(
        run, manifest_path=tmp_path / "data" / "manifest.csv", labels_dir=tmp_path / "ground_truth"
    ).cases


class TestCollect:
    def test_every_labeled_field_becomes_a_case_with_both_outcomes(self, tmp_path: Path) -> None:
        cases = cases_of(tmp_path)
        assert len(cases) == LABELED
        by = {(c.doc_id, c.name): c for c in cases}
        total = by[("doc_a", "total")]
        assert (total.returned.outcome, total.raw.outcome) == ("abstained", "correct")
        assert total.confidence == 0.5 and total.raw_confidence == 0.9
        assert total.codes == (GROUNDING,)

    def test_unlabeled_fields_are_not_cases(self, tmp_path: Path) -> None:
        names = {c.name for c in cases_of(tmp_path)}
        assert names == {"vendor_name", "total", "subtotal", "tax", "document_number"}

    def test_a_failed_document_is_listed_not_analysed(self, tmp_path: Path) -> None:
        rows = [manifest_row("doc_a"), manifest_row("doc_b")]
        dataset(tmp_path, rows, [label("doc_a", total=Decimal("1")), label("doc_b", total=None)])
        records = [
            DocRecord(
                doc_id="doc_a", status="ok", parsed=document(), validated=document(total=none())
            ),
            DocRecord(doc_id="doc_b", status="failed", error="HTTP 500"),
        ]
        run = run_dir(tmp_path, records, name="r2")
        result = collect(
            run,
            manifest_path=tmp_path / "data" / "manifest.csv",
            labels_dir=tmp_path / "ground_truth",
        )
        assert result.not_ok == ["doc_b"] and {c.doc_id for c in result.cases} == {"doc_a"}

    def test_a_missing_record_is_an_error(self, tmp_path: Path) -> None:
        run = build(tmp_path)
        (run / "docs" / "doc_b.json").unlink()
        with pytest.raises(FailureError, match="cannot be read"):
            collect(
                run,
                manifest_path=tmp_path / "data" / "manifest.csv",
                labels_dir=tmp_path / "ground_truth",
            )

    def test_a_folder_without_run_json_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FailureError, match="run.json"):
            collect(tmp_path, manifest_path=tmp_path / "m.csv", labels_dir=tmp_path)


class TestWithheld:
    def test_reason_and_effect_of_each_withheld_field(self, tmp_path: Path) -> None:
        withheld = {
            (c.doc_id, c.name): (c.reason, c.effect) for c in withheld_cases(cases_of(tmp_path))
        }
        assert withheld == {
            ("doc_a", "total"): (GROUNDING, COST),
            ("doc_a", "subtotal"): (THRESHOLD, COST),
            ("doc_b", "vendor_name"): (GROUNDING, SAVED),
            ("doc_b", "total"): (GROUNDING, SAVED),
            ("doc_b", "subtotal"): ("model_abstained", NO_ANSWER),
        }

    def test_the_reason_table_counts_cost_saved_and_no_answer(self, tmp_path: Path) -> None:
        assert reason_table(cases_of(tmp_path)) == [
            (GROUNDING, 3, 1, 2, 0),
            (THRESHOLD, 1, 1, 0, 0),
            ("model_abstained", 1, 0, 0, 1),
        ]

    def test_the_outcome_cross_table(self, tmp_path: Path) -> None:
        table = {(a, b): n for a, b, n in outcome_table(cases_of(tmp_path))}
        assert table == {
            ("correct", "correct"): 2,
            ("abstained", "correct"): 2,
            ("value_wrong", "value_wrong"): 1,
            ("abstained", "value_wrong"): 1,
            ("abstained", "should_be_absent"): 1,
            ("abstained", "abstained"): 1,
        }
        assert sum(table.values()) == LABELED

    def test_headline_counts_the_correct_values_that_were_withheld(self, tmp_path: Path) -> None:
        # returned right: vendor_name, tax; the two `cost` fields would make it 4
        assert headline(cases_of(tmp_path)) == (LABELED, 2, 4)

    def test_wrong_cases_are_the_non_abstentions_only(self, tmp_path: Path) -> None:
        assert [(c.doc_id, c.name) for c in wrong_cases(cases_of(tmp_path))] == [
            ("doc_a", "document_number")
        ]

    def test_the_field_table_splits_cost_from_other_withholding(self, tmp_path: Path) -> None:
        by = {name: rest for name, *rest in field_table(cases_of(tmp_path))}
        # labeled, correct, withheld with a right raw value, withheld otherwise, wrong
        assert by["total"] == [2, 0, 1, 1, 0]
        assert by["vendor_name"] == [2, 1, 0, 1, 0]
        assert by["document_number"] == [1, 0, 0, 0, 1]


class TestMissedValues:
    def test_a_value_returned_as_absent_is_a_wrong_case_not_an_abstention(
        self, tmp_path: Path
    ) -> None:
        dataset(tmp_path, [manifest_row("doc_a")], [label("doc_a", total=Decimal("10"))])
        record = DocRecord(
            doc_id="doc_a",
            status="ok",
            parsed=document(total=none()),
            validated=document(total=none()),
        )
        run = run_dir(tmp_path, [record], name="r3")
        result = collect(
            run,
            manifest_path=tmp_path / "data" / "manifest.csv",
            labels_dir=tmp_path / "ground_truth",
        )
        (case,) = wrong_cases(result.cases)
        assert (case.name, case.returned.outcome, case.withheld) == (
            "total",
            "should_be_value",
            False,
        )


class TestRows:
    def test_abstained_rows_pair_with_nothing_but_the_raw_rows_do(self, tmp_path: Path) -> None:
        run = build(tmp_path)
        result = collect(
            run,
            manifest_path=tmp_path / "data" / "manifest.csv",
            labels_dir=tmp_path / "ground_truth",
        )
        (counts,) = result.rows
        assert counts.doc_id == "doc_a" and counts.labeled_rows == 2
        assert (counts.matched, counts.missed, counts.spurious) == (1, 1, 1)
        assert counts.abstained_rows == 1
        assert (counts.raw_matched, counts.raw_missed, counts.raw_spurious) == (2, 0, 0)


class TestMarkdown:
    def render(self, tmp_path: Path, notes: str | None = None) -> str:
        tmp_path.mkdir(parents=True, exist_ok=True)
        run = build(tmp_path)
        result = collect(
            run,
            manifest_path=tmp_path / "data" / "manifest.csv",
            labels_dir=tmp_path / "ground_truth",
        )
        return to_markdown(result, notes=notes)

    def test_it_states_the_counterfactual_as_arithmetic_not_a_score(self, tmp_path: Path) -> None:
        text = self.render(tmp_path)
        assert "5 of 8 fields were returned as `abstained`" in text
        assert "Counterfactual, not a score: returned 2/8 correct (25.0%)" in text
        assert "it would be 4/8 (50.0%)" in text

    def test_the_withheld_table_lists_each_field_with_its_effect(self, tmp_path: Path) -> None:
        text = self.render(tmp_path)
        assert "| doc_a | total | 10 | 10 | 0.90 | 0.50 | evidence_not_grounded | cost |" in text
        assert (
            "| doc_b | subtotal | 7 | (none) | 0.40 | 0.40 | model_abstained | no_answer |" in text
        )

    def test_the_rows_table_has_the_raw_columns(self, tmp_path: Path) -> None:
        assert "| doc_a | 2 | 1 | 1 | 1 | 1 | 2 | 0 | 0 |" in self.render(tmp_path)

    def test_notes_become_a_reading_section_and_are_optional(self, tmp_path: Path) -> None:
        assert "## Reading" not in self.render(tmp_path / "a")
        text = self.render(tmp_path / "b", notes="  Hand-written.  \n")
        assert text.endswith("## Reading\n\nHand-written.\n")

    def test_the_same_run_gives_the_same_bytes(self, tmp_path: Path) -> None:
        assert self.render(tmp_path / "a") == self.render(tmp_path / "b")
