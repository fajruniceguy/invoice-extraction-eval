from datetime import date
from decimal import Decimal

import pytest

from docextract.extract.schema import SCALAR_FIELD_NAMES, DocumentExtraction, Field, LineItem
from evalkit.groundtruth import GroundTruth, GTLineItem
from evalkit.metrics import (
    FieldMetrics,
    RowsScore,
    score_document,
    score_field,
    score_rows,
)


def extracted(value: object) -> Field:
    return Field(value=value, confidence=0.9, status="extracted", evidence="quote", page=1)


def absent() -> Field:
    return Field(value=None, confidence=0.9, status="absent")


def abstained() -> Field:
    return Field(
        value=None,
        confidence=0.2,
        status="abstained",
        reasons=["confidence_below_threshold"],
    )


def document(**overrides: object) -> DocumentExtraction:
    fields: dict[str, object] = {name: absent() for name in SCALAR_FIELD_NAMES}
    fields.update(overrides)
    rows = fields.pop("line_items", [])
    return DocumentExtraction(line_items=rows, **fields)


def row(
    description: str,
    quantity: str,
    unit_price: str,
    amount: str,
) -> tuple[GTLineItem, LineItem]:
    expected = GTLineItem(
        description=description,
        quantity=Decimal(quantity),
        unit_price=Decimal(unit_price),
        amount=Decimal(amount),
    )
    predicted = LineItem(
        description=description,
        quantity=Decimal(quantity),
        unit_price=Decimal(unit_price),
        amount=Decimal(amount),
        status="extracted",
        confidence=1.0,
    )
    return expected, predicted


class TestScoreField:
    @pytest.mark.parametrize(
        ("expected", "prediction", "outcome", "error"),
        [
            (Decimal("10"), extracted(Decimal("10")), "correct", None),
            (Decimal("10"), extracted(Decimal("11")), "value_wrong", "numeric_mismatch"),
            (Decimal("10"), absent(), "should_be_value", "missing_value"),
            (Decimal("10"), abstained(), "abstained", "abstained_on_value"),
            (Decimal("10"), None, "no_prediction", "no_prediction"),
            (None, absent(), "correct", None),
            (None, extracted(Decimal("10")), "should_be_absent", "hallucinated_value"),
            (None, abstained(), "abstained", "abstained_on_absent"),
            (None, None, "no_prediction", "no_prediction"),
        ],
    )
    def test_the_outcome_matrix(
        self, expected: object, prediction: object, outcome: str, error: str | None
    ) -> None:
        score = score_field("total", expected, prediction)
        assert (score.outcome, score.error_class) == (outcome, error)

    def test_a_correct_value_records_how_it_matched(self) -> None:
        assert score_field("total", Decimal("10"), extracted(Decimal("10.00"))).how == "numeric"
        assert score_field("vendor_name", "PT. Acme", extracted("pt acme")).how == "normalized"

    def test_a_near_miss_name_is_classified_as_such(self) -> None:
        score = score_field("vendor_name", "acme corporation", extracted("acme corporat"))
        assert (score.outcome, score.error_class) == ("value_wrong", "name_near_miss")
        assert score.similarity is not None

    def test_a_wrong_date_is_a_date_mismatch(self) -> None:
        score = score_field("document_date", date(2024, 3, 5), extracted(date(2024, 3, 6)))
        assert score.error_class == "date_mismatch"

    def test_the_predicted_status_is_reported(self) -> None:
        assert score_field("total", Decimal("10"), abstained()).predicted_status == "abstained"
        assert score_field("total", Decimal("10"), None).predicted_status is None


class TestFieldMetrics:
    def test_counts_and_ratios(self) -> None:
        metrics = FieldMetrics.from_outcomes(
            ["correct", "correct", "value_wrong", "abstained", "no_prediction"]
        )
        assert metrics.labeled == 5
        assert metrics.answered == 3
        assert metrics.accuracy == pytest.approx(2 / 5)  # abstentions and failures are errors
        assert metrics.selective_accuracy == pytest.approx(2 / 3)
        assert metrics.coverage == pytest.approx(3 / 5)
        assert metrics.abstention_rate == pytest.approx(1 / 5)
        assert metrics.failure_rate == pytest.approx(1 / 5)

    def test_a_ratio_over_nothing_is_one(self) -> None:
        empty = FieldMetrics.from_outcomes([])
        assert (empty.accuracy, empty.selective_accuracy, empty.coverage) == (1.0, 1.0, 1.0)

    def test_merging_sums_the_buckets(self) -> None:
        merged = FieldMetrics.merge(
            [
                FieldMetrics.from_outcomes(["correct", "abstained"]),
                FieldMetrics.from_outcomes(["correct", "value_wrong", "no_prediction"]),
            ]
        )
        assert (merged.labeled, merged.correct, merged.abstained) == (5, 2, 1)
        assert merged.accuracy == pytest.approx(2 / 5)


class TestScoreRows:
    def test_an_unlabeled_row_list_is_not_scored(self) -> None:
        assert score_rows("__unlabeled__", []).labeled is False

    def test_a_document_with_no_itemization_is_correct_when_nothing_is_predicted(self) -> None:
        score = score_rows(None, [])
        assert (score.labeled, score.no_itemization, score.spurious) == (True, True, 0)

    def test_rows_against_a_labeled_no_itemization_are_spurious(self) -> None:
        score = score_rows(None, [LineItem(description="x", status="extracted", confidence=1.0)])
        assert (score.spurious, score.errors) == (1, ("row_spurious",))

    def test_a_document_that_was_not_run_loses_every_row(self) -> None:
        expected, _ = row("Item", "1", "1000", "1000")
        score = score_rows([expected], None)
        assert (score.missed, score.errors) == (1, ("row_missed",))
        assert score.row_recall == 0.0

    def test_matched_rows_are_scored_cell_by_cell(self) -> None:
        first = row("Item 00", "1", "1000", "1000")
        second = row("Item 01", "2", "1000", "2000")
        wrong = second[1].model_copy(update={"amount": Decimal("2500")})
        score = score_rows([first[0], second[0]], [first[1], wrong])
        assert (score.matched, score.missed, score.spurious) == (2, 0, 0)
        assert (score.perfect, score.cells_compared, score.cells_correct) == (1, 8, 7)
        assert score.row_precision == 1.0
        assert score.row_recall == 1.0
        assert score.row_f1 == 1.0
        assert score.row_accuracy == 0.5
        assert score.cell_accuracy == pytest.approx(7 / 8)
        assert score.errors == ("numeric_mismatch",)

    def test_a_row_that_shares_nothing_counts_on_both_sides(self) -> None:
        expected, _ = row("Item 00", "1", "1000", "1000")
        unrelated = LineItem(description="Something else", status="extracted", confidence=1.0)
        score = score_rows([expected], [unrelated])
        assert (score.matched, score.missed, score.spurious) == (0, 1, 1)
        assert (score.row_precision, score.row_f1) == (0.0, 0.0)
        assert set(score.errors) == {"row_missed", "row_spurious"}

    def test_a_value_where_the_label_says_absent_is_a_hallucinated_cell(self) -> None:
        expected = GTLineItem(
            description="Item 00", quantity=None, unit_price=Decimal("1000"), amount=Decimal("1000")
        )
        predicted = LineItem(
            description="Item 00",
            quantity=Decimal("1"),
            unit_price=Decimal("1000"),
            amount=Decimal("1000"),
            status="extracted",
            confidence=1.0,
        )
        assert score_rows([expected], [predicted]).errors == ("hallucinated_value",)

    def test_a_cell_the_label_has_but_the_prediction_nulled_is_missing(self) -> None:
        expected = GTLineItem(
            description="Item 00",
            quantity=Decimal("1"),
            unit_price=Decimal("1000"),
            amount=Decimal("1000"),
        )
        predicted = LineItem(
            description="Item 00",
            quantity=None,
            unit_price=Decimal("1000"),
            amount=Decimal("1000"),
            status="extracted",
            confidence=1.0,
        )
        assert score_rows([expected], [predicted]).errors == ("missing_value",)

    def test_merging_sums_the_row_counts(self) -> None:
        merged = RowsScore.merge(
            [RowsScore(labeled=True, matched=1, cells_compared=4, cells_correct=4)]
        )
        assert (merged.matched, merged.cells_correct) == (1, 4)


class TestScoreDocument:
    def test_unlabeled_fields_are_skipped(self) -> None:
        label = GroundTruth(doc_id="doc_001", total=Decimal("10"))
        score = score_document(
            label, owner="doc_001", status="ok", document=document(total=extracted(Decimal("10")))
        )
        assert [(field.name, field.outcome) for field in score.fields] == [("total", "correct")]
        assert score.metrics.labeled == 1

    def test_a_document_that_was_not_run_gives_every_label_no_prediction(self) -> None:
        label = GroundTruth(doc_id="doc_001", total=Decimal("10"), vendor_name="Acme")
        score = score_document(label, owner="doc_001", status="failed", document=None)
        assert {field.outcome for field in score.fields} == {"no_prediction"}
        assert score.metrics.accuracy == 0.0
        assert score.errors == {"no_prediction": 2}

    def test_the_error_counts_cover_fields_and_rows_together(self) -> None:
        expected, predicted = row("Item 00", "1", "1000", "1000")
        wrong = predicted.model_copy(update={"amount": Decimal("2500")})
        label = GroundTruth(doc_id="doc_001", total=Decimal("10"), line_items=[expected])
        score = score_document(
            label,
            owner="doc_001",
            status="ok",
            document=document(total=extracted(Decimal("11")), line_items=[wrong]),
        )
        assert score.errors == {"numeric_mismatch": 2}
        assert score.metrics.value_wrong == 1
        assert score.rows.matched == 1
