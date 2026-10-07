from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from docextract.extract.schema import DocumentExtraction, Field, LineItem
from docextract.validate.arithmetic import (
    MISMATCH_CONFIDENCE_CAP,
    summed_amounts,
    validate_document_totals,
)
from docextract.validate.codes import COMPONENTS_NE_TOTAL, LINES_NE_SUBTOTAL, REASON_CODES

D = Decimal


def item(
    quantity: str | None = "2",
    unit_price: str | None = "4.50",
    amount: str | None = "9.00",
    **extra: Any,
) -> LineItem:
    values: dict[str, Any] = {
        "description": "Widget",
        "quantity": quantity,
        "unit_price": unit_price,
        "amount": amount,
        "status": "extracted",
        "confidence": 0.9,
        **extra,
    }
    return LineItem.model_validate(values)


def extracted(value: str, **extra: Any) -> Field[Decimal]:
    values: dict[str, Any] = {
        "value": D(value),
        "status": "extracted",
        "confidence": 0.9,
        "evidence": value,
        "page": 1,
        **extra,
    }
    return Field.model_validate(values)


def absent(**extra: Any) -> Field[Decimal]:
    return Field.absent(0.9, **extra)


def abstained(**extra: Any) -> Field[Decimal]:
    return Field.abstain("not confident", confidence=0.4, **extra)


def make_doc(
    subtotal: Field[Decimal] | None = None,
    tax: Field[Decimal] | None = None,
    discount: Field[Decimal] | None = None,
    total: Field[Decimal] | None = None,
    rows: list[LineItem] | None = None,
) -> DocumentExtraction:
    """A consistent document: one row of 9.00, subtotal 9.00 + tax 0.99 = total 9.99."""
    return DocumentExtraction(
        vendor_name=Field(value="PT Contoh", status="extracted", confidence=0.9),
        vendor_address=absent(),
        vendor_tax_id=absent(),
        buyer_name=absent(),
        document_number=Field(value="INV-1", status="extracted", confidence=0.9),
        document_date=Field(value=date(2024, 3, 5), status="extracted", confidence=0.9),
        due_date=absent(),
        currency=Field(value="IDR", status="extracted", confidence=0.9),
        payment_method=absent(),
        line_items=rows if rows is not None else [item()],
        subtotal=subtotal if subtotal is not None else extracted("9.00"),
        tax=tax if tax is not None else extracted("0.99"),
        discount=discount if discount is not None else absent(),
        total=total if total is not None else extracted("9.99"),
    )


def reasons_of(document: DocumentExtraction, name: str) -> list[str]:
    return getattr(document, name).reasons


def validator_codes(document: DocumentExtraction) -> set[str]:
    """The reason CODES anywhere on the summary fields and rows (a free-text note the model put
    on a field it abstained is not a validator code)."""
    summaries = (document.subtotal, document.tax, document.discount, document.total)
    found = {reason for field in summaries for reason in field.reasons if reason in REASON_CODES}
    for row in document.line_items:
        found.update(reason for reason in row.reasons if reason in REASON_CODES)
    return found


class TestSummedAmounts:
    def test_sums_the_row_amounts(self) -> None:
        assert summed_amounts(make_doc()) == D("9.00")

    def test_two_rows(self) -> None:
        rows = [item(amount="100.00"), item(amount="250.50")]
        assert summed_amounts(make_doc(rows=rows)) == D("350.50")

    def test_no_rows_cannot_be_summed(self) -> None:
        assert summed_amounts(make_doc(rows=[])) is None

    def test_a_row_without_an_amount_cannot_be_summed(self) -> None:
        rows = [item(amount="9.00"), item(quantity=None, amount=None)]
        assert summed_amounts(make_doc(rows=rows)) is None


class TestLinesVsSubtotal:
    def test_consistent_document_is_untouched(self) -> None:
        document = make_doc()
        result = validate_document_totals(document)
        assert validator_codes(result) == set()
        assert result.subtotal.confidence == pytest.approx(0.9)
        assert result == document

    def test_sum_ne_subtotal_is_flagged_on_the_subtotal(self) -> None:
        result = validate_document_totals(
            make_doc(subtotal=extracted("10.00"), total=extracted("10.99"))
        )
        assert reasons_of(result, "subtotal") == [LINES_NE_SUBTOTAL]
        assert result.subtotal.status == "extracted"
        assert result.subtotal.value == D("10.00")  # never rewritten
        assert result.subtotal.confidence == pytest.approx(MISMATCH_CONFIDENCE_CAP)

    def test_within_the_absolute_tolerance(self) -> None:
        result = validate_document_totals(make_doc(subtotal=extracted("9.01")))
        assert LINES_NE_SUBTOTAL not in reasons_of(result, "subtotal")
        assert validator_codes(result) == set()

    def test_at_the_absolute_tolerance_floor_small_amounts(self) -> None:
        # below amount 2 the 0.01 absolute floor, not the 0.5% relative part, decides
        rows = [item(amount="1.00")]
        document = make_doc(rows=rows, subtotal=extracted("1.01"), total=extracted("2.00"))
        assert validator_codes(validate_document_totals(document)) == set()

    def test_just_beyond_the_absolute_tolerance_floor(self) -> None:
        rows = [item(amount="1.00")]
        document = make_doc(rows=rows, subtotal=extracted("1.02"), total=extracted("2.01"))
        result = validate_document_totals(document)
        assert reasons_of(result, "subtotal") == [LINES_NE_SUBTOTAL]

    def test_within_the_relative_tolerance_rupiah_rounding(self) -> None:
        rows = [item(quantity=None, unit_price=None, amount="9116.8")]
        document = make_doc(rows=rows, subtotal=extracted("9117"), total=extracted("9117.99"))
        result = validate_document_totals(document)
        assert validator_codes(result) == set()

    def test_beyond_the_relative_tolerance(self) -> None:
        rows = [item(quantity=None, unit_price=None, amount="1000")]
        document = make_doc(rows=rows, subtotal=extracted("1010"), total=extracted("1010.99"))
        result = validate_document_totals(document)
        assert reasons_of(result, "subtotal") == [LINES_NE_SUBTOTAL]

    def test_skipped_without_rows(self) -> None:
        document = make_doc(rows=[], subtotal=extracted("10.00"), total=extracted("10.99"))
        result = validate_document_totals(document)
        assert LINES_NE_SUBTOTAL not in reasons_of(result, "subtotal")
        assert validator_codes(result) == set()

    def test_skipped_when_a_row_has_no_amount(self) -> None:
        rows = [item(amount="9.00"), item(quantity=None, amount=None)]
        document = make_doc(rows=rows, subtotal=extracted("10.00"), total=extracted("10.99"))
        result = validate_document_totals(document)
        assert validator_codes(result) == set()

    def test_skipped_when_the_subtotal_is_abstained(self) -> None:
        result = validate_document_totals(make_doc(subtotal=abstained()))
        assert LINES_NE_SUBTOTAL not in reasons_of(result, "subtotal")


class TestComponentsVsTotal:
    def test_consistent_document_is_untouched(self) -> None:
        assert validator_codes(validate_document_totals(make_doc())) == set()

    def test_total_ne_components_flags_every_extracted_participant(self) -> None:
        result = validate_document_totals(make_doc(total=extracted("10.99")))
        assert reasons_of(result, "subtotal") == [COMPONENTS_NE_TOTAL]
        assert reasons_of(result, "tax") == [COMPONENTS_NE_TOTAL]
        assert reasons_of(result, "total") == [COMPONENTS_NE_TOTAL]
        assert reasons_of(result, "discount") == []
        for name in ("subtotal", "tax", "total"):
            field = getattr(result, name)
            assert field.status == "extracted"
            assert field.confidence == pytest.approx(MISMATCH_CONFIDENCE_CAP)
        assert result.total.value == D("10.99")  # flagged, never rewritten

    def test_discount_participates_with_a_minus(self) -> None:
        rows = [item(amount="100.00")]
        document = make_doc(
            subtotal=extracted("100.00"),
            tax=extracted("11.00"),
            discount=extracted("10.00"),
            total=extracted("101.00"),
            rows=rows,
        )
        assert validator_codes(validate_document_totals(document)) == set()

    def test_wrong_discount_is_flagged(self) -> None:
        rows = [item(amount="100.00")]
        document = make_doc(
            subtotal=extracted("100.00"),
            tax=extracted("11.00"),
            discount=extracted("10.00"),
            total=extracted("102.00"),
            rows=rows,
        )
        result = validate_document_totals(document)
        for name in ("subtotal", "tax", "discount", "total"):
            assert reasons_of(result, name) == [COMPONENTS_NE_TOTAL]

    def test_absent_tax_counts_as_zero(self) -> None:
        document = make_doc(subtotal=extracted("9.00"), tax=absent(), total=extracted("9.00"))
        assert validator_codes(validate_document_totals(document)) == set()

    def test_components_at_the_absolute_tolerance_floor(self) -> None:
        rows = [item(amount="1.00")]
        document = make_doc(
            rows=rows, subtotal=extracted("1.00"), tax=absent(), total=extracted("1.01")
        )
        assert validator_codes(validate_document_totals(document)) == set()

    def test_components_just_beyond_the_floor(self) -> None:
        rows = [item(amount="1.00")]
        document = make_doc(
            rows=rows, subtotal=extracted("1.00"), tax=absent(), total=extracted("1.02")
        )
        result = validate_document_totals(document)
        assert reasons_of(result, "total") == [COMPONENTS_NE_TOTAL]

    def test_absent_tax_with_a_wrong_total_is_flagged(self) -> None:
        document = make_doc(subtotal=extracted("9.00"), tax=absent(), total=extracted("9.50"))
        result = validate_document_totals(document)
        assert reasons_of(result, "subtotal") == [COMPONENTS_NE_TOTAL]
        assert reasons_of(result, "total") == [COMPONENTS_NE_TOTAL]
        assert reasons_of(result, "tax") == []

    def test_absent_subtotal_falls_back_to_the_summed_rows(self) -> None:
        rows = [item(amount="100.00")]
        document = make_doc(
            subtotal=absent(),
            tax=extracted("11.00"),
            rows=rows,
            total=extracted("111.00"),
        )
        assert validator_codes(validate_document_totals(document)) == set()

    def test_abstained_subtotal_falls_back_to_the_summed_rows_too(self) -> None:
        rows = [item(amount="100.00")]
        document = make_doc(
            subtotal=abstained(),
            tax=extracted("11.00"),
            rows=rows,
            total=extracted("111.00"),
        )
        assert validator_codes(validate_document_totals(document)) == set()

    def test_skipped_when_tax_is_abstained(self) -> None:
        document = make_doc(tax=abstained(), total=extracted("999.00"))
        assert validator_codes(validate_document_totals(document)) == set()

    def test_skipped_when_discount_is_abstained(self) -> None:
        document = make_doc(discount=abstained(), total=extracted("999.00"))
        assert validator_codes(validate_document_totals(document)) == set()

    def test_skipped_without_an_extracted_total(self) -> None:
        for total in (absent(), abstained()):
            document = make_doc(total=total, tax=extracted("99.00"))
            assert validator_codes(validate_document_totals(document)) == set()

    def test_both_equations_can_flag_the_same_field(self) -> None:
        result = validate_document_totals(make_doc(subtotal=extracted("10.00")))
        assert set(reasons_of(result, "subtotal")) == {LINES_NE_SUBTOTAL, COMPONENTS_NE_TOTAL}
        assert result.subtotal.confidence == pytest.approx(MISMATCH_CONFIDENCE_CAP)


class TestIdempotenceAndCopies:
    def test_validating_twice_gives_the_same_result(self) -> None:
        document = make_doc(subtotal=extracted("10.00"))
        once = validate_document_totals(document)
        assert validate_document_totals(once) == once

    def test_the_input_is_never_mutated(self) -> None:
        document = make_doc(subtotal=extracted("10.00"))
        validate_document_totals(document)
        assert document.subtotal.reasons == []
        assert document.subtotal.confidence == pytest.approx(0.9)
        assert document.total.reasons == []
