from datetime import date
from decimal import Decimal
from typing import Any

from docextract.extract.schema import CELL_NAMES, DocumentExtraction, Field, LineItem
from docextract.validate import REASON_CODES, validate_document
from docextract.validate.codes import (
    COMPONENTS_NE_TOTAL,
    CONFIDENCE_BELOW_THRESHOLD,
    DOCUMENT_DATE_IN_FUTURE,
    EVIDENCE_AMBIGUOUS,
    EVIDENCE_NOT_GROUNDED,
    LINES_NE_SUBTOTAL,
    QTY_X_PRICE_NE_AMOUNT,
)

D = Decimal
TODAY = date(2024, 3, 20)


def scalar(value: Any, evidence: str = "quote", **extra: Any) -> Field[Any]:
    return Field.model_validate(
        {
            "value": value,
            "status": "extracted",
            "confidence": 0.9,
            "evidence": evidence,
            "page": 1,
            **extra,
        }
    )


def row(**extra: Any) -> LineItem:
    return LineItem.model_validate(
        {
            "description": "Widget",
            "quantity": "2",
            "unit_price": "4.50",
            "amount": "9.00",
            "status": "extracted",
            "confidence": 0.9,
            "evidence": "2 x 4.50 = 9.00 Widget",
            "page": 1,
            **extra,
        }
    )


def make_doc(**overrides: Any) -> DocumentExtraction:
    """A consistent, grounded document: one 9.00 row, subtotal 9.00 + tax 0.99 = total 9.99."""
    values: dict[str, Any] = {
        "vendor_name": scalar("PT Contoh", "PT Contoh"),
        "vendor_address": Field.absent(0.9),
        "vendor_tax_id": Field.absent(0.9),
        "buyer_name": Field.absent(0.9),
        "document_number": scalar("INV-1", "INV-1"),
        "document_date": scalar(date(2024, 3, 5), "5 Maret 2024"),
        "due_date": Field.absent(0.9),
        "currency": scalar("IDR", "Rp"),
        "payment_method": Field.absent(0.9),
        "line_items": [row()],
        "subtotal": scalar(D("9.00"), "9.00"),
        "tax": scalar(D("0.99"), "0.99"),
        "discount": Field.absent(0.9),
        "total": scalar(D("9.99"), "9.99"),
    }
    values.update(overrides)
    return DocumentExtraction(**values)


def assert_the_dod_shape(document: DocumentExtraction) -> None:
    """Every field carries value, confidence, evidence, status and reasons."""
    for name in DocumentExtraction.model_fields:
        if name == "line_items":
            continue
        found = getattr(document, name)
        assert found.status in ("extracted", "absent", "abstained"), name
        assert 0.0 <= found.confidence <= 1.0, name
        if found.status == "extracted":
            assert found.value is not None, name
            assert found.evidence and found.page is not None, name
        else:
            assert found.value is None, name
        if found.status == "abstained":
            assert any(reason.strip() for reason in found.reasons), name
        assert set(found.reasons) <= set(REASON_CODES) or found.status == "abstained"
    for item in document.line_items:
        if item.status == "extracted":
            assert any(getattr(item, cell) is not None for cell in CELL_NAMES)
            assert item.evidence and item.page is not None
        else:
            assert all(getattr(item, cell) is None for cell in CELL_NAMES)
        if item.status == "abstained":
            assert any(reason.strip() for reason in item.reasons)


class TestTheChain:
    def test_a_clean_document_is_untouched(self) -> None:
        document = make_doc()
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        assert result == document

    def test_the_raw_output_is_never_mutated(self) -> None:
        document = make_doc(subtotal=scalar(D("12.00"), "12.00"))
        validate_document(document, abstain_below=0.3, today=TODAY)
        assert document.subtotal.value == D("12.00")
        assert document.subtotal.confidence == 0.9
        assert document.subtotal.reasons == []

    def test_an_arithmetic_mismatch_is_flagged_and_capped(self) -> None:
        # 12.00 vs the 9.00 row: lines vs subtotal, and the components no longer reach the total
        document = make_doc(subtotal=scalar(D("12.00"), "12.00"))
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        assert set(result.subtotal.reasons) == {LINES_NE_SUBTOTAL, COMPONENTS_NE_TOTAL}
        assert result.subtotal.confidence == 0.5
        assert result.subtotal.status == "extracted"

    def test_a_row_mismatch_is_flagged_and_capped(self) -> None:
        document = make_doc(line_items=[row(amount="8.00")])
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        assert QTY_X_PRICE_NE_AMOUNT in result.line_items[0].reasons
        assert result.line_items[0].confidence == 0.5

    def test_an_ungrounded_value_is_abstained(self) -> None:
        document = make_doc(vendor_name=scalar("Toko Sebelah", "PT Contoh"))
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        assert result.vendor_name.status == "abstained"
        assert result.vendor_name.value is None
        assert result.vendor_name.evidence == "PT Contoh"
        assert result.vendor_name.reasons == [EVIDENCE_NOT_GROUNDED]

    def test_a_row_ungrounded_in_one_cell_is_abstained(self) -> None:
        # the row adds up (2 x 4.50 = 9.00) but the quote says 10.00, so only the amount fails
        document = make_doc(line_items=[row(evidence="Widget 2 x 4.50 = 10.00")])
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        item = result.line_items[0]
        assert item.status == "abstained"
        assert all(getattr(item, cell) is None for cell in CELL_NAMES)
        assert item.flagged_cells == ["amount"]
        assert EVIDENCE_NOT_GROUNDED in item.reasons

    def test_an_ambiguous_quote_keeps_the_value_with_a_capped_confidence(self) -> None:
        # no currency anywhere, so "9.990" is 9990 or 9.99: the model's 9.99 is one of them
        document = make_doc(
            currency=Field.absent(0.9), total=scalar(D("9.99"), "TOTAL 9.990", confidence=0.95)
        )
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        assert result.total.status == "extracted"
        assert result.total.value == D("9.99")  # the validators never rewrite a value
        assert result.total.reasons == [EVIDENCE_AMBIGUOUS]
        assert result.total.confidence == 0.7
        assert result.total.evidence == "TOTAL 9.990"

    def test_an_ambiguous_quote_for_another_number_is_still_abstained(self) -> None:
        document = make_doc(
            currency=Field.absent(0.9), total=scalar(D("19.99"), "TOTAL 9.990", confidence=0.95)
        )
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        assert result.total.status == "abstained"
        assert EVIDENCE_NOT_GROUNDED in result.total.reasons

    def test_an_ambiguous_row_is_kept_capped_and_flagged(self) -> None:
        document = make_doc(
            currency=Field.absent(0.9),
            line_items=[row(description="Widget", evidence="Widget 2 x 4.500 = 9.000")],
        )
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        item = result.line_items[0]
        assert item.status == "extracted"
        assert item.reasons == [EVIDENCE_AMBIGUOUS]
        assert item.confidence == 0.7
        assert item.flagged_cells == ["unit_price", "amount"]

    def test_a_normalized_value_is_capped(self) -> None:
        document = make_doc(total=scalar(D("9.99"), "Rp 9,99"))
        result = validate_document(
            document,
            abstain_below=0.3,
            normalizations=["total.value: '9,99' -> '9.99'"],
            today=TODAY,
        )
        assert result.total.confidence == 0.85

    def test_the_threshold_abstains_a_low_confidence_field(self) -> None:
        document = make_doc(vendor_name=scalar("PT Contoh", "PT Contoh", confidence=0.2))
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        assert result.vendor_name.status == "abstained"
        assert result.vendor_name.reasons == [CONFIDENCE_BELOW_THRESHOLD]

    def test_today_reaches_the_future_date_check(self) -> None:
        document = make_doc(document_date=scalar(date(2024, 4, 1), "1 April 2024"))
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        assert DOCUMENT_DATE_IN_FUTURE in result.document_date.reasons
        assert result.document_date.confidence == 0.7

    def test_the_shape_invariant_holds_on_a_messy_document(self) -> None:
        document = make_doc(
            vendor_name=scalar("Toko Sebelah", "PT Contoh"),
            document_date=scalar(date(2024, 4, 1), "nonsense"),
            currency=scalar("CHF", "We charge in CHF"),
            subtotal=scalar(D("12.00"), "12.00", confidence=0.2),
            line_items=[row(amount="9.50"), row(description="Paper", evidence="2 x 4.50 = 9.00")],
        )
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        assert_the_dod_shape(result)

    def test_twice_is_once(self) -> None:
        document = make_doc(
            subtotal=scalar(D("12.00"), "12.00"),
            vendor_name=scalar("Toko Sebelah", "PT Contoh"),
            line_items=[row(amount="9.50")],
        )
        once = validate_document(document, abstain_below=0.3, today=TODAY)
        assert validate_document(document, abstain_below=0.3, today=TODAY) == once

    def test_every_reason_on_a_validated_document_is_a_code(self) -> None:
        document = make_doc(
            currency=scalar("CHF", "We charge in CHF"),
            subtotal=scalar(D("12.00"), "12.00"),
        )
        result = validate_document(document, abstain_below=0.3, today=TODAY)
        reasons = {
            reason
            for name in DocumentExtraction.model_fields
            if name != "line_items"
            for reason in getattr(result, name).reasons
        }
        assert reasons <= set(REASON_CODES)
