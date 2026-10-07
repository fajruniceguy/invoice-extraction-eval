from datetime import date
from decimal import Decimal
from typing import Any

from docextract.confidence import (
    ARITHMETIC_CAP,
    CAP_BY_CODE,
    NORMALIZED_CAP,
    SHAPE_CAP,
    apply_abstention,
    apply_confidence,
    cap_for,
)
from docextract.extract.schema import CELL_NAMES, DocumentExtraction, Field, LineItem
from docextract.validate.codes import (
    COMPONENTS_NE_TOTAL,
    CONFIDENCE_BELOW_THRESHOLD,
    CURRENCY_UNKNOWN,
    DESCRIPTION_NOT_VERBATIM,
    DUE_DATE_BEFORE_DOCUMENT_DATE,
    EVIDENCE_NOT_GROUNDED,
    LINES_NE_SUBTOTAL,
    QTY_X_PRICE_NE_AMOUNT,
    REASON_CODES,
    TAX_ID_UNUSUAL_SHAPE,
)

D = Decimal


def field(
    value: Any,
    *,
    confidence: float = 0.9,
    reasons: tuple[str, ...] = (),
    status: str = "extracted",
    evidence: str = "quote",
) -> Field[Any]:
    return Field.model_validate(
        {
            "value": value,
            "status": status,
            "confidence": confidence,
            "evidence": evidence,
            "page": 1,
            "reasons": list(reasons),
        }
    )


def row(
    *,
    confidence: float = 0.9,
    reasons: tuple[str, ...] = (),
    amount: str = "9.00",
) -> LineItem:
    return LineItem.model_validate(
        {
            "description": "Widget",
            "quantity": "2",
            "unit_price": "4.50",
            "amount": amount,
            "status": "extracted",
            "confidence": confidence,
            "evidence": "2 x 4.50 = 9.00",
            "page": 1,
            "reasons": list(reasons),
        }
    )


def make_doc(**overrides: Any) -> DocumentExtraction:
    values: dict[str, Any] = {
        "vendor_name": field("PT Contoh"),
        "vendor_address": Field.absent(0.9),
        "vendor_tax_id": Field.absent(0.9),
        "buyer_name": Field.absent(0.9),
        "document_number": field("INV-1"),
        "document_date": field(date(2024, 3, 5)),
        "due_date": Field.absent(0.9),
        "currency": field("IDR"),
        "payment_method": Field.absent(0.9),
        "line_items": [row()],
        "subtotal": field(D("9.00")),
        "tax": field(D("0.99")),
        "discount": Field.absent(0.9),
        "total": field(D("9.99")),
    }
    values.update(overrides)
    return DocumentExtraction(**values)


def assert_abstained_fields_are_valid(document: DocumentExtraction) -> None:
    """The invariant the schema states, re-checked on output built by model_copy."""
    for name in DocumentExtraction.model_fields:
        if name == "line_items":
            continue
        found = getattr(document, name)
        if found.status == "abstained":
            assert found.value is None, name
            assert any(reason.strip() for reason in found.reasons), name
    for item in document.line_items:
        if item.status == "abstained":
            assert all(getattr(item, cell) is None for cell in CELL_NAMES)
            assert any(reason.strip() for reason in item.reasons)


class TestCapFor:
    def test_no_codes_no_cap(self) -> None:
        assert cap_for([]) is None

    def test_arithmetic_codes_cap_at_the_arithmetic_cap(self) -> None:
        for code in (
            QTY_X_PRICE_NE_AMOUNT,
            LINES_NE_SUBTOTAL,
            COMPONENTS_NE_TOTAL,
            EVIDENCE_NOT_GROUNDED,
            DUE_DATE_BEFORE_DOCUMENT_DATE,
        ):
            assert cap_for([code]) == ARITHMETIC_CAP

    def test_shape_codes_cap_at_the_shape_cap(self) -> None:
        for code in (
            CURRENCY_UNKNOWN,
            TAX_ID_UNUSUAL_SHAPE,
            DESCRIPTION_NOT_VERBATIM,
        ):
            assert cap_for([code]) == SHAPE_CAP

    def test_the_lowest_cap_wins(self) -> None:
        assert cap_for([TAX_ID_UNUSUAL_SHAPE, LINES_NE_SUBTOTAL]) == ARITHMETIC_CAP

    def test_the_abstention_outcome_is_not_a_cap(self) -> None:
        assert cap_for([CONFIDENCE_BELOW_THRESHOLD]) is None

    def test_every_reason_code_but_the_abstention_outcome_has_a_cap(self) -> None:
        assert set(REASON_CODES) - {CONFIDENCE_BELOW_THRESHOLD} == set(CAP_BY_CODE)


class TestApplyConfidence:
    def test_a_flagged_field_is_capped(self) -> None:
        document = make_doc(subtotal=field(D("9.00"), reasons=(LINES_NE_SUBTOTAL,)))
        assert apply_confidence(document).subtotal.confidence == ARITHMETIC_CAP

    def test_confidence_is_never_raised(self) -> None:
        document = make_doc(subtotal=field(D("9.00"), confidence=0.2, reasons=(LINES_NE_SUBTOTAL,)))
        assert apply_confidence(document).subtotal.confidence == 0.2

    def test_a_clean_document_is_untouched(self) -> None:
        document = make_doc()
        assert apply_confidence(document) == document

    def test_the_lowest_of_the_caps_applies(self) -> None:
        document = make_doc(
            total=field(D("9.99"), reasons=(TAX_ID_UNUSUAL_SHAPE, COMPONENTS_NE_TOTAL))
        )
        assert apply_confidence(document).total.confidence == ARITHMETIC_CAP

    def test_a_normalized_field_is_capped(self) -> None:
        document = make_doc(total=field(D("9.99")))
        result = apply_confidence(document, normalizations=["total.value: '9,99' -> '9.99'"])
        assert result.total.confidence == NORMALIZED_CAP

    def test_a_normalized_field_keeps_the_lower_of_the_two(self) -> None:
        document = make_doc(total=field(D("9.99"), reasons=(LINES_NE_SUBTOTAL,)))
        result = apply_confidence(document, normalizations=["total.value: '9,99' -> '9.99'"])
        assert result.total.confidence == ARITHMETIC_CAP

    def test_a_normalized_row_is_capped(self) -> None:
        document = make_doc(line_items=[row(), row()])
        result = apply_confidence(
            document, normalizations=["line_items[2].amount: '9,00' -> '9.00'"]
        )
        assert result.line_items[1].confidence == NORMALIZED_CAP
        assert result.line_items[0].confidence == 0.9  # the payload's row 1 is untouched

    def test_the_first_row_is_the_payloads_row_one(self) -> None:
        document = make_doc(line_items=[row()])
        result = apply_confidence(document, normalizations=["line_items[1].amount: 'x' -> '9.00'"])
        assert result.line_items[0].confidence == NORMALIZED_CAP

    def test_an_entry_that_is_not_a_change_is_ignored(self) -> None:
        document = make_doc()
        assert apply_confidence(document, normalizations=["not a change", ""]) == document

    def test_an_unknown_target_is_ignored(self) -> None:
        document = make_doc()
        result = apply_confidence(
            document,
            normalizations=["nonsense.value: 'a' -> 'b'", "line_items[x].amount: 'a' -> 'b'"],
        )
        assert result == document

    def test_twice_is_once(self) -> None:
        document = make_doc(
            total=field(D("9.99"), reasons=(LINES_NE_SUBTOTAL,)),
            line_items=[row(reasons=(QTY_X_PRICE_NE_AMOUNT,))],
        )
        once = apply_confidence(document, normalizations=["total.value: 'x' -> 'y'"])
        assert apply_confidence(once, normalizations=["total.value: 'x' -> 'y'"]) == once

    def test_the_input_is_never_mutated(self) -> None:
        document = make_doc(subtotal=field(D("9.00"), reasons=(LINES_NE_SUBTOTAL,)))
        apply_confidence(document)
        assert document.subtotal.confidence == 0.9


class TestApplyAbstention:
    def test_below_the_threshold_abstains_with_the_reason(self) -> None:
        document = make_doc(subtotal=field(D("9.00"), confidence=0.29))
        result = apply_abstention(document, abstain_below=0.3)
        assert result.subtotal.status == "abstained"
        assert result.subtotal.value is None
        assert result.subtotal.reasons == [CONFIDENCE_BELOW_THRESHOLD]
        assert result.subtotal.evidence == "quote"  # the quote is kept
        assert_abstained_fields_are_valid(result)

    def test_at_the_threshold_stays_extracted(self) -> None:
        document = make_doc(subtotal=field(D("9.00"), confidence=0.3))
        assert apply_abstention(document, abstain_below=0.3) == document

    def test_a_confident_field_is_untouched(self) -> None:
        assert apply_abstention(make_doc(), abstain_below=0.3) == make_doc()

    def test_the_threshold_can_be_disabled(self) -> None:
        document = make_doc(subtotal=field(D("9.00"), confidence=0.0))
        assert apply_abstention(document, abstain_below=0) == document

    def test_a_hard_code_abstains_whatever_the_confidence(self) -> None:
        document = make_doc(
            subtotal=field(D("9.00"), confidence=0.99, reasons=(EVIDENCE_NOT_GROUNDED,))
        )
        result = apply_abstention(document, abstain_below=0.3)
        assert result.subtotal.status == "abstained"
        assert result.subtotal.value is None
        assert result.subtotal.reasons == [EVIDENCE_NOT_GROUNDED]

    def test_a_hard_code_with_a_low_confidence_keeps_only_the_hard_reason(self) -> None:
        document = make_doc(
            due_date=field(
                date(2024, 1, 1), confidence=0.1, reasons=(DUE_DATE_BEFORE_DOCUMENT_DATE,)
            )
        )
        result = apply_abstention(document, abstain_below=0.3)
        assert result.due_date.reasons == [DUE_DATE_BEFORE_DOCUMENT_DATE]

    def test_an_absent_field_is_never_flipped(self) -> None:
        document = make_doc(vendor_address=Field.absent(0.0))
        result = apply_abstention(document, abstain_below=0.3)
        assert result.vendor_address.status == "absent"
        assert result.vendor_address.value is None

    def test_an_abstained_field_is_left_alone(self) -> None:
        document = make_doc(vendor_tax_id=Field.abstain("redacted", confidence=0.0))
        assert apply_abstention(document, abstain_below=0.3) == document

    def test_a_low_confidence_row_is_abstained_with_all_cells_null(self) -> None:
        document = make_doc(line_items=[row(confidence=0.29)])
        result = apply_abstention(document, abstain_below=0.3)
        item = result.line_items[0]
        assert item.status == "abstained"
        assert all(getattr(item, cell) is None for cell in CELL_NAMES)
        assert item.reasons == [CONFIDENCE_BELOW_THRESHOLD]
        assert item.evidence == "2 x 4.50 = 9.00"
        assert_abstained_fields_are_valid(result)

    def test_an_ungrounded_row_is_abstained_and_keeps_its_flags(self) -> None:
        flagged = row(confidence=0.95, reasons=(EVIDENCE_NOT_GROUNDED,)).model_copy(
            update={"flagged_cells": ["amount"]}
        )
        result = apply_abstention(make_doc(line_items=[flagged]), abstain_below=0.3)
        assert result.line_items[0].status == "abstained"
        assert result.line_items[0].reasons == [EVIDENCE_NOT_GROUNDED]
        assert result.line_items[0].flagged_cells == ["amount"]

    def test_a_confident_row_is_untouched(self) -> None:
        document = make_doc(line_items=[row()])
        assert apply_abstention(document, abstain_below=0.3) == document

    def test_twice_is_once(self) -> None:
        document = make_doc(subtotal=field(D("9.00"), confidence=0.1))
        once = apply_abstention(document, abstain_below=0.3)
        assert apply_abstention(once, abstain_below=0.3) == once

    def test_the_input_is_never_mutated(self) -> None:
        document = make_doc(subtotal=field(D("9.00"), confidence=0.1))
        apply_abstention(document, abstain_below=0.3)
        assert document.subtotal.status == "extracted"
        assert document.subtotal.value == D("9.00")
