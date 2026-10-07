from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from docextract.extract.schema import DocumentExtraction, Field, LineItem
from docextract.validate.codes import (
    CURRENCY_CONTRADICTS_EVIDENCE,
    CURRENCY_UNKNOWN,
    DOCUMENT_DATE_IN_FUTURE,
    DUE_DATE_BEFORE_DOCUMENT_DATE,
    REASON_CODES,
    TAX_ID_IMPLAUSIBLE,
    TAX_ID_UNUSUAL_SHAPE,
)
from docextract.validate.fields import validate_fields

TODAY = date(2024, 3, 20)


def extracted_str(value: str, evidence: str | None = None, **extra: Any) -> Field[str]:
    values: dict[str, Any] = {
        "value": value,
        "status": "extracted",
        "confidence": 0.9,
        "evidence": evidence,
        "page": 1,
        **extra,
    }
    return Field.model_validate(values)


def extracted_date(value: str, evidence: str | None = None) -> Field[date]:
    return Field.model_validate(
        {
            "value": date.fromisoformat(value),
            "status": "extracted",
            "confidence": 0.9,
            "evidence": evidence,
            "page": 1,
        }
    )


def extracted_decimal(value: str) -> Any:
    from decimal import Decimal

    return Field.model_validate(
        {
            "value": Decimal(value),
            "status": "extracted",
            "confidence": 0.9,
            "evidence": value,
            "page": 1,
        }
    )


def row(amount: str = "9.00") -> LineItem:
    return LineItem.model_validate(
        {
            "description": "Widget",
            "quantity": "2",
            "unit_price": "4.50",
            "amount": amount,
            "status": "extracted",
            "confidence": 0.9,
            "evidence": "2 x 4.50 = 9.00",
            "page": 1,
        }
    )


def make_doc(
    *,
    document_date: Field[date] | None = None,
    due_date: Field[date] | None = None,
    currency: Field[str] | None = None,
    vendor_tax_id: Field[str] | None = None,
    subtotal_evidence: str | None = None,
) -> DocumentExtraction:
    if subtotal_evidence is None:
        subtotal: Any = extracted_decimal("9.00")
    else:
        subtotal = Field.model_validate(
            {
                "value": Decimal("9.00"),
                "status": "extracted",
                "confidence": 0.9,
                "evidence": subtotal_evidence,
                "page": 1,
            }
        )
    return DocumentExtraction(
        vendor_name=extracted_str("PT Contoh"),
        vendor_address=Field.absent(0.9),
        vendor_tax_id=vendor_tax_id if vendor_tax_id is not None else Field.absent(0.9),
        buyer_name=Field.absent(0.9),
        document_number=extracted_str("INV-1"),
        document_date=document_date
        if document_date is not None
        else extracted_date("2024-03-05"),
        due_date=due_date if due_date is not None else Field.absent(0.9),
        currency=currency if currency is not None else extracted_str("IDR"),
        payment_method=Field.absent(0.9),
        line_items=[row()],
        subtotal=subtotal,
        tax=extracted_decimal("0.99"),
        discount=Field.absent(0.9),
        total=extracted_decimal("9.99"),
    )


def reasons_of(document: DocumentExtraction, name: str) -> list[str]:
    return getattr(document, name).reasons


class TestDates:
    def test_due_date_before_document_date_is_a_contradiction(self) -> None:
        result = validate_fields(
            make_doc(
                document_date=extracted_date("2024-03-05"),
                due_date=extracted_date("2024-03-01"),
            ),
            today=TODAY,
        )
        assert reasons_of(result, "due_date") == [DUE_DATE_BEFORE_DOCUMENT_DATE]
        assert reasons_of(result, "document_date") == []

    def test_status_is_never_changed_here(self) -> None:
        result = validate_fields(
            make_doc(
                document_date=extracted_date("2024-03-05"),
                due_date=extracted_date("2024-03-01"),
            ),
            today=TODAY,
        )
        assert result.due_date.status == "extracted"
        assert result.due_date.value == date(2024, 3, 1)  # flagged, never rewritten
        assert result.due_date.confidence == pytest.approx(0.9)  # the formula caps, not this

    def test_due_date_after_document_date_is_clean(self) -> None:
        result = validate_fields(
            make_doc(
                document_date=extracted_date("2024-03-05"),
                due_date=extracted_date("2024-04-01"),
            ),
            today=TODAY,
        )
        assert reasons_of(result, "due_date") == []

    def test_the_same_day_is_clean(self) -> None:
        result = validate_fields(
            make_doc(
                document_date=extracted_date("2024-03-05"),
                due_date=extracted_date("2024-03-05"),
            ),
            today=TODAY,
        )
        assert reasons_of(result, "due_date") == []

    def test_skipped_without_a_due_date(self) -> None:
        for due in (Field.absent(0.9), Field.abstain("unreadable", confidence=0.4)):
            result = validate_fields(
                make_doc(document_date=extracted_date("2024-03-05"), due_date=due),
                today=TODAY,
            )
            assert DUE_DATE_BEFORE_DOCUMENT_DATE not in reasons_of(result, "due_date")

    def test_future_document_date_is_flagged(self) -> None:
        result = validate_fields(
            make_doc(document_date=extracted_date("2024-04-01")), today=TODAY
        )
        assert reasons_of(result, "document_date") == [DOCUMENT_DATE_IN_FUTURE]
        assert reasons_of(result, "due_date") == []

    def test_past_document_date_is_clean(self) -> None:
        result = validate_fields(make_doc(document_date=extracted_date("2024-03-19")), today=TODAY)
        assert reasons_of(result, "document_date") == []

    def test_today_defaults_to_the_real_today(self) -> None:
        result = validate_fields(make_doc(document_date=extracted_date("2024-03-05")))
        assert reasons_of(result, "document_date") == []

    def test_twice_is_once(self) -> None:
        document = make_doc(
            document_date=extracted_date("2024-04-01"),
            due_date=extracted_date("2024-03-01"),
        )
        once = validate_fields(document, today=TODAY)
        assert validate_fields(once, today=TODAY) == once


class TestCurrency:
    def test_a_known_currency_with_matching_evidence_is_clean(self) -> None:
        result = validate_fields(
            make_doc(currency=extracted_str("IDR"), subtotal_evidence="Rp 9.000"),
            today=TODAY,
        )
        assert reasons_of(result, "currency") == []

    def test_lowercase_code_is_normalized(self) -> None:
        result = validate_fields(make_doc(currency=extracted_str("idr")), today=TODAY)
        assert reasons_of(result, "currency") == []

    def test_a_currency_the_normalizer_cannot_read_is_unknown(self) -> None:
        result = validate_fields(make_doc(currency=extracted_str("CHF")), today=TODAY)
        assert reasons_of(result, "currency") == [CURRENCY_UNKNOWN]

    def test_a_marker_that_is_not_a_code_is_unknown(self) -> None:
        result = validate_fields(make_doc(currency=extracted_str("Rupiah")), today=TODAY)
        assert reasons_of(result, "currency") == [CURRENCY_UNKNOWN]

    def test_a_contradicting_marker_in_the_evidence_is_flagged(self) -> None:
        result = validate_fields(
            make_doc(currency=extracted_str("USD"), subtotal_evidence="Rp 9.000"),
            today=TODAY,
        )
        assert reasons_of(result, "currency") == [CURRENCY_CONTRADICTS_EVIDENCE]

    def test_a_matching_marker_is_clean(self) -> None:
        result = validate_fields(
            make_doc(currency=extracted_str("USD"), subtotal_evidence="$9.00"),
            today=TODAY,
        )
        assert reasons_of(result, "currency") == []

    def test_evidence_without_a_marker_is_clean(self) -> None:
        result = validate_fields(
            make_doc(currency=extracted_str("USD"), subtotal_evidence="Subtotal 9.00"),
            today=TODAY,
        )
        assert reasons_of(result, "currency") == []

    def test_skipped_without_an_extracted_currency(self) -> None:
        for currency in (Field.absent(0.9), Field.abstain("unreadable", confidence=0.4)):
            result = validate_fields(
                make_doc(currency=currency, subtotal_evidence="Rp 9.000"), today=TODAY
            )
            assert CURRENCY_UNKNOWN not in reasons_of(result, "currency")
            assert CURRENCY_CONTRADICTS_EVIDENCE not in reasons_of(result, "currency")

    def test_unknown_and_contradicted_can_combine(self) -> None:
        result = validate_fields(
            make_doc(currency=extracted_str("CHF"), subtotal_evidence="Rp 9.000"),
            today=TODAY,
        )
        assert reasons_of(result, "currency") == [CURRENCY_UNKNOWN, CURRENCY_CONTRADICTS_EVIDENCE]


class TestTaxId:
    def test_a_15_digit_npwp_is_clean(self) -> None:
        result = validate_fields(
            make_doc(vendor_tax_id=extracted_str("01.234.567.8-901.000")), today=TODAY
        )
        assert reasons_of(result, "vendor_tax_id") == []

    def test_a_16_digit_npwp_is_clean(self) -> None:
        result = validate_fields(
            make_doc(vendor_tax_id=extracted_str("31.532.144.3.604.000.5")), today=TODAY
        )
        assert reasons_of(result, "vendor_tax_id") == []

    def test_another_digit_count_is_unusual(self) -> None:
        result = validate_fields(
            make_doc(vendor_tax_id=extracted_str("12-3456789")), today=TODAY
        )
        assert reasons_of(result, "vendor_tax_id") == [TAX_ID_UNUSUAL_SHAPE]

    def test_no_digits_at_all_is_implausible(self) -> None:
        result = validate_fields(
            make_doc(vendor_tax_id=extracted_str("NPWP-ABC")), today=TODAY
        )
        assert reasons_of(result, "vendor_tax_id") == [TAX_ID_IMPLAUSIBLE]

    def test_skipped_without_an_extracted_tax_id(self) -> None:
        for tax_id in (Field.absent(0.9), Field.abstain("redacted", confidence=0.4)):
            result = validate_fields(make_doc(vendor_tax_id=tax_id), today=TODAY)
            assert TAX_ID_IMPLAUSIBLE not in reasons_of(result, "vendor_tax_id")
            assert TAX_ID_UNUSUAL_SHAPE not in reasons_of(result, "vendor_tax_id")

    def test_twice_is_once(self) -> None:
        document = make_doc(vendor_tax_id=extracted_str("12-3456789"))
        once = validate_fields(document, today=TODAY)
        assert validate_fields(once, today=TODAY) == once


class TestCodesOnly:
    def test_every_recorded_reason_is_a_known_code(self) -> None:
        document = make_doc(
            document_date=extracted_date("2024-04-01"),
            due_date=extracted_date("2024-03-01"),
            currency=extracted_str("CHF", "We charge in CHF"),
            vendor_tax_id=extracted_str("12-3456789"),
        )
        result = validate_fields(document, today=TODAY)
        for name in ("document_date", "due_date", "currency", "vendor_tax_id"):
            assert set(reasons_of(result, name)) <= set(REASON_CODES)
