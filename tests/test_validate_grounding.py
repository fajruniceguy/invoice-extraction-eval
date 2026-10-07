from datetime import date
from decimal import Decimal
from typing import Any

from docextract.extract.schema import DocumentExtraction, Field, LineItem
from docextract.validate.codes import DESCRIPTION_NOT_VERBATIM, EVIDENCE_NOT_GROUNDED
from docextract.validate.grounding import validate_grounding

D = Decimal


def scalar(value: Any, evidence: str | None, **extra: Any) -> Field[Any]:
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


def row(
    description: str | None = "Widget",
    quantity: str | None = "2",
    unit_price: str | None = "4.50",
    amount: str | None = "9.00",
    evidence: str | None = "2 x 4.50 = 9.00 Widget",
    **extra: Any,
) -> LineItem:
    return LineItem.model_validate(
        {
            "description": description,
            "quantity": quantity,
            "unit_price": unit_price,
            "amount": amount,
            "status": "extracted",
            "confidence": 0.9,
            "evidence": evidence,
            "page": 1,
            **extra,
        }
    )


def make_doc(
    *,
    currency: Field[str] | None = None,
    vendor_name: Field[str] | None = None,
    vendor_tax_id: Field[str] | None = None,
    document_number: Field[str] | None = None,
    document_date: Field[date] | None = None,
    due_date: Field[date] | None = None,
    subtotal: Field[Decimal] | None = None,
    tax: Field[Decimal] | None = None,
    discount: Field[Decimal] | None = None,
    total: Field[Decimal] | None = None,
    rows: list[LineItem] | None = None,
) -> DocumentExtraction:
    return DocumentExtraction(
        vendor_name=vendor_name if vendor_name is not None else scalar("PT Contoh", "PT Contoh"),
        vendor_address=Field.absent(0.9),
        vendor_tax_id=vendor_tax_id
        if vendor_tax_id is not None
        else scalar("01.234.567.8-901.000", "01.234.567.8-901.000"),
        buyer_name=Field.absent(0.9),
        document_number=document_number
        if document_number is not None
        else scalar("INV-1", "INV-1"),
        document_date=document_date
        if document_date is not None
        else scalar(date(2024, 3, 5), "5 Maret 2024"),
        due_date=due_date if due_date is not None else Field.absent(0.9),
        currency=currency if currency is not None else scalar("IDR", "Rp"),
        payment_method=Field.absent(0.9),
        line_items=rows if rows is not None else [row()],
        subtotal=subtotal if subtotal is not None else scalar(D("9.00"), "9.00"),
        tax=tax if tax is not None else scalar(D("0.99"), "0.99"),
        discount=discount if discount is not None else Field.absent(0.9),
        total=total if total is not None else scalar(D("9.99"), "9.99"),
    )


class TestNumbers:
    def test_a_printed_rupiah_amount_grounds_the_decimal(self) -> None:
        result = validate_grounding(
            make_doc(subtotal=scalar(D("1250000.00"), "Rp 1.250.000,00"))
        )
        assert result.subtotal.reasons == []

    def test_a_plain_amount_grounds(self) -> None:
        assert validate_grounding(make_doc()).subtotal.reasons == []

    def test_the_value_may_have_a_different_scale(self) -> None:
        result = validate_grounding(make_doc(subtotal=scalar(D("9"), "9.00")))
        assert result.subtotal.reasons == []

    def test_a_marker_beats_the_document_currency(self) -> None:
        result = validate_grounding(
            make_doc(
                currency=scalar("IDR", "Rp"),
                subtotal=scalar(D("9.00"), "$ 9.00"),
            )
        )
        assert result.subtotal.reasons == []

    def test_an_amount_not_in_the_quote_is_not_grounded(self) -> None:
        result = validate_grounding(make_doc(subtotal=scalar(D("9.50"), "Subtotal 9.00")))
        assert result.subtotal.reasons == [EVIDENCE_NOT_GROUNDED]

    def test_an_ambiguous_span_is_not_grounding(self) -> None:
        # "1.250" alone is 1250 or 1.25; with no currency context it confirms neither
        result = validate_grounding(
            make_doc(currency=Field.absent(0.9), subtotal=scalar(D("1250"), "1.250"))
        )
        assert result.subtotal.reasons == [EVIDENCE_NOT_GROUNDED]

    def test_an_ambiguous_span_grounds_with_the_currency_context(self) -> None:
        result = validate_grounding(
            make_doc(currency=scalar("IDR", "Rp"), subtotal=scalar(D("1250"), "1.250"))
        )
        assert result.subtotal.reasons == []

    def test_a_negative_amount_in_parentheses_grounds(self) -> None:
        result = validate_grounding(
            make_doc(discount=scalar(D("-9.00"), "Potongan Harga (9.00)"))
        )
        assert result.discount.reasons == []

    def test_no_evidence_grounds_nothing(self) -> None:
        result = validate_grounding(make_doc(subtotal=scalar(D("9.00"), None)))
        assert result.subtotal.reasons == [EVIDENCE_NOT_GROUNDED]


class TestDates:
    def test_an_indonesian_date_grounds(self) -> None:
        result = validate_grounding(
            make_doc(document_date=scalar(date(2024, 3, 5), "Tanggal 5 Maret 2024"))
        )
        assert result.document_date.reasons == []

    def test_an_iso_date_grounds(self) -> None:
        result = validate_grounding(make_doc(document_date=scalar(date(2024, 3, 5), "2024-03-05")))
        assert result.document_date.reasons == []

    def test_day_first_resolves_for_an_idr_document(self) -> None:
        result = validate_grounding(
            make_doc(
                currency=scalar("IDR", "Rp"),
                document_date=scalar(date(2024, 3, 5), "05/03/2024"),
            )
        )
        assert result.document_date.reasons == []

    def test_a_different_date_is_not_grounded(self) -> None:
        result = validate_grounding(
            make_doc(document_date=scalar(date(2024, 3, 6), "5 Maret 2024"))
        )
        assert result.document_date.reasons == [EVIDENCE_NOT_GROUNDED]

    def test_a_money_amount_does_not_ground_a_date(self) -> None:
        result = validate_grounding(
            make_doc(document_date=scalar(date(2024, 3, 5), "9.99"))
        )
        assert result.document_date.reasons == [EVIDENCE_NOT_GROUNDED]


class TestStrings:
    def test_case_and_punctuation_are_ignored(self) -> None:
        result = validate_grounding(
            make_doc(vendor_name=scalar("PT Contoh Makmur", "penjual: PT. CONTOH MAKMUR"))
        )
        assert result.vendor_name.reasons == []

    def test_a_paraphrased_name_is_not_grounded(self) -> None:
        result = validate_grounding(
            make_doc(vendor_name=scalar("Toko Sebelah", "PT Contoh Makmur"))
        )
        assert result.vendor_name.reasons == [EVIDENCE_NOT_GROUNDED]

    def test_a_tax_id_compares_on_digits(self) -> None:
        result = validate_grounding(
            make_doc(
                vendor_tax_id=scalar("012345678901000", "NPWP 01.234.567.8-901.000")
            )
        )
        assert result.vendor_tax_id.reasons == []

    def test_a_document_number_grounds_with_other_formatting(self) -> None:
        result = validate_grounding(
            make_doc(document_number=scalar("INV123", "No. INV/123"))
        )
        assert result.document_number.reasons == []

    def test_an_absent_or_abstained_field_is_skipped(self) -> None:
        result = validate_grounding(
            make_doc(
                vendor_name=Field.absent(0.9),
                vendor_tax_id=Field.abstain("redacted", confidence=0.4),
            )
        )
        assert result.vendor_name.reasons == []
        assert result.vendor_tax_id.reasons == ["redacted"]


class TestRows:
    def test_a_consistent_row_is_untouched(self) -> None:
        result = validate_grounding(make_doc())
        assert result.line_items[0].reasons == []
        assert result.line_items[0].flagged_cells == []

    def test_an_ungrounded_amount_abstains_the_row_later(self) -> None:
        result = validate_grounding(make_doc(rows=[row(amount="9.50")]))
        assert result.line_items[0].reasons == [EVIDENCE_NOT_GROUNDED]
        assert result.line_items[0].flagged_cells == ["amount"]

    def test_an_ungrounded_quantity_and_price_are_flagged(self) -> None:
        result = validate_grounding(
            make_doc(rows=[row(quantity="5", unit_price="1.80", amount="9.00")])
        )
        assert result.line_items[0].reasons == [EVIDENCE_NOT_GROUNDED]
        assert result.line_items[0].flagged_cells == ["quantity", "unit_price"]

    def test_a_reworded_description_is_a_soft_code(self) -> None:
        result = validate_grounding(
            make_doc(rows=[row(description="Paper", evidence="2 x 4.50 = 9.00")])
        )
        assert result.line_items[0].reasons == [DESCRIPTION_NOT_VERBATIM]
        assert result.line_items[0].flagged_cells == ["description"]

    def test_a_missing_description_is_not_checked(self) -> None:
        result = validate_grounding(make_doc(rows=[row(description=None)]))
        assert result.line_items[0].reasons == []
        assert result.line_items[0].flagged_cells == []

    def test_a_description_quoted_with_other_case_grounds(self) -> None:
        result = validate_grounding(
            make_doc(rows=[row(description="kertas A4", evidence="KERTAS A4 2 x 4.50 = 9.00")])
        )
        assert result.line_items[0].reasons == []

    def test_an_abstained_row_is_skipped(self) -> None:
        abstained = LineItem.model_validate(
            {
                "description": None,
                "quantity": None,
                "unit_price": None,
                "amount": None,
                "status": "abstained",
                "confidence": 0.2,
                "reasons": ["unreadable"],
            }
        )
        result = validate_grounding(make_doc(rows=[abstained]))
        assert result.line_items[0].reasons == ["unreadable"]
        assert result.line_items[0].flagged_cells == []

    def test_grounding_flags_join_row_arithmetic_flags(self) -> None:
        flagged = row(amount="8.00")
        flagged = flagged.model_copy(update={"flagged_cells": ["quantity", "unit_price", "amount"]})
        result = validate_grounding(make_doc(rows=[flagged]))
        assert result.line_items[0].flagged_cells == ["quantity", "unit_price", "amount"]


class TestIdempotenceAndCopies:
    def test_twice_is_once(self) -> None:
        document = make_doc(
            vendor_name=scalar("Toko Sebelah", "PT Contoh"),
            rows=[row(amount="9.50"), row(description="Paper", evidence="2 x 4.50 = 9.00")],
        )
        once = validate_grounding(document)
        assert validate_grounding(once) == once

    def test_the_input_is_never_mutated(self) -> None:
        document = make_doc(vendor_name=scalar("Toko Sebelah", "PT Contoh"))
        validate_grounding(document)
        assert document.vendor_name.reasons == []
        assert document.line_items[0].flagged_cells == []

    def test_every_flagged_reason_is_a_code(self) -> None:
        result = validate_grounding(
            make_doc(vendor_name=scalar("Toko Sebelah", "PT Contoh"), rows=[row(amount="9.50")])
        )
        assert result.vendor_name.reasons == [EVIDENCE_NOT_GROUNDED]
        assert result.line_items[0].reasons == [EVIDENCE_NOT_GROUNDED]
