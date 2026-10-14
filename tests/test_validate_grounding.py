from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from docextract.extract.schema import DocumentExtraction, Field, LineItem
from docextract.validate import validate_document
from docextract.validate.codes import (
    DESCRIPTION_NOT_VERBATIM,
    EVIDENCE_AMBIGUOUS,
    EVIDENCE_NOT_GROUNDED,
    HARD_CODES,
    REASON_CODES,
)
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
        result = validate_grounding(make_doc(subtotal=scalar(D("1250000.00"), "Rp 1.250.000,00")))
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

    def test_an_ambiguous_span_that_contains_the_value_is_soft_not_grounding(self) -> None:
        # "1.250" alone is 1250 or 1.25; with no currency context it confirms neither, but it does
        # not contradict the model's 1250 either: the value is kept, flagged, never hard
        result = validate_grounding(
            make_doc(currency=Field.absent(0.9), subtotal=scalar(D("1250"), "1.250"))
        )
        assert result.subtotal.reasons == [EVIDENCE_AMBIGUOUS]
        assert result.subtotal.status == "extracted"
        assert result.subtotal.value == D("1250")  # never rewritten

    def test_either_reading_of_an_ambiguous_span_is_consistent(self) -> None:
        for value in (D("1250"), D("1.25")):
            result = validate_grounding(
                make_doc(currency=Field.absent(0.9), subtotal=scalar(value, "1.250"))
            )
            assert result.subtotal.reasons == [EVIDENCE_AMBIGUOUS], value

    def test_an_ambiguous_span_that_does_not_contain_the_value_is_still_hard(self) -> None:
        # "1.250" is 1250 or 1.25; 12.50 is neither (and 125 is neither)
        for value in (D("12.50"), D("125"), D("1250000")):
            result = validate_grounding(
                make_doc(currency=Field.absent(0.9), subtotal=scalar(value, "1.250"))
            )
            assert result.subtotal.reasons == [EVIDENCE_NOT_GROUNDED], value

    def test_a_value_near_a_reading_is_not_that_reading(self) -> None:
        # exact decimal equality: one cent or one unit off a candidate is a contradiction
        for value in (D("1.26"), D("1.24"), D("1249"), D("1251")):
            result = validate_grounding(
                make_doc(currency=Field.absent(0.9), subtotal=scalar(value, "1.250"))
            )
            assert result.subtotal.reasons == [EVIDENCE_NOT_GROUNDED], value

    def test_a_money_value_that_is_not_a_number_grounds_nothing(self) -> None:
        document = make_doc(subtotal=scalar(D("9.00"), "Subtotal 9.00"))
        object.__setattr__(document.subtotal, "value", "9.00")  # bypass the model's own typing
        result = validate_grounding(document)
        assert result.subtotal.reasons == [EVIDENCE_NOT_GROUNDED]

    def test_a_confirming_span_beats_an_ambiguous_one_in_the_same_quote(self) -> None:
        result = validate_grounding(
            make_doc(
                currency=Field.absent(0.9),
                subtotal=scalar(D("1250"), "Harga 1.250 (seribu 1250)"),
            )
        )
        assert result.subtotal.reasons == []

    def test_an_ambiguous_total_is_soft_in_a_document_with_no_currency(self) -> None:
        result = validate_grounding(
            make_doc(currency=Field.absent(0.9), total=scalar(D("208179"), "TOTAL 208,179"))
        )
        assert result.total.reasons == [EVIDENCE_AMBIGUOUS]

    def test_an_ambiguous_span_grounds_with_the_currency_context(self) -> None:
        result = validate_grounding(
            make_doc(currency=scalar("IDR", "Rp"), subtotal=scalar(D("1250"), "1.250"))
        )
        assert result.subtotal.reasons == []

    def test_a_negative_amount_in_parentheses_grounds(self) -> None:
        result = validate_grounding(make_doc(discount=scalar(D("-9.00"), "Potongan Harga (9.00)")))
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
        result = validate_grounding(make_doc(document_date=scalar(date(2024, 3, 5), "9.99")))
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
            make_doc(vendor_tax_id=scalar("012345678901000", "NPWP 01.234.567.8-901.000"))
        )
        assert result.vendor_tax_id.reasons == []

    def test_a_document_number_grounds_with_other_formatting(self) -> None:
        result = validate_grounding(make_doc(document_number=scalar("INV123", "No. INV/123")))
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

    def test_an_ambiguous_amount_is_a_soft_row_code_and_the_row_stays(self) -> None:
        result = validate_grounding(
            make_doc(
                currency=Field.absent(0.9),
                rows=[
                    row(
                        description="Kopi",
                        quantity="1",
                        unit_price="1250",
                        amount="1250",
                        evidence="Kopi 1 x 1.250 1.250",
                    )
                ],
            )
        )
        item = result.line_items[0]
        assert item.reasons == [EVIDENCE_AMBIGUOUS]
        assert item.flagged_cells == ["unit_price", "amount"]
        assert item.status == "extracted"

    def test_an_ungrounded_cell_beats_an_ambiguous_one_in_the_same_row(self) -> None:
        result = validate_grounding(
            make_doc(
                currency=Field.absent(0.9),
                rows=[
                    row(
                        description="Kopi",
                        quantity="1",
                        unit_price="1250",
                        amount="9999",
                        evidence="Kopi 1 x 1.250 1.250",
                    )
                ],
            )
        )
        item = result.line_items[0]
        assert item.reasons == [EVIDENCE_NOT_GROUNDED]
        assert item.flagged_cells == ["amount"]

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


class TestMultiplicationSigns:
    """A quantity printed with a multiplication sign glued on (`x1`, `2x`) is in its own quote.

    Only a quantity is read that way, and only a sign + a number: `250ml` or `MAX1` must not
    ground a quantity (G3, docs/milestones/G3.md)."""

    @pytest.mark.parametrize(
        ("quantity", "unit_price", "amount", "evidence"),
        [
            ("1", "5000", "5000", "AIR MINERAL 5.000 x1 5.000"),
            ("2", "169000", "338000", "Matcha 2x 169.000 338.000"),
            ("3", "5000", "15000", "Kopi X3 5.000 15.000"),
            ("3", "5000", "15000", "Kopi ×3 5.000 15.000"),
            ("3", "5000", "15000", "Kopi 3× 5.000 15.000"),
            ("2", "5000", "10000", "Kopi (2x) 5.000 10.000"),
            ("1", "5000", "5000", "Kopi x1, 5.000 5.000"),
            ("10", "5000", "50000", "Kopi 10X 5.000 50.000"),
        ],
    )
    def test_a_quantity_with_a_sign_glued_on_is_grounded(
        self, quantity: str, unit_price: str, amount: str, evidence: str
    ) -> None:
        result = validate_grounding(
            make_doc(
                rows=[
                    row(
                        description=None,
                        quantity=quantity,
                        unit_price=unit_price,
                        amount=amount,
                        evidence=evidence,
                    )
                ]
            )
        )
        assert result.line_items[0].reasons == []
        assert result.line_items[0].flagged_cells == []

    def test_a_different_quantity_is_still_not_grounded(self) -> None:
        result = validate_grounding(
            make_doc(
                rows=[
                    row(
                        description="Kopi",
                        quantity="2",
                        unit_price="5000",
                        amount="5000",
                        evidence="Kopi x1 5.000 5.000",
                    )
                ]
            )
        )
        assert result.line_items[0].reasons == [EVIDENCE_NOT_GROUNDED]
        assert result.line_items[0].flagged_cells == ["quantity"]

    @pytest.mark.parametrize(
        ("quantity", "evidence"),
        [
            ("1", "MAX1 Widget"),  # a letter run that ends in a digit
            ("1", "x1x Widget"),  # a sign on both sides
            ("2", "2xl Widget"),  # more letters than the sign
            ("1", "Widget x"),  # a sign with no number
            ("250", "Widget 250ml"),  # a unit is not a sign: it must not ground a quantity
        ],
    )
    def test_what_is_not_a_sign_and_a_number_grounds_nothing(
        self, quantity: str, evidence: str
    ) -> None:
        result = validate_grounding(
            make_doc(
                rows=[
                    row(
                        description="Widget",
                        quantity=quantity,
                        unit_price=None,
                        amount=None,
                        evidence=evidence,
                    )
                ]
            )
        )
        assert result.line_items[0].reasons == [EVIDENCE_NOT_GROUNDED]
        assert result.line_items[0].flagged_cells == ["quantity"]

    def test_the_sign_is_read_for_a_quantity_only(self) -> None:
        """`x4.50` is not a unit price: a looser rule widens what a wrong price can hide behind."""
        result = validate_grounding(
            make_doc(
                rows=[
                    row(
                        description="Widget",
                        quantity="2",
                        unit_price="4.50",
                        amount="9.00",
                        evidence="Widget 2 x4.50 9.00",
                    )
                ]
            )
        )
        assert result.line_items[0].reasons == [EVIDENCE_NOT_GROUNDED]
        assert result.line_items[0].flagged_cells == ["unit_price"]

    def test_a_scalar_is_not_read_through_a_sign(self) -> None:
        result = validate_grounding(make_doc(subtotal=scalar(D("9.00"), "x9.00")))
        assert result.subtotal.reasons == [EVIDENCE_NOT_GROUNDED]

    def test_an_ambiguous_number_behind_a_sign_stays_ambiguous(self) -> None:
        """The normalizer still never picks a reading: `1.250` with no currency is 1250 or 1.25."""
        ambiguous = validate_grounding(
            make_doc(
                currency=Field.absent(0.9),
                rows=[
                    row(
                        description="Kopi",
                        quantity="1250",
                        unit_price=None,
                        amount=None,
                        evidence="Kopi x1.250",
                    )
                ],
            )
        )
        assert ambiguous.line_items[0].reasons == [EVIDENCE_AMBIGUOUS]
        assert ambiguous.line_items[0].flagged_cells == ["quantity"]

        neither = validate_grounding(
            make_doc(
                currency=Field.absent(0.9),
                rows=[
                    row(
                        description="Kopi",
                        quantity="7",
                        unit_price=None,
                        amount=None,
                        evidence="Kopi x1.250",
                    )
                ],
            )
        )
        assert neither.line_items[0].reasons == [EVIDENCE_NOT_GROUNDED]


# The nine rows of `results/test-public` withheld only because the quote prints the quantity as
# `x1` / `2x`, exactly as the model answered: (description, quantity, unit_price, amount, evidence).
# Embedded, not read from results/: the mutation tool works on a copy of the repo without it.
DOC_027_ROWS = [
    ("AIR MINERAL", "1", "5000", "5000", "AIR MINERAL 5.000 x1 5.000"),
    ("ADEM SARI", "1", "7000", "7000", "ADEM SARI 7.000 x1 7.000"),
    ("MADU LEMON (ICE/HOT)", "1", "15000", "15000", "MADU LEMON (ICE/HOT) 15.000 x1 15.000"),
    ("Indomie Kuah Kari Biasa", "1", "15000", "15000", "Indomie Kuah Kari Biasa 15.000 x1 15.000"),
    (
        "INDOMIE BANGLADESH JUMBO + PUDING TELUR AYAM KAMPUNG",
        "1",
        "38000",
        "38000",
        "INDOMIE BANGLADESH JUMBO + PUDIN G TELUR AYAM KAMPUNG 38.000 x1 38.000",
    ),
    ("INDOMIE PAKISTAN JUMBO", "1", "23000", "23000", "INDOMIE PAKISTAN JUMBO 23.000 x1 23.000"),
    (
        "INDOMIE GORENG ALOK BIASA",
        "1",
        "15000",
        "15000",
        "INDOMIE GORENG ALOK BIASA 15.000 x1 15.000",
    ),
]
DOC_020_ROWS = [  # this document printed no currency, so its `169.000` is ambiguous
    (
        "Extra Dark 70% (Pisah Bubuk)",
        "2",
        "169000",
        "338000",
        "Extra Dark 70% (Pisah Bubuk)\n2x 169.000 338.000",
    ),
    ("Matcha", "2", "169000", "338000", "Matcha\n2x 169.000 338.000"),
]


def observed(rows: list[tuple[str, str, str, str, str]]) -> list[LineItem]:
    return [
        row(description=d, quantity=q, unit_price=u, amount=a, evidence=e) for d, q, u, a, e in rows
    ]


class TestRowsObservedInTestPublic:
    """The two real documents that print `x1` / `2x`, through the whole validator chain."""

    def test_doc_027_keeps_all_seven_rows(self) -> None:
        result = validate_document(make_doc(rows=observed(DOC_027_ROWS)), abstain_below=0.3)
        assert len(result.line_items) == 7
        for item in result.line_items:
            assert EVIDENCE_NOT_GROUNDED not in item.reasons
            assert item.status == "extracted"
            assert item.quantity == D("1")

    def test_doc_020_keeps_its_rows_with_the_soft_code_for_the_ambiguous_price(self) -> None:
        document = make_doc(currency=Field.absent(0.9), rows=observed(DOC_020_ROWS))
        result = validate_document(document, abstain_below=0.3)
        assert len(result.line_items) == 2
        for item in result.line_items:
            assert EVIDENCE_NOT_GROUNDED not in item.reasons
            assert EVIDENCE_AMBIGUOUS in item.reasons  # 169.000 with no currency: 169000 or 169
            assert item.status == "extracted"
            assert item.quantity == D("2")


class TestCurrencyField:
    def test_a_code_is_grounded_by_its_marker(self) -> None:
        result = validate_grounding(make_doc(currency=scalar("IDR", "Total Rp 9.000")))
        assert result.currency.reasons == []

    def test_a_code_is_grounded_by_the_code_itself(self) -> None:
        result = validate_grounding(make_doc(currency=scalar("IDR", "Mata uang: IDR")))
        assert result.currency.reasons == []

    def test_a_code_is_grounded_by_the_currency_word(self) -> None:
        result = validate_grounding(make_doc(currency=scalar("IDR", "dibayar dalam Rupiah")))
        assert result.currency.reasons == []

    def test_a_contradicting_marker_does_not_ground_it(self) -> None:
        result = validate_grounding(make_doc(currency=scalar("USD", "Total Rp 9.000")))
        assert result.currency.reasons == [EVIDENCE_NOT_GROUNDED]

    def test_evidence_with_no_currency_at_all_does_not_ground_it(self) -> None:
        result = validate_grounding(make_doc(currency=scalar("IDR", "Total 9.000")))
        assert result.currency.reasons == [EVIDENCE_NOT_GROUNDED]


class TestTheNewCode:
    def test_it_is_a_soft_reason_code(self) -> None:
        assert EVIDENCE_AMBIGUOUS in REASON_CODES
        assert EVIDENCE_AMBIGUOUS not in HARD_CODES
        assert EVIDENCE_NOT_GROUNDED in HARD_CODES


class TestIdempotenceAndCopies:
    def test_twice_is_once(self) -> None:
        document = make_doc(
            vendor_name=scalar("Toko Sebelah", "PT Contoh"),
            rows=[row(amount="9.50"), row(description="Paper", evidence="2 x 4.50 = 9.00")],
        )
        once = validate_grounding(document)
        assert validate_grounding(once) == once

    def test_twice_is_once_for_the_ambiguous_code_too(self) -> None:
        document = make_doc(
            currency=Field.absent(0.9),
            subtotal=scalar(D("1250"), "1.250"),
            rows=[
                row(
                    description="Kopi",
                    amount="1250",
                    unit_price="1250",
                    quantity="1",
                    evidence="Kopi 1 x 1.250 1.250",
                )
            ],
        )
        once = validate_grounding(document)
        assert once.subtotal.reasons == [EVIDENCE_AMBIGUOUS]
        assert once.line_items[0].reasons == [EVIDENCE_AMBIGUOUS]
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
