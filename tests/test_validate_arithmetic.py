from decimal import Decimal
from typing import Any

import pytest

from docextract.extract.schema import DocumentExtraction, LineItem
from docextract.validate.arithmetic import (
    MISMATCH_CELLS,
    MISMATCH_CONFIDENCE_CAP,
    QTY_X_PRICE_NE_AMOUNT,
    has_arithmetic_mismatch,
    tolerance,
    validate_line_item,
    validate_line_items,
)

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


class TestTolerance:
    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            ("0", "0.01"),
            ("1.00", "0.01"),
            ("2.00", "0.01"),  # 0.5% of 2 is 0.01
            ("100", "0.5"),
            ("9117", "45.585"),
            ("-100", "0.5"),  # discount rows: uses |amount|
        ],
    )
    def test_max_of_absolute_and_relative(self, amount: str, expected: str) -> None:
        assert tolerance(D(amount)) == D(expected)


class TestMismatchDetection:
    def test_exact_match(self) -> None:
        assert not has_arithmetic_mismatch(item())

    def test_decimal_arithmetic_is_exact(self) -> None:
        assert not has_arithmetic_mismatch(item("3", "0.1", "0.3"))  # 0.1 * 3 == 0.3 exactly

    def test_weight_item_rounded_by_the_printer_is_not_flagged(self) -> None:
        # 0.352 kg x 25,900 = 9,116.8, printed as 9,117
        assert not has_arithmetic_mismatch(item("0.352", "25900", "9117"))

    def test_rounded_rupiah_amounts_are_not_flagged(self) -> None:
        assert not has_arithmetic_mismatch(item("3", "3333", "10000"))  # 9,999 vs 10,000

    def test_dropped_digit_is_flagged(self) -> None:
        assert has_arithmetic_mismatch(item("2", "45000", "9000"))  # amount lost a zero

    def test_quantity_misread_as_price_is_flagged(self) -> None:
        assert has_arithmetic_mismatch(item("12500", "2", "12500"))

    def test_off_by_a_few_percent_is_flagged(self) -> None:
        assert has_arithmetic_mismatch(item("1", "100", "103"))

    @pytest.mark.parametrize(
        ("quantity", "unit_price", "amount", "flagged"),
        [
            ("1", "100.50", "100", False),  # diff 0.5 == tolerance 0.5: not over
            ("1", "100.51", "100", True),  # diff 0.51 > 0.5
            ("1", "1.01", "1.00", False),  # diff 0.01 == tolerance 0.01
            ("1", "1.02", "1.00", True),
            ("1", "99.50", "100", False),  # below as well as above
            ("1", "99.49", "100", True),
        ],
    )
    def test_boundary_is_strictly_greater_than(
        self, quantity: str, unit_price: str, amount: str, flagged: bool
    ) -> None:
        assert has_arithmetic_mismatch(item(quantity, unit_price, amount)) is flagged

    def test_negative_discount_row(self) -> None:
        assert not has_arithmetic_mismatch(item("1", "-5000", "-5000"))
        assert has_arithmetic_mismatch(item("1", "-5000", "-500"))

    @pytest.mark.parametrize(
        "cells",
        [
            {"quantity": None},
            {"unit_price": None},
            {"amount": None},
            {"quantity": None, "unit_price": None},
        ],
    )
    def test_missing_cell_is_skipped_not_flagged(self, cells: dict[str, None]) -> None:
        values = {"quantity": "2", "unit_price": "4.50", "amount": "999", **cells}
        assert not has_arithmetic_mismatch(item(**values))


class TestValidateLineItem:
    def test_mismatch_flags_adds_reason_and_caps_confidence_but_keeps_status(self) -> None:
        result = validate_line_item(item("2", "45000", "9000", confidence=0.95))
        assert result.flagged_cells == list(MISMATCH_CELLS) == ["quantity", "unit_price", "amount"]
        assert result.reasons == [QTY_X_PRICE_NE_AMOUNT]
        assert result.confidence == MISMATCH_CONFIDENCE_CAP
        assert result.status == "extracted"  # a mismatch alone never forces an abstain

    def test_cells_are_untouched(self) -> None:
        before = item("2", "45000", "9000")
        after = validate_line_item(before)
        for name in ("description", "quantity", "unit_price", "amount", "evidence", "page"):
            assert getattr(after, name) == getattr(before, name)

    def test_confidence_is_only_ever_lowered(self) -> None:
        assert validate_line_item(item("2", "45000", "9000", confidence=0.3)).confidence == 0.3
        assert validate_line_item(item("2", "45000", "9000", confidence=0.5)).confidence == 0.5
        assert validate_line_item(item("2", "45000", "9000", confidence=1.0)).confidence == 0.5

    def test_clean_row_is_unchanged(self) -> None:
        before = item(confidence=0.97)
        after = validate_line_item(before)
        assert after.flagged_cells == []
        assert after.reasons == []
        assert after.confidence == 0.97
        assert after == before

    def test_existing_reasons_are_kept_in_order_and_the_code_is_appended(self) -> None:
        result = validate_line_item(item("2", "45000", "9000", reasons=["blurry", "second pass"]))
        assert result.reasons == ["blurry", "second pass", QTY_X_PRICE_NE_AMOUNT]

    def test_code_is_not_duplicated_when_the_model_already_sent_it(self) -> None:
        result = validate_line_item(item("2", "45000", "9000", reasons=[QTY_X_PRICE_NE_AMOUNT]))
        assert result.reasons == [QTY_X_PRICE_NE_AMOUNT]

    def test_model_sent_flags_are_ignored_on_a_clean_row(self) -> None:
        sent = item(flagged_cells=["amount"], reasons=["ok", QTY_X_PRICE_NE_AMOUNT])
        result = validate_line_item(sent)
        assert result.flagged_cells == []
        assert result.reasons == ["ok"]

    def test_model_sent_flags_are_overwritten_on_a_bad_row(self) -> None:
        result = validate_line_item(item("2", "45000", "9000", flagged_cells=["description"]))
        assert result.flagged_cells == ["quantity", "unit_price", "amount"]

    def test_abstained_and_absent_rows_get_no_flags(self) -> None:
        abstained = LineItem.model_validate(
            {"status": "abstained", "confidence": 0.2, "reasons": ["unreadable"]}
        )
        absent = LineItem.model_validate({"status": "absent", "confidence": 0.9})
        for row in (abstained, absent):
            result = validate_line_item(row)
            assert result.flagged_cells == []
            assert result.status == row.status
            assert result.confidence == row.confidence

    def test_input_is_never_mutated(self) -> None:
        before = item("2", "45000", "9000", reasons=["blurry"], flagged_cells=["description"])
        snapshot = before.model_dump()
        result = validate_line_item(before)
        assert before.model_dump() == snapshot
        assert result is not before
        assert result.reasons is not before.reasons

    @pytest.mark.parametrize("cells", [("2", "45000", "9000"), ("2", "4.50", "9.00")])
    def test_validating_twice_equals_validating_once(self, cells: tuple[str, str, str]) -> None:
        once = validate_line_item(item(*cells, confidence=0.9))
        twice = validate_line_item(once)
        assert twice == once


def make_document(rows: list[dict[str, Any]]) -> DocumentExtraction:
    def field(value: Any) -> dict[str, Any]:
        return {"value": value, "confidence": 0.9, "status": "extracted"}

    def none() -> dict[str, Any]:
        return {"value": None, "confidence": 0.9, "status": "absent"}

    return DocumentExtraction.model_validate(
        {
            "vendor_name": field("ACME"),
            "vendor_address": none(),
            "vendor_tax_id": none(),
            "buyer_name": none(),
            "document_number": none(),
            "document_date": none(),
            "due_date": none(),
            "currency": field("IDR"),
            "payment_method": none(),
            "line_items": rows,
            "subtotal": none(),
            "tax": none(),
            "discount": none(),
            "total": field("100000"),
        }
    )


def row(quantity: str, unit_price: str, amount: str) -> dict[str, Any]:
    return {
        "description": "x",
        "quantity": quantity,
        "unit_price": unit_price,
        "amount": amount,
        "status": "extracted",
        "confidence": 0.9,
    }


class TestValidateLineItems:
    def test_each_row_is_judged_independently(self) -> None:
        document = make_document(
            [row("2", "4500", "9000"), row("2", "45000", "9000"), row("0.352", "25900", "9117")]
        )
        result = validate_line_items(document)
        assert [r.flagged_cells for r in result.line_items] == [[], list(MISMATCH_CELLS), []]
        assert [r.confidence for r in result.line_items] == [0.9, MISMATCH_CONFIDENCE_CAP, 0.9]

    def test_scalar_fields_are_untouched(self) -> None:
        document = make_document([row("2", "45000", "9000")])
        result = validate_line_items(document)
        for name in ("vendor_name", "currency", "total", "due_date"):
            assert getattr(result, name) == getattr(document, name)

    def test_raw_document_is_never_mutated_so_raw_vs_validated_is_inspectable(self) -> None:
        document = make_document([row("2", "45000", "9000")])
        snapshot = document.model_dump()
        result = validate_line_items(document)
        assert document.model_dump() == snapshot
        assert document.line_items[0].flagged_cells == []
        assert document.line_items[0].confidence == 0.9
        assert result.line_items[0].flagged_cells == list(MISMATCH_CELLS)

    def test_no_rows(self) -> None:
        document = make_document([])
        assert validate_line_items(document) == document
