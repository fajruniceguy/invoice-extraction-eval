import json
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from docextract.extract.schema import SCALAR_FIELD_NAMES, DocumentExtraction, Field, LineItem


def extracted(value: Any, confidence: float = 0.9, **extra: Any) -> dict[str, Any]:
    return {"value": value, "confidence": confidence, "status": "extracted", **extra}


def abstained() -> dict[str, Any]:
    return {"value": None, "confidence": 0.0, "status": "abstained", "reasons": ["not on document"]}


def full_payload() -> dict[str, Any]:
    return {
        "vendor_name": extracted("ACME Corp", evidence="ACME Corp", page=1),
        "vendor_address": extracted("1 Main St"),
        "vendor_tax_id": abstained(),
        "buyer_name": extracted("Jane Doe"),
        "document_number": extracted("INV-001"),
        "document_date": extracted("2024-03-05"),
        "due_date": abstained(),
        "currency": extracted("USD"),
        "payment_method": abstained(),
        "line_items": [
            {
                "description": "Widget",
                "quantity": 2,
                "unit_price": "4.50",
                "amount": 9.0,
                "confidence": 0.8,
                "evidence": "Widget 2 4.50 9.00",
                "page": 1,
            }
        ],
        "subtotal": extracted("9.00"),
        "tax": extracted(0.72),
        "discount": abstained(),
        "total": extracted("9.72"),
    }


class TestField:
    def test_extracted_field(self) -> None:
        f = Field[str].model_validate(extracted("x", evidence="x", page=2))
        assert f.value == "x"
        assert f.status == "extracted"
        assert f.page == 2
        assert f.reasons == []

    def test_abstained_field(self) -> None:
        f = Field[str].model_validate(abstained())
        assert f.value is None
        assert f.status == "abstained"
        assert f.reasons == ["not on document"]

    def test_abstain_helper(self) -> None:
        f = Field[Decimal].abstain("no discount shown")
        assert (f.value, f.confidence, f.status) == (None, 0.0, "abstained")
        assert f.reasons == ["no discount shown"]

    def test_abstained_with_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="abstained"):
            Field[str](value="oops", confidence=0.5, status="abstained")

    def test_extracted_with_null_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extracted"):
            Field[str](value=None, confidence=0.5, status="extracted")

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2])
    def test_confidence_bounds(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            Field[str].model_validate(extracted("x", confidence=bad))

    @pytest.mark.parametrize("ok", [0, 0.0, 0.5, 1, 1.0])
    def test_confidence_bounds_inclusive(self, ok: float) -> None:
        assert Field[str].model_validate(extracted("x", confidence=ok)).confidence == ok

    def test_unknown_status_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Field[str].model_validate({**extracted("x"), "status": "maybe"})

    def test_page_must_be_one_based(self) -> None:
        with pytest.raises(ValidationError):
            Field[str].model_validate(extracted("x", page=0))

    def test_value_status_and_confidence_are_required(self) -> None:
        for missing in ("value", "confidence", "status"):
            payload = extracted("x")
            del payload[missing]
            with pytest.raises(ValidationError):
                Field[str].model_validate(payload)

    def test_unknown_keys_are_ignored(self) -> None:
        f = Field[str].model_validate(extracted("x", rationale="because"))
        assert not hasattr(f, "rationale")

    def test_decimal_from_float_does_not_pick_up_binary_noise(self) -> None:
        f = Field[Decimal].model_validate(extracted(12.1))
        assert f.value == Decimal("12.1")

    def test_decimal_from_string_and_int(self) -> None:
        assert Field[Decimal].model_validate(extracted("1234.50")).value == Decimal("1234.50")
        assert Field[Decimal].model_validate(extracted(7)).value == Decimal(7)

    def test_decimal_rejects_garbage(self) -> None:
        with pytest.raises(ValidationError):
            Field[Decimal].model_validate(extracted("twelve"))

    def test_date_from_iso_string(self) -> None:
        assert Field[date].model_validate(extracted("2024-03-05")).value == date(2024, 3, 5)

    def test_date_rejects_non_iso(self) -> None:
        with pytest.raises(ValidationError):
            Field[date].model_validate(extracted("5th of March"))


class TestLineItem:
    def test_row_level_metadata_and_nullable_cells(self) -> None:
        item = LineItem.model_validate({"description": "Widget", "confidence": 0.7})
        assert item.quantity is None
        assert item.unit_price is None
        assert item.amount is None
        assert item.confidence == 0.7

    def test_cells_are_decimal(self) -> None:
        item = LineItem.model_validate(
            {"quantity": 3, "unit_price": "1.10", "amount": 3.3, "confidence": 1}
        )
        assert item.quantity == Decimal(3)
        assert item.unit_price == Decimal("1.10")
        assert item.amount == Decimal("3.3")

    def test_row_confidence_is_required_and_bounded(self) -> None:
        with pytest.raises(ValidationError):
            LineItem.model_validate({"description": "x"})
        with pytest.raises(ValidationError):
            LineItem.model_validate({"description": "x", "confidence": 1.5})


class TestDocumentExtraction:
    def test_parses_full_payload(self) -> None:
        doc = DocumentExtraction.model_validate(full_payload())
        assert doc.vendor_name.value == "ACME Corp"
        assert doc.vendor_name.evidence == "ACME Corp"
        assert doc.vendor_tax_id.status == "abstained"
        assert doc.document_date.value == date(2024, 3, 5)
        assert doc.due_date.value is None
        assert doc.subtotal.value == Decimal("9.00")
        assert doc.tax.value == Decimal("0.72")
        assert doc.total.value == Decimal("9.72")
        assert len(doc.line_items) == 1
        assert doc.line_items[0].amount == Decimal("9.0")

    def test_money_fields_are_decimal_and_dates_are_date(self) -> None:
        doc = DocumentExtraction.model_validate(full_payload())
        for name in ("subtotal", "tax", "total"):
            assert isinstance(getattr(doc, name).value, Decimal)
        assert isinstance(doc.document_date.value, date)

    def test_line_items_default_to_empty(self) -> None:
        payload = full_payload()
        del payload["line_items"]
        assert DocumentExtraction.model_validate(payload).line_items == []

    @pytest.mark.parametrize("name", SCALAR_FIELD_NAMES)
    def test_every_scalar_field_is_required(self, name: str) -> None:
        payload = full_payload()
        del payload[name]
        with pytest.raises(ValidationError) as exc:
            DocumentExtraction.model_validate(payload)
        assert exc.value.errors()[0]["loc"] == (name,)

    def test_invalid_field_inside_document_is_reported_with_its_path(self) -> None:
        payload = full_payload()
        payload["total"] = {"value": "9.72", "confidence": 2.0, "status": "extracted"}
        with pytest.raises(ValidationError) as exc:
            DocumentExtraction.model_validate(payload)
        assert exc.value.errors()[0]["loc"] == ("total", "confidence")

    def test_json_round_trip(self) -> None:
        doc = DocumentExtraction.model_validate(full_payload())
        again = DocumentExtraction.model_validate_json(doc.model_dump_json())
        assert again == doc

    def test_dump_keeps_decimal_precision_as_string(self) -> None:
        doc = DocumentExtraction.model_validate(full_payload())
        data = json.loads(doc.model_dump_json())
        assert data["subtotal"]["value"] == "9.00"
        assert data["document_date"]["value"] == "2024-03-05"

    def test_field_names(self) -> None:
        assert SCALAR_FIELD_NAMES == (
            "vendor_name",
            "vendor_address",
            "vendor_tax_id",
            "buyer_name",
            "document_number",
            "document_date",
            "due_date",
            "currency",
            "payment_method",
            "subtotal",
            "tax",
            "discount",
            "total",
        )

    def test_json_schema_can_be_generated_for_structured_output(self) -> None:
        schema = DocumentExtraction.model_json_schema()
        assert set(SCALAR_FIELD_NAMES) | {"line_items"} == set(schema["properties"])
        # the parametrised Field[...] models and LineItem must be reachable from $defs
        assert "LineItem" in schema["$defs"]
        assert any(name.startswith("Field_") for name in schema["$defs"])
