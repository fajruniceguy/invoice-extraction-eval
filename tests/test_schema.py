import json
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from docextract.extract.schema import SCALAR_FIELD_NAMES, DocumentExtraction, Field, LineItem
from docextract.providers.schema_utils import simplify_schema


def extracted(value: Any, confidence: float = 0.9, **extra: Any) -> dict[str, Any]:
    return {"value": value, "confidence": confidence, "status": "extracted", **extra}


def absent(confidence: float = 0.95, **extra: Any) -> dict[str, Any]:
    return {"value": None, "confidence": confidence, "status": "absent", **extra}


def abstained(reasons: list[str] | None = None) -> dict[str, Any]:
    return {
        "value": None,
        "confidence": 0.2,
        "status": "abstained",
        "reasons": ["smudged"] if reasons is None else reasons,
    }


def row(**extra: Any) -> dict[str, Any]:
    return {
        "description": "Widget",
        "quantity": 2,
        "unit_price": "4.50",
        "amount": 9.0,
        "status": "extracted",
        "confidence": 0.8,
        "evidence": "Widget 2 4.50 9.00",
        "page": 1,
        **extra,
    }


def full_payload() -> dict[str, Any]:
    return {
        "vendor_name": extracted("ACME Corp", evidence="ACME Corp", page=1),
        "vendor_address": extracted("1 Main St"),
        "vendor_tax_id": absent(),
        "buyer_name": extracted("Jane Doe"),
        "document_number": extracted("INV-001"),
        "document_date": extracted("2024-03-05"),
        "due_date": absent(0.7, reasons=["no due date printed"]),
        "currency": extracted("USD"),
        "payment_method": abstained(),
        "line_items": [row()],
        "subtotal": extracted("9.00"),
        "tax": extracted(0.72),
        "discount": absent(),
        "total": extracted("9.72"),
    }


class TestFieldStatus:
    def test_extracted_field(self) -> None:
        f = Field[str].model_validate(extracted("x", evidence="x", page=2))
        assert f.value == "x"
        assert f.status == "extracted"
        assert f.page == 2
        assert f.reasons == []

    def test_extracted_may_carry_reasons(self) -> None:
        f = Field[str].model_validate(extracted("x", reasons=["two candidates"]))
        assert f.reasons == ["two candidates"]

    def test_extracted_with_null_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extracted"):
            Field[str](value=None, confidence=0.5, status="extracted")

    @pytest.mark.parametrize("status", ["absent", "abstained"])
    def test_non_extracted_with_a_value_is_rejected(self, status: str) -> None:
        with pytest.raises(ValidationError, match=f"{status} field must have value=null"):
            Field[str].model_validate(
                {"value": "oops", "confidence": 0.5, "status": status, "reasons": ["r"]}
            )

    def test_absent_can_be_highly_confident_and_needs_no_reason(self) -> None:
        f = Field[date].model_validate(absent(1.0))
        assert (f.value, f.status, f.confidence, f.reasons) == (None, "absent", 1.0, [])

    def test_absent_may_carry_reasons(self) -> None:
        f = Field[date].model_validate(absent(0.8, reasons=["no due date printed"]))
        assert f.reasons == ["no due date printed"]

    def test_abstained_requires_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="at least one reason"):
            Field[str].model_validate(abstained(reasons=[]))

    @pytest.mark.parametrize("reasons", [[""], ["   "], ["", " "]])
    def test_blank_reasons_do_not_count(self, reasons: list[str]) -> None:
        with pytest.raises(ValidationError, match="at least one reason"):
            Field[str].model_validate(abstained(reasons=reasons))

    def test_abstained_with_a_reason_is_valid(self) -> None:
        f = Field[str].model_validate(abstained(["blurry"]))
        assert (f.value, f.status, f.reasons) == (None, "abstained", ["blurry"])

    def test_unknown_status_is_rejected(self) -> None:
        for bad in ("maybe", "missing", "EXTRACTED", None):
            with pytest.raises(ValidationError):
                Field[str].model_validate({**extracted("x"), "status": bad})

    def test_value_status_and_confidence_are_required(self) -> None:
        for missing in ("value", "confidence", "status"):
            payload = extracted("x")
            del payload[missing]
            with pytest.raises(ValidationError):
                Field[str].model_validate(payload)


class TestFieldHelpers:
    def test_absent_helper(self) -> None:
        f = Field[Decimal].absent(0.9, "no discount shown")
        assert (f.value, f.confidence, f.status) == (None, 0.9, "absent")
        assert f.reasons == ["no discount shown"]

    def test_absent_helper_without_reasons(self) -> None:
        assert Field[str].absent(0.99).reasons == []

    def test_abstain_helper(self) -> None:
        f = Field[Decimal].abstain("two totals printed", "ink smudge")
        assert (f.value, f.confidence, f.status) == (None, 0.0, "abstained")
        assert f.reasons == ["two totals printed", "ink smudge"]

    def test_abstain_helper_confidence_keyword(self) -> None:
        assert Field[str].abstain("unsure", confidence=0.3).confidence == 0.3

    def test_abstain_helper_requires_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="at least one reason"):
            Field[str].abstain()


class TestFieldBasics:
    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2])
    def test_confidence_bounds(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            Field[str].model_validate(extracted("x", confidence=bad))

    @pytest.mark.parametrize("ok", [0, 0.0, 0.5, 1, 1.0])
    def test_confidence_bounds_inclusive(self, ok: float) -> None:
        assert Field[str].model_validate(extracted("x", confidence=ok)).confidence == ok

    def test_page_must_be_one_based(self) -> None:
        with pytest.raises(ValidationError):
            Field[str].model_validate(extracted("x", page=0))

    def test_unknown_keys_are_ignored(self) -> None:
        f = Field[str].model_validate(extracted("x", rationale="because"))
        assert not hasattr(f, "rationale")

    def test_decimal_from_float_does_not_pick_up_binary_noise(self) -> None:
        assert Field[Decimal].model_validate(extracted(12.1)).value == Decimal("12.1")

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
    def test_extracted_row(self) -> None:
        item = LineItem.model_validate(row())
        assert item.status == "extracted"
        assert item.quantity == Decimal(2)
        assert item.unit_price == Decimal("4.50")
        assert item.amount == Decimal("9.0")
        assert item.reasons == []
        assert item.flagged_cells == []

    def test_cells_may_be_null_on_an_extracted_row(self) -> None:
        item = LineItem.model_validate(
            {"description": "Widget", "status": "extracted", "confidence": 0.7}
        )
        assert item.quantity is None
        assert item.unit_price is None
        assert item.amount is None

    def test_status_is_required(self) -> None:
        payload = row()
        del payload["status"]
        with pytest.raises(ValidationError):
            LineItem.model_validate(payload)

    def test_row_confidence_is_required_and_bounded(self) -> None:
        payload = row()
        del payload["confidence"]
        with pytest.raises(ValidationError):
            LineItem.model_validate(payload)
        with pytest.raises(ValidationError):
            LineItem.model_validate(row(confidence=1.5))

    def test_extracted_row_needs_a_cell(self) -> None:
        with pytest.raises(ValidationError, match="at least one non-null cell"):
            LineItem.model_validate({"status": "extracted", "confidence": 0.9})

    @pytest.mark.parametrize("status", ["absent", "abstained"])
    def test_non_extracted_row_must_have_all_cells_null(self, status: str) -> None:
        with pytest.raises(ValidationError, match=f"an {status} row must have all cells null"):
            LineItem.model_validate(
                {"description": "Widget", "status": status, "confidence": 0.5, "reasons": ["r"]}
            )

    def test_absent_row_with_no_cells(self) -> None:
        item = LineItem.model_validate({"status": "absent", "confidence": 0.9})
        assert item.status == "absent"

    def test_abstained_row_requires_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="at least one reason"):
            LineItem.model_validate({"status": "abstained", "confidence": 0.1})
        item = LineItem.model_validate(
            {"status": "abstained", "confidence": 0.1, "reasons": ["row unreadable"]}
        )
        assert item.reasons == ["row unreadable"]

    def test_flagged_cells_are_accepted_on_input(self) -> None:
        # the model is never asked for it, but if it sends one we parse it (validate/ overwrites)
        item = LineItem.model_validate(row(flagged_cells=["amount"]))
        assert item.flagged_cells == ["amount"]

    def test_flagged_cells_are_hidden_from_the_llm_schema(self) -> None:
        schema = DocumentExtraction.model_json_schema()
        assert "flagged_cells" not in schema["$defs"]["LineItem"]["properties"]
        assert "flagged_cells" not in json.dumps(simplify_schema(schema))
        # the other row fields are still there, including the new status and reasons
        assert {"status", "reasons", "confidence", "evidence", "page"} <= set(
            schema["$defs"]["LineItem"]["properties"]
        )

    def test_flagged_cells_still_round_trip_through_json(self) -> None:
        item = LineItem.model_validate(row(flagged_cells=["quantity", "amount"]))
        assert LineItem.model_validate_json(item.model_dump_json()).flagged_cells == [
            "quantity",
            "amount",
        ]


class TestDocumentExtraction:
    def test_parses_full_payload(self) -> None:
        doc = DocumentExtraction.model_validate(full_payload())
        assert doc.vendor_name.value == "ACME Corp"
        assert doc.vendor_name.evidence == "ACME Corp"
        assert doc.vendor_tax_id.status == "absent"
        assert doc.payment_method.status == "abstained"
        assert doc.payment_method.reasons == ["smudged"]
        assert doc.document_date.value == date(2024, 3, 5)
        assert doc.due_date.value is None
        assert doc.due_date.confidence == 0.7
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

    def test_status_violation_inside_document_is_reported_with_its_path(self) -> None:
        payload = full_payload()
        payload["vendor_name"] = abstained(reasons=[])
        with pytest.raises(ValidationError) as exc:
            DocumentExtraction.model_validate(payload)
        assert exc.value.errors()[0]["loc"] == ("vendor_name",)

    def test_invalid_row_inside_document_is_reported_with_its_path(self) -> None:
        payload = full_payload()
        payload["line_items"] = [row(), {"status": "extracted", "confidence": 0.9}]
        with pytest.raises(ValidationError) as exc:
            DocumentExtraction.model_validate(payload)
        assert exc.value.errors()[0]["loc"] == ("line_items", 1)

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

    def test_status_enum_in_the_schema_has_all_three_values(self) -> None:
        text = json.dumps(DocumentExtraction.model_json_schema())
        assert '"enum": ["extracted", "absent", "abstained"]' in text
