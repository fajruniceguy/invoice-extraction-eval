import json
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from docextract.extract.schema import SCALAR_FIELD_NAMES as EXTRACTION_FIELD_NAMES
from evalkit.groundtruth import (
    SCALAR_FIELD_NAMES,
    UNLABELED,
    GroundTruth,
    GTLineItem,
    is_unlabeled,
)


def test_sentinel_value() -> None:
    assert UNLABELED == "__unlabeled__"


def test_scalar_field_names_match_extraction_schema() -> None:
    assert SCALAR_FIELD_NAMES == EXTRACTION_FIELD_NAMES


class TestIsUnlabeled:
    def test_sentinel(self) -> None:
        assert is_unlabeled(UNLABELED)

    @pytest.mark.parametrize("value", [None, "", "ACME", 0, Decimal(0), date(2024, 1, 1), []])
    def test_everything_else_is_a_label(self, value: object) -> None:
        assert not is_unlabeled(value)


class TestTriState:
    def test_omitted_keys_are_unlabeled(self) -> None:
        gt = GroundTruth.model_validate({"doc_id": "d1"})
        for name in SCALAR_FIELD_NAMES:
            assert is_unlabeled(getattr(gt, name)), name
        assert is_unlabeled(gt.line_items)

    def test_explicit_null_is_labeled_absent_not_unlabeled(self) -> None:
        gt = GroundTruth.model_validate({"doc_id": "d1", "due_date": None, "tax": None})
        assert gt.due_date is None
        assert gt.tax is None
        assert not is_unlabeled(gt.due_date)

    def test_values_are_typed(self) -> None:
        gt = GroundTruth.model_validate(
            {
                "doc_id": "d1",
                "vendor_name": "ACME",
                "document_date": "2024-03-05",
                "total": "9.72",
                "tax": 0.72,
            }
        )
        assert gt.vendor_name == "ACME"
        assert gt.document_date == date(2024, 3, 5)
        assert gt.total == Decimal("9.72")
        assert gt.tax == Decimal("0.72")

    def test_explicit_sentinel_string_is_accepted_for_every_type(self) -> None:
        gt = GroundTruth.model_validate(
            {
                "doc_id": "d1",
                "vendor_name": UNLABELED,
                "document_date": UNLABELED,
                "total": UNLABELED,
                "line_items": UNLABELED,
            }
        )
        assert is_unlabeled(gt.vendor_name)
        assert is_unlabeled(gt.document_date)
        assert is_unlabeled(gt.total)
        assert is_unlabeled(gt.line_items)

    def test_garbage_is_an_error_not_silently_unlabeled(self) -> None:
        with pytest.raises(ValidationError):
            GroundTruth.model_validate({"doc_id": "d1", "total": "not a number"})
        with pytest.raises(ValidationError):
            GroundTruth.model_validate({"doc_id": "d1", "document_date": "yesterday"})

    def test_json_round_trip_keeps_the_three_states_apart(self) -> None:
        gt = GroundTruth.model_validate(
            {"doc_id": "d1", "vendor_name": "ACME", "due_date": None, "total": "9.72"}
        )
        data = json.loads(gt.model_dump_json())
        assert data["vendor_name"] == "ACME"
        assert data["due_date"] is None
        assert data["buyer_name"] == UNLABELED
        again = GroundTruth.model_validate_json(gt.model_dump_json())
        assert again == gt
        assert again.due_date is None
        assert is_unlabeled(again.buyer_name)


class TestStrictness:
    def test_typo_in_field_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="vendor_nme"):
            GroundTruth.model_validate({"doc_id": "d1", "vendor_nme": "ACME"})

    def test_typo_in_line_item_cell_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GroundTruth.model_validate({"doc_id": "d1", "line_items": [{"amout": "1"}]})

    @pytest.mark.parametrize("bad", [None, "", 5])
    def test_doc_id_is_required_and_non_empty(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            GroundTruth.model_validate({"doc_id": bad})
        with pytest.raises(ValidationError):
            GroundTruth.model_validate({})


class TestLineItems:
    def test_omitted_is_unlabeled(self) -> None:
        assert is_unlabeled(GroundTruth(doc_id="d1").line_items)

    def test_null_is_labeled_no_itemization(self) -> None:
        gt = GroundTruth.model_validate({"doc_id": "d1", "line_items": None})
        assert gt.line_items is None

    def test_list_is_labeled_rows(self) -> None:
        gt = GroundTruth.model_validate(
            {
                "doc_id": "d1",
                "line_items": [
                    {"description": "Widget", "quantity": 2, "unit_price": "4.50", "amount": "9.00"}
                ],
            }
        )
        assert isinstance(gt.line_items, list)
        row = gt.line_items[0]
        assert row.description == "Widget"
        assert row.quantity == Decimal(2)
        assert row.amount == Decimal("9.00")

    def test_row_cells_are_tri_state_too(self) -> None:
        row = GTLineItem.model_validate({"description": "Widget", "quantity": None})
        assert row.description == "Widget"
        assert row.quantity is None  # labeled absent
        assert is_unlabeled(row.unit_price)  # omitted
        assert is_unlabeled(row.amount)


class TestLabeledFields:
    def test_returns_values_and_nulls_but_not_unlabeled(self) -> None:
        gt = GroundTruth.model_validate(
            {"doc_id": "d1", "vendor_name": "ACME", "due_date": None, "buyer_name": UNLABELED}
        )
        assert gt.labeled_fields() == {"vendor_name": "ACME", "due_date": None}

    def test_empty_when_nothing_is_labeled(self) -> None:
        assert GroundTruth(doc_id="d1").labeled_fields() == {}

    def test_never_includes_metadata_or_line_items(self) -> None:
        gt = GroundTruth.model_validate(
            {"doc_id": "d1", "notes": "n", "source_file": "a.png", "line_items": None}
        )
        assert gt.labeled_fields() == {}
