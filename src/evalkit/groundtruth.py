"""Ground-truth schema.

Each labeled field is one of three things:
  - a value                 -> the document states this; score the prediction against it
  - explicit null           -> "labeled absent": the document does NOT state it, so the
                               right answer is status "absent" (how an "abstained"
                               prediction is scored against it is decided in metrics)
  - "__unlabeled__"         -> nobody labeled this field; skip it in scoring

The default for every field is unlabeled, so an omitted JSON key means unlabeled, never
absent. Unknown keys are rejected so that a typo like "vendor_nme" fails loudly instead of
silently turning into an unlabeled field.

"__unlabeled__" is a reserved string: it can never be a real field value.
"""

from datetime import date
from decimal import Decimal
from typing import Any, Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

UNLABELED: Final = "__unlabeled__"

T = TypeVar("T")

# value | null (labeled absent) | sentinel (unlabeled)
GTValue = Literal["__unlabeled__"] | T | None


def is_unlabeled(value: object) -> bool:
    return isinstance(value, str) and value == UNLABELED


class GTLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: GTValue[str] = UNLABELED
    quantity: GTValue[Decimal] = UNLABELED
    unit_price: GTValue[Decimal] = UNLABELED
    amount: GTValue[Decimal] = UNLABELED


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    source_file: str | None = None
    notes: str | None = None

    vendor_name: GTValue[str] = UNLABELED
    vendor_address: GTValue[str] = UNLABELED
    vendor_tax_id: GTValue[str] = UNLABELED
    buyer_name: GTValue[str] = UNLABELED
    document_number: GTValue[str] = UNLABELED
    document_date: GTValue[date] = UNLABELED
    due_date: GTValue[date] = UNLABELED
    currency: GTValue[str] = UNLABELED
    payment_method: GTValue[str] = UNLABELED

    # null = labeled "no itemization"; a list = the labeled rows; sentinel = unlabeled.
    line_items: Literal["__unlabeled__"] | list[GTLineItem] | None = UNLABELED

    subtotal: GTValue[Decimal] = UNLABELED
    tax: GTValue[Decimal] = UNLABELED
    discount: GTValue[Decimal] = UNLABELED
    total: GTValue[Decimal] = UNLABELED

    def labeled_fields(self) -> dict[str, Any]:
        """Scalar fields that carry a label (a value or explicit null), keyed by name."""
        return {
            name: value
            for name in SCALAR_FIELD_NAMES
            if not is_unlabeled(value := getattr(self, name))
        }


_NON_SCALAR = {"doc_id", "source_file", "notes", "line_items"}

# Same names as docextract.extract.schema.SCALAR_FIELD_NAMES (a test keeps them in sync).
SCALAR_FIELD_NAMES: tuple[str, ...] = tuple(
    name for name in GroundTruth.model_fields if name not in _NON_SCALAR
)
