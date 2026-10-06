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

A label also records HOW it was made: `label_mode` ("prelabel": a model drafted it and a human
verified every field; "blind": a human labeled from the page alone, no draft ever shown) and
`labeled_at` (when, timezone-aware). `field_notes` is an optional {field: note} for the humans
reading the label ("redacted", "total not printed"); notes are never scored.

"__unlabeled__" is a reserved string: it can never be a real field value.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

UNLABELED: Final = "__unlabeled__"

LabelMode = Literal["prelabel", "blind"]

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
    label_mode: LabelMode | None = None  # None: generated labels (constructed documents)
    labeled_at: datetime | None = None  # timezone-aware
    notes: str | None = None
    field_notes: dict[str, str] = Field(default_factory=dict)

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

    @field_validator("labeled_at")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("labeled_at needs a timezone (write it as 2026-10-06T09:15:00Z)")
        return value

    @field_validator("field_notes")
    @classmethod
    def _known_fields_and_real_notes(cls, notes: dict[str, str]) -> dict[str, str]:
        unknown = sorted(name for name in notes if name not in NOTE_FIELDS)
        if unknown:
            raise ValueError(f"field_notes for unknown field(s): {', '.join(unknown)}")
        blank = sorted(name for name, note in notes.items() if not note.strip())
        if blank:
            raise ValueError(f"empty field_notes for: {', '.join(blank)}")
        return notes

    def labeled_fields(self) -> dict[str, Any]:
        """Scalar fields that carry a label (a value or explicit null), keyed by name."""
        return {
            name: value
            for name in SCALAR_FIELD_NAMES
            if not is_unlabeled(value := getattr(self, name))
        }


_NON_SCALAR = {
    "doc_id",
    "source_file",
    "notes",
    "line_items",
    "label_mode",
    "labeled_at",
    "field_notes",
}

# Same names as docextract.extract.schema.SCALAR_FIELD_NAMES (a test keeps them in sync).
SCALAR_FIELD_NAMES: tuple[str, ...] = tuple(
    name for name in GroundTruth.model_fields if name not in _NON_SCALAR
)

# the fields a note can be about: the scalars and the line items as a whole
NOTE_FIELDS: tuple[str, ...] = (*SCALAR_FIELD_NAMES, "line_items")
