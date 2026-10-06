"""Extraction output schema.

Every scalar document field is wrapped in `Field[T]`, which carries the value together with
the model's confidence, a verbatim evidence quote, the source page and a three-way status:

  extracted  the document states it             value non-null
  absent     the document does not state it     value null; confidence = certainty it is
                                                absent (can be high); reasons optional
  abstained  may be there, but not confident    value null; reasons REQUIRED

Money is `Decimal` and dates are `date`. Line items carry one status/confidence/evidence/page
for the whole row rather than per cell (cheaper in tokens, and rows are the unit we match
and score).

This module only defines shape. Business rules (e.g. line-item arithmetic) live in
`docextract.validate`, so the model's raw output stays intact for debugging.

Unknown keys in model output are ignored here; the ground-truth schema is the strict one.
"""

from datetime import date
from decimal import Decimal
from typing import Generic, Literal, Self, TypeVar

from pydantic import BaseModel, model_validator
from pydantic import Field as PydanticField
from pydantic.json_schema import SkipJsonSchema

T = TypeVar("T")

FieldStatus = Literal["extracted", "absent", "abstained"]

CELL_NAMES: tuple[str, ...] = ("description", "quantity", "unit_price", "amount")


def _has_reason(reasons: list[str]) -> bool:
    return any(reason.strip() for reason in reasons)


class Field(BaseModel, Generic[T]):
    value: T | None
    confidence: float = PydanticField(ge=0.0, le=1.0)
    status: FieldStatus
    evidence: str | None = None  # verbatim quote from the document supporting the value
    page: int | None = PydanticField(default=None, ge=1)  # 1-based page of the evidence
    reasons: list[str] = PydanticField(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_value(self) -> Self:
        if self.status == "extracted":
            if self.value is None:
                raise ValueError("an extracted field must have a non-null value")
        elif self.value is not None:
            raise ValueError(f"an {self.status} field must have value=null")
        if self.status == "abstained" and not _has_reason(self.reasons):
            raise ValueError("an abstained field must give at least one reason")
        return self

    @classmethod
    def absent(cls, confidence: float, *reasons: str) -> Self:
        """The document does not state this field; `confidence` is how sure we are of that."""
        return cls(value=None, confidence=confidence, status="absent", reasons=list(reasons))

    @classmethod
    def abstain(cls, *reasons: str, confidence: float = 0.0) -> Self:
        """The field may be present but we are not confident; at least one reason is required."""
        return cls(value=None, confidence=confidence, status="abstained", reasons=list(reasons))


class LineItem(BaseModel):
    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None
    status: FieldStatus
    confidence: float = PydanticField(ge=0.0, le=1.0)
    evidence: str | None = None
    page: int | None = PydanticField(default=None, ge=1)
    reasons: list[str] = PydanticField(default_factory=list)
    # Derived by docextract.validate, never by the model: hidden from the JSON schema we send,
    # and anything the model sends here is ignored and recomputed there.
    flagged_cells: SkipJsonSchema[list[str]] = PydanticField(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_cells(self) -> Self:
        has_cells = any(getattr(self, name) is not None for name in CELL_NAMES)
        if self.status == "extracted":
            if not has_cells:
                raise ValueError("an extracted row must have at least one non-null cell")
        elif has_cells:
            raise ValueError(f"an {self.status} row must have all cells null")
        if self.status == "abstained" and not _has_reason(self.reasons):
            raise ValueError("an abstained row must give at least one reason")
        return self


class DocumentExtraction(BaseModel):
    vendor_name: Field[str]
    vendor_address: Field[str]
    vendor_tax_id: Field[str]
    buyer_name: Field[str]
    document_number: Field[str]
    document_date: Field[date]
    due_date: Field[date]
    currency: Field[str]
    payment_method: Field[str]
    line_items: list[LineItem] = PydanticField(default_factory=list)
    subtotal: Field[Decimal]
    tax: Field[Decimal]
    discount: Field[Decimal]
    total: Field[Decimal]


# Names of the `Field[T]`-wrapped document fields (everything except line_items).
SCALAR_FIELD_NAMES: tuple[str, ...] = tuple(
    name for name in DocumentExtraction.model_fields if name != "line_items"
)
