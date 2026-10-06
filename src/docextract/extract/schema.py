"""Extraction output schema.

Every scalar document field is wrapped in `Field[T]`, which carries the value together with
the model's confidence, a verbatim evidence quote, the source page and an explicit
extracted/abstained status. A field that is not on the document must be reported as
abstained (value null), not guessed.

Money is `Decimal` and dates are `date`. Line items carry one confidence/evidence/page for
the whole row rather than per cell (cheaper in tokens, and rows are the unit we match and
score).

Unknown keys in model output are ignored here; the ground-truth schema is the strict one.
"""

from datetime import date
from decimal import Decimal
from typing import Generic, Literal, Self, TypeVar

from pydantic import BaseModel, model_validator
from pydantic import Field as PydanticField

T = TypeVar("T")

FieldStatus = Literal["extracted", "abstained"]


class Field(BaseModel, Generic[T]):
    value: T | None
    confidence: float = PydanticField(ge=0.0, le=1.0)
    status: FieldStatus
    evidence: str | None = None  # verbatim quote from the document supporting the value
    page: int | None = PydanticField(default=None, ge=1)  # 1-based page of the evidence
    reasons: list[str] = PydanticField(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_value(self) -> Self:
        if self.status == "abstained" and self.value is not None:
            raise ValueError("an abstained field must have value=null")
        if self.status == "extracted" and self.value is None:
            raise ValueError("an extracted field must have a non-null value")
        return self

    @classmethod
    def abstain(cls, *reasons: str) -> Self:
        return cls(value=None, confidence=0.0, status="abstained", reasons=list(reasons))


class LineItem(BaseModel):
    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None
    confidence: float = PydanticField(ge=0.0, le=1.0)
    evidence: str | None = None
    page: int | None = PydanticField(default=None, ge=1)


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
