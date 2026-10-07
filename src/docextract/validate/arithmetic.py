"""Arithmetic checks: row-level (quantity x unit_price should equal amount) and document-level
(sum of the lines should equal the subtotal; the summary should add up to the total).

This is a business rule, so it lives here and not in the Pydantic schema: the schema only
defines shape, and the model's raw output stays intact so we can always see what the model
said versus what the validators concluded. Every function here returns a copy and never
mutates its input. Always validate from the raw parsed output.

A row mismatch:
  - flags quantity, unit_price and amount (we cannot tell which of the three is wrong),
  - adds the reason code `qty_x_price_ne_amount` to the row,
  - lowers the row's confidence to at most MISMATCH_CONFIDENCE_CAP.
It does NOT change the row's status: some receipts legitimately print a per-line discount on
the next line, so a mismatch alone is not grounds to abstain.

The document reconciliation is the same policy on the summary fields: a printed document can
legitimately not add up (a discount printed as a row, a rounding line), so a mismatch adds its
reason code to the summary fields involved and caps their confidence, but never changes a
status. The summary fields carry the aggregate claims; the rows are judged cell by cell by the
row check above.

`flagged_cells` and the reason codes are owned by this module. Whatever the model sent for
them is discarded and recomputed, so re-validating gives the same result.
"""

from decimal import Decimal

from docextract.extract.schema import DocumentExtraction, Field, LineItem
from docextract.validate.codes import (
    COMPONENTS_NE_TOTAL,
    LINES_NE_SUBTOTAL,
    QTY_X_PRICE_NE_AMOUNT,
)

MISMATCH_CELLS: tuple[str, ...] = ("quantity", "unit_price", "amount")

# Tolerance is max(ABSOLUTE, RELATIVE x |amount|). The relative part matters for currencies
# that round amounts (Rupiah) and for items sold by weight: 0.352 kg x 25,900 = 9,116.8 is
# printed as 9,117. Real extraction errors (a dropped digit, a quantity read as the price)
# miss by far more than 0.5%.
ABSOLUTE_TOLERANCE = Decimal("0.01")
RELATIVE_TOLERANCE = Decimal("0.005")

# Placeholder, to be calibrated in the confidence milestone. A cap rather than a multiplier so
# that validating twice cannot lower the confidence twice.
MISMATCH_CONFIDENCE_CAP = 0.5


def tolerance(amount: Decimal) -> Decimal:
    return max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * abs(amount))


def has_arithmetic_mismatch(item: LineItem) -> bool:
    """True only when all three cells are present and disagree beyond the tolerance."""
    quantity, unit_price, amount = item.quantity, item.unit_price, item.amount
    if quantity is None or unit_price is None or amount is None:
        return False
    return abs(quantity * unit_price - amount) > tolerance(amount)


def validate_line_item(item: LineItem) -> LineItem:
    reasons = [reason for reason in item.reasons if reason != QTY_X_PRICE_NE_AMOUNT]
    if not has_arithmetic_mismatch(item):
        return item.model_copy(update={"flagged_cells": [], "reasons": reasons}, deep=True)
    return item.model_copy(
        update={
            "flagged_cells": list(MISMATCH_CELLS),
            "reasons": [*reasons, QTY_X_PRICE_NE_AMOUNT],
            "confidence": min(item.confidence, MISMATCH_CONFIDENCE_CAP),
        },
        deep=True,
    )


def validate_line_items(document: DocumentExtraction) -> DocumentExtraction:
    rows = [validate_line_item(item) for item in document.line_items]
    return document.model_copy(update={"line_items": rows}, deep=True)


# ---------------------------------------------------------------- document totals


def _flag_summary(document: DocumentExtraction, name: str, code: str) -> DocumentExtraction:
    """One summary field with `code` in its reasons and its confidence at most the cap."""
    field: Field[Decimal] = getattr(document, name)
    reasons = [reason for reason in field.reasons if reason != code]
    flagged = field.model_copy(
        update={
            "reasons": [*reasons, code],
            "confidence": min(field.confidence, MISMATCH_CONFIDENCE_CAP),
        },
        deep=True,
    )
    return document.model_copy(update={name: flagged}, deep=True)


def summed_amounts(document: DocumentExtraction) -> Decimal | None:
    """The sum of the row amounts, or None when the rows cannot be summed: no rows at all, or
    a row without an amount (an equation over an incomplete list would be a guess)."""
    if not document.line_items:
        return None
    amounts = [item.amount for item in document.line_items]
    if any(amount is None for amount in amounts):
        return None
    return sum(amounts, start=Decimal(0))


def validate_document_totals(document: DocumentExtraction) -> DocumentExtraction:
    """The printed summary should add up. Both equations are skipped unless every participant
    is decided: `absent` money counts as 0 (the document does not state it), but an `abstained`
    one could be anything, so the equation is not done at all.

    - `lines_ne_subtotal`: the summed row amounts vs the subtotal (needs extracted rows and an
      extracted subtotal). Blames the subtotal.
    - `components_ne_total`: subtotal + tax - discount vs the total (a subtotal the document
      does not print is replaced by the summed rows). Blames every extracted participant.
    """
    result = document
    lines_sum = summed_amounts(result)
    subtotal = result.subtotal
    if subtotal.status == "extracted" and lines_sum is not None:
        assert subtotal.value is not None  # an extracted field always has a value
        if abs(lines_sum - subtotal.value) > tolerance(subtotal.value):
            result = _flag_summary(result, "subtotal", LINES_NE_SUBTOTAL)

    total = result.total
    if total.status != "extracted":
        return result
    assert total.value is not None
    if subtotal.status == "extracted":
        assert subtotal.value is not None
        base: Decimal | None = subtotal.value
    else:
        base = lines_sum
    tax, discount = result.tax, result.discount
    if base is None or tax.status == "abstained" or discount.status == "abstained":
        return result
    expected = base
    if tax.status == "extracted":
        assert tax.value is not None
        expected += tax.value
    if discount.status == "extracted":
        assert discount.value is not None
        expected -= discount.value
    if abs(expected - total.value) > tolerance(total.value):
        for name in ("subtotal", "tax", "discount", "total"):
            if getattr(result, name).status == "extracted":
                result = _flag_summary(result, name, COMPONENTS_NE_TOTAL)
    return result
