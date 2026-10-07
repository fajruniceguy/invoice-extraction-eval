"""Line-item arithmetic check: quantity x unit_price should equal amount.

This is a business rule, so it lives here and not in the Pydantic schema: the schema only
defines shape, and the model's raw output stays intact so we can always see what the model
said versus what the validators concluded. Every function here returns a copy and never
mutates its input. Always validate from the raw parsed output.

A mismatch:
  - flags quantity, unit_price and amount (we cannot tell which of the three is wrong),
  - adds the reason code `qty_x_price_ne_amount` to the row,
  - lowers the row's confidence to at most MISMATCH_CONFIDENCE_CAP.
It does NOT change the row's status: some receipts legitimately print a per-line discount on
the next line, so a mismatch alone is not grounds to abstain.

`flagged_cells` and the reason code are owned by this module. Whatever the model sent for
them is discarded and recomputed, so re-validating a row gives the same result.
"""

from decimal import Decimal

from docextract.extract.schema import DocumentExtraction, LineItem
from docextract.validate.codes import QTY_X_PRICE_NE_AMOUNT

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
