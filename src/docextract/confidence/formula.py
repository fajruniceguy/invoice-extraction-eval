"""The confidence formula: the model's own score, lowered by what the validators found.

    final = min(model score, every cap its reason codes imply, NORMALIZED_CAP if it was rewritten)

Caps are absolute ceilings, never multipliers, so applying the formula twice cannot lower a
field twice, and confidence is never raised. The model's number is the starting point: it is
the only signal about how sure the model was, and a cap says what we know better.

The cap values are placeholders to be calibrated on the dev split (S1), exactly like the row
mismatch cap in `validate.arithmetic`. `CONFIDENCE_BELOW_THRESHOLD` is deliberately absent:
it is not a finding about the value but the abstention outcome itself (see abstain.py).
"""

import re
from collections.abc import Sequence
from typing import Any

from docextract.extract.preprocess import change_label
from docextract.extract.schema import SCALAR_FIELD_NAMES, DocumentExtraction
from docextract.validate.arithmetic import MISMATCH_CONFIDENCE_CAP
from docextract.validate.codes import (
    COMPONENTS_NE_TOTAL,
    CURRENCY_CONTRADICTS_EVIDENCE,
    CURRENCY_UNKNOWN,
    DESCRIPTION_NOT_VERBATIM,
    DOCUMENT_DATE_IN_FUTURE,
    DUE_DATE_BEFORE_DOCUMENT_DATE,
    EVIDENCE_AMBIGUOUS,
    EVIDENCE_NOT_GROUNDED,
    LINES_NE_SUBTOTAL,
    QTY_X_PRICE_NE_AMOUNT,
    TAX_ID_IMPLAUSIBLE,
    TAX_ID_UNUSUAL_SHAPE,
)

# A contradiction: the arithmetic does not add up, or the value is contradicted by its own
# evidence. Same ceiling as the row arithmetic check has always applied.
ARITHMETIC_CAP = MISMATCH_CONFIDENCE_CAP
# A single-field shape or plausibility finding (a tax id of an unusual length, a currency we
# cannot read, a description that is not verbatim, a future document date, a number whose own
# quote is ambiguous and contains the value).
SHAPE_CAP = 0.7
# A value the normalizer had to rewrite: the model did not follow the format contract.
NORMALIZED_CAP = 0.85

CAP_BY_CODE: dict[str, float] = {
    QTY_X_PRICE_NE_AMOUNT: ARITHMETIC_CAP,
    LINES_NE_SUBTOTAL: ARITHMETIC_CAP,
    COMPONENTS_NE_TOTAL: ARITHMETIC_CAP,
    EVIDENCE_NOT_GROUNDED: ARITHMETIC_CAP,
    EVIDENCE_AMBIGUOUS: SHAPE_CAP,
    DUE_DATE_BEFORE_DOCUMENT_DATE: ARITHMETIC_CAP,
    DOCUMENT_DATE_IN_FUTURE: SHAPE_CAP,
    CURRENCY_UNKNOWN: SHAPE_CAP,
    CURRENCY_CONTRADICTS_EVIDENCE: SHAPE_CAP,
    TAX_ID_IMPLAUSIBLE: SHAPE_CAP,
    TAX_ID_UNUSUAL_SHAPE: SHAPE_CAP,
    DESCRIPTION_NOT_VERBATIM: SHAPE_CAP,
}

_ROW_TARGET = re.compile(r"line_items\[(\d+)\]\.\w+$")


def cap_for(reasons: Sequence[str]) -> float | None:
    """The lowest cap the reason codes imply, or None when they imply none."""
    caps = [CAP_BY_CODE[reason] for reason in reasons if reason in CAP_BY_CODE]
    return min(caps) if caps else None


def _normalized_targets(
    normalizations: Sequence[str],
) -> tuple[set[str], set[int]]:
    """Which scalar fields and which rows (0-based) a normalizer rewrote."""
    fields: set[str] = set()
    rows: set[int] = set()
    for change in normalizations:
        label = change_label(change)
        if label is None:
            continue
        row = _ROW_TARGET.fullmatch(label)
        if row is not None:
            rows.add(int(row.group(1)) - 1)  # the payload lists rows from 1
            continue
        name = label.removesuffix(".value")
        if name in SCALAR_FIELD_NAMES:
            fields.add(name)
    return fields, rows


def apply_confidence(
    document: DocumentExtraction, *, normalizations: Sequence[str] = ()
) -> DocumentExtraction:
    """Lower every field's and row's confidence to its caps. Never raises one, and running it
    again on its own output changes nothing."""
    normalized_fields, normalized_rows = _normalized_targets(normalizations)
    result = document
    for name in SCALAR_FIELD_NAMES:
        field: Any = getattr(result, name)
        cap = cap_for(field.reasons)
        if name in normalized_fields:
            cap = NORMALIZED_CAP if cap is None else min(cap, NORMALIZED_CAP)
        if cap is not None and cap < field.confidence:
            lowered = field.model_copy(update={"confidence": cap}, deep=True)
            result = result.model_copy(update={name: lowered}, deep=True)

    rows = [*result.line_items]
    for index, row in enumerate(rows):
        cap = cap_for(row.reasons)
        if index in normalized_rows:
            cap = NORMALIZED_CAP if cap is None else min(cap, NORMALIZED_CAP)
        if cap is not None and cap < row.confidence:
            rows[index] = row.model_copy(update={"confidence": cap}, deep=True)
    return result.model_copy(update={"line_items": rows}, deep=True)
