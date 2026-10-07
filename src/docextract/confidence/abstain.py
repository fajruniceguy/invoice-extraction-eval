"""The abstention policy: what cannot be claimed is abstained, with the reason why.

An `extracted` field or row becomes `abstained` — value (or every cell) null, the evidence and
page kept, `status = abstained`, at least one reason — when

  - a HARD check failed (`validate.codes.HARD_CODES`: the value is not grounded in its own
    evidence, or the due date precedes the document date): its own reason code is already
    recorded, or
  - its final confidence is below `ABSTAIN_BELOW`: then `confidence_below_threshold` is added.

`absent` is never flipped: "the document does not state it" is a claim of its own, and a
`abstained` field is left alone, so running the policy twice gives the same result.
"""

from collections.abc import Sequence

from docextract.extract.schema import CELL_NAMES, SCALAR_FIELD_NAMES, DocumentExtraction
from docextract.validate.codes import CONFIDENCE_BELOW_THRESHOLD, HARD_CODES
from docextract.validate.reasons import with_document_reason, with_row, with_row_reason


def has_hard_reason(reasons: Sequence[str]) -> bool:
    """True when one of the reasons is a contradiction (see `validate.codes.HARD_CODES`)."""
    return any(reason in HARD_CODES for reason in reasons)


def should_abstain(reasons: Sequence[str], confidence: float, abstain_below: float) -> bool:
    """A contradicted field (a hard code) or one whose confidence is below the threshold."""
    return has_hard_reason(reasons) or confidence < abstain_below


def apply_abstention(document: DocumentExtraction, *, abstain_below: float) -> DocumentExtraction:
    """Abstain every extracted field and row the policy cannot claim."""
    result = document
    for name in SCALAR_FIELD_NAMES:
        field = getattr(result, name)
        if field.status != "extracted" or not should_abstain(
            field.reasons, field.confidence, abstain_below
        ):
            continue
        abstained = field.model_copy(update={"value": None, "status": "abstained"}, deep=True)
        result = result.model_copy(update={name: abstained}, deep=True)
        if not has_hard_reason(field.reasons):
            result = with_document_reason(result, name, CONFIDENCE_BELOW_THRESHOLD)

    for index in range(len(result.line_items)):
        row = result.line_items[index]
        if row.status != "extracted" or not should_abstain(
            row.reasons, row.confidence, abstain_below
        ):
            continue
        update: dict[str, object] = {cell: None for cell in CELL_NAMES}
        update["status"] = "abstained"
        result = with_row(result, index, row.model_copy(update=update, deep=True))
        if not has_hard_reason(row.reasons):
            result = with_row_reason(result, index, CONFIDENCE_BELOW_THRESHOLD)
    return result
