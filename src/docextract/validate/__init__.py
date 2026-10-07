"""Validation of one extraction: everything the model said is checked, capped and, where it
cannot be claimed, abstained — always on a copy, so the model's raw output stays inspectable.

    validate_document(parsed) -> row arithmetic -> document totals -> field checks
      -> evidence grounding -> confidence (caps from the recorded reason codes) -> abstention

Run it on the RAW parsed output (never on its own result; each step is idempotent but the
confidence caps are absolute ceilings, not multipliers).
"""

from collections.abc import Sequence
from datetime import date

from docextract.extract.schema import DocumentExtraction
from docextract.validate.arithmetic import validate_document_totals, validate_line_items
from docextract.validate.codes import REASON_CODES
from docextract.validate.fields import validate_fields
from docextract.validate.grounding import validate_grounding

__all__ = ["REASON_CODES", "validate_document"]


def validate_document(
    document: DocumentExtraction,
    *,
    abstain_below: float,
    normalizations: Sequence[str] = (),
    today: date | None = None,
) -> DocumentExtraction:
    """The whole chain, on a copy. `abstain_below` is `Settings.abstain_below`; `normalizations`
    are the rewrites the preprocess hook reported (`LLMResponse.normalizations`), which cap the
    fields they touched; `today` is injectable for the future-document-date check."""
    # Imported inside the function on purpose: docextract.confidence names this package's reason
    # codes, so a module-level import here would make the two packages' import order matter
    # (importing docextract.confidence first would meet a half-initialized docextract.validate).
    from docextract.confidence.abstain import apply_abstention
    from docextract.confidence.formula import apply_confidence

    result = validate_line_items(document)
    result = validate_document_totals(result)
    result = validate_fields(result, today=today)
    result = validate_grounding(result)
    result = apply_confidence(result, normalizations=normalizations)
    return apply_abstention(result, abstain_below=abstain_below)
