"""Confidence and abstention (M4).

`formula` turns the model's own confidence into a final one by lowering it with the caps its
reason codes imply; `abstain` decides which fields that leaves unclaimable. Both work on
copies: what the model said stays intact in the extraction record.
"""

from docextract.confidence.abstain import apply_abstention, has_hard_reason, should_abstain
from docextract.confidence.formula import (
    ARITHMETIC_CAP,
    CAP_BY_CODE,
    NORMALIZED_CAP,
    SHAPE_CAP,
    apply_confidence,
    cap_for,
)

__all__ = [
    "ARITHMETIC_CAP",
    "CAP_BY_CODE",
    "NORMALIZED_CAP",
    "SHAPE_CAP",
    "apply_abstention",
    "apply_confidence",
    "cap_for",
    "has_hard_reason",
    "should_abstain",
]
