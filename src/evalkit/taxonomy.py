"""Error taxonomy for classifying extraction failures.

Every labeled field a run gets wrong lands in exactly one class here, so a scorecard can say not
just how much was wrong but what kind of wrong it was (and M6 can diff the mix between runs). The
classes are the vocabulary the metrics module classifies into; a correct answer has no class.

  numeric_mismatch     a number (or quantity) outside the comparator's tolerance
  date_mismatch        a different date
  string_mismatch      a different string (address, payment method, document number, tax id)
  name_near_miss       a vendor/buyer/description that came close (< FUZZY_NAME_MIN but >= 0.5)
  hallucinated_value   a value where the label says the field is absent
  missing_value        `absent` where the label has a value
  abstained_on_value   the model abstained on a field that has a value
  abstained_on_absent  the model abstained on a field the label calls absent
  no_prediction        the document was not extracted (failed, refused, not run)
  row_missed           a labeled row with no predicted counterpart
  row_spurious         a predicted row with no labeled counterpart
"""

from collections.abc import Iterable
from typing import Literal

from evalkit.comparators import Kind

FieldOutcome = Literal[
    "correct",
    "value_wrong",
    "should_be_absent",
    "should_be_value",
    "abstained",
    "no_prediction",
]

ErrorClass = Literal[
    "numeric_mismatch",
    "date_mismatch",
    "string_mismatch",
    "name_near_miss",
    "hallucinated_value",
    "missing_value",
    "abstained_on_value",
    "abstained_on_absent",
    "no_prediction",
    "row_missed",
    "row_spurious",
]

ERROR_CLASSES: tuple[ErrorClass, ...] = (
    "numeric_mismatch",
    "date_mismatch",
    "string_mismatch",
    "name_near_miss",
    "hallucinated_value",
    "missing_value",
    "abstained_on_value",
    "abstained_on_absent",
    "no_prediction",
    "row_missed",
    "row_spurious",
)

# Below this a name mismatch is just a string mismatch: a near miss is worth telling apart.
NEAR_MISS_MIN = 0.5

_WRONG_KIND: dict[Kind, ErrorClass] = {
    "number": "numeric_mismatch",
    "date": "date_mismatch",
    "text": "string_mismatch",
    "code": "string_mismatch",
    "name": "string_mismatch",  # name_near_miss when the similarity is close enough
}


def classify_field(
    outcome: FieldOutcome,
    kind: Kind,
    *,
    similarity: float | None = None,
    expected_absent: bool = False,
) -> ErrorClass | None:
    """The class of a wrong field (or of one cell of a matched row), or None when it is correct."""
    if outcome == "correct":
        return None
    if outcome == "value_wrong":
        if kind == "name" and similarity is not None and similarity >= NEAR_MISS_MIN:
            return "name_near_miss"
        return _WRONG_KIND[kind]
    if outcome == "should_be_absent":
        return "hallucinated_value"
    if outcome == "should_be_value":
        return "missing_value"
    if outcome == "abstained":
        return "abstained_on_absent" if expected_absent else "abstained_on_value"
    return "no_prediction"


def count_errors(classes: Iterable[ErrorClass | None]) -> dict[str, int]:
    """How many of each class, in vocabulary order, with the empty classes left out."""
    tally = dict.fromkeys(ERROR_CLASSES, 0)
    for name in classes:
        if name is not None:
            tally[name] += 1
    return {name: count for name, count in tally.items() if count}
