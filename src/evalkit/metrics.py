"""Scoring metrics over matched fields.

One labeled field, or one cell of a matched row pair, ends in exactly one outcome:

  correct           the value matches, or the label says absent and the prediction says absent
  value_wrong       both sides have a value and they disagree
  should_be_absent  a value where the label says the document does not state the field
  should_be_value   `absent` where the label has a value
  abstained         the prediction abstained (scored as an error, reported apart)
  no_prediction     the document was not extracted, so no answer exists

A document's fields are scored by "both numbers": `accuracy` (the headline) counts abstentions and
missing predictions as errors, while `selective_accuracy` looks only at what was answered and
`coverage` says how much that was. Every labeled field is in exactly one bucket, so
`labeled = correct + value_wrong + should_be_absent + should_be_value + abstained + no_prediction`.
A ratio over nothing (no labeled field, or nothing answered) is defined as 1.0: there is nothing
wrong to report. Ground truth `"__unlabeled__"` is skipped everywhere and is never a denominator.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from docextract.extract.schema import SCALAR_FIELD_NAMES, DocumentExtraction, Field, LineItem
from evalkit.comparators import CELL_KIND, FIELD_KIND, compare_field
from evalkit.groundtruth import GroundTruth, GTLineItem, is_unlabeled
from evalkit.matching import RowMatch, RowPair, match_rows
from evalkit.taxonomy import ErrorClass, FieldOutcome, classify_field, count_errors

OUTCOMES: tuple[FieldOutcome, ...] = (
    "correct",
    "value_wrong",
    "should_be_absent",
    "should_be_value",
    "abstained",
    "no_prediction",
)


def _ratio(part: int, whole: int) -> float:
    """A ratio over nothing is 1.0: with no labeled answer there is nothing wrong."""
    return part / whole if whole else 1.0


# ---------------------------------------------------------------- fields


@dataclass(frozen=True)
class FieldScore:
    """One labeled field's verdict. `predicted_status` is None when the document was not run."""

    name: str
    expected: Any
    predicted: Any
    predicted_status: str | None
    outcome: FieldOutcome
    error_class: ErrorClass | None
    how: str | None = None
    similarity: float | None = None

    @property
    def correct(self) -> bool:
        return self.outcome == "correct"


def score_field(name: str, expected: Any, prediction: Field[Any] | None) -> FieldScore:
    """Score one labeled field against its prediction (None = the document was not extracted).
    `expected` must be a labeled value or an explicit None (absent)."""
    outcome: FieldOutcome
    if prediction is None:
        predicted = None
        predicted_status = None
        how = similarity = None
        outcome = "no_prediction"
    else:
        predicted = prediction.value
        predicted_status = prediction.status
        how = similarity = None
        if prediction.status == "abstained":
            outcome = "abstained"
        elif expected is None:
            outcome = "correct" if prediction.status == "absent" else "should_be_absent"
        elif prediction.status == "absent":
            outcome = "should_be_value"
        else:
            comparison = compare_field(name, expected, predicted)
            outcome = "correct" if comparison.matched else "value_wrong"
            how, similarity = comparison.how, comparison.similarity
    return FieldScore(
        name=name,
        expected=expected,
        predicted=predicted,
        predicted_status=predicted_status,
        outcome=outcome,
        error_class=classify_field(
            outcome, FIELD_KIND[name], similarity=similarity, expected_absent=expected is None
        ),
        how=how,
        similarity=similarity,
    )


@dataclass(frozen=True)
class FieldMetrics:
    """The field counts of one document or of a whole run, with the ratios derived from them."""

    labeled: int = 0
    correct: int = 0
    value_wrong: int = 0
    should_be_absent: int = 0
    should_be_value: int = 0
    abstained: int = 0
    no_prediction: int = 0

    @classmethod
    def from_outcomes(cls, outcomes: Iterable[FieldOutcome]) -> FieldMetrics:
        counts = dict.fromkeys(OUTCOMES, 0)
        labeled = 0
        for outcome in outcomes:
            counts[outcome] += 1
            labeled += 1
        return cls(labeled=labeled, **counts)

    @classmethod
    def merge(cls, parts: Iterable[FieldMetrics]) -> FieldMetrics:
        parts = list(parts)
        names = ("labeled", *OUTCOMES)
        return cls(**{name: sum(getattr(part, name) for part in parts) for name in names})

    @property
    def answered(self) -> int:
        return self.labeled - self.abstained - self.no_prediction

    @property
    def accuracy(self) -> float:
        return _ratio(self.correct, self.labeled)

    @property
    def selective_accuracy(self) -> float:
        return _ratio(self.correct, self.answered)

    @property
    def coverage(self) -> float:
        return _ratio(self.answered, self.labeled)

    @property
    def abstention_rate(self) -> float:
        return _ratio(self.abstained, self.labeled)

    @property
    def failure_rate(self) -> float:
        return _ratio(self.no_prediction, self.labeled)


# ---------------------------------------------------------------- rows


@dataclass(frozen=True)
class RowsScore:
    """The row outcome of one document: how many rows paired, and how well the cells agreed."""

    labeled: bool = False
    no_itemization: bool = False
    matched: int = 0
    missed: int = 0
    spurious: int = 0
    perfect: int = 0
    cells_compared: int = 0
    cells_correct: int = 0
    errors: tuple[ErrorClass, ...] = ()

    @classmethod
    def merge(cls, parts: Iterable[RowsScore]) -> RowsScore:
        parts = list(parts)
        names = ("matched", "missed", "spurious", "perfect", "cells_compared", "cells_correct")
        return cls(
            labeled=any(part.labeled for part in parts),
            errors=tuple(error for part in parts for error in part.errors),
            **{name: sum(getattr(part, name) for part in parts) for name in names},
        )

    @property
    def row_precision(self) -> float:
        return _ratio(self.matched, self.matched + self.spurious)

    @property
    def row_recall(self) -> float:
        return _ratio(self.matched, self.matched + self.missed)

    @property
    def row_f1(self) -> float:
        precision, recall = self.row_precision, self.row_recall
        return _ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0

    @property
    def row_accuracy(self) -> float:
        return _ratio(self.perfect, self.matched)

    @property
    def cell_accuracy(self) -> float:
        return _ratio(self.cells_correct, self.cells_compared)


def _cell_outcome(expected: Any, predicted: Any, matched: bool) -> FieldOutcome:
    if expected is None or predicted is None:
        if expected is None and predicted is None:
            return "correct"
        return "should_be_absent" if expected is None else "should_be_value"
    return "correct" if matched else "value_wrong"


def _score_pairs(
    pairs: Sequence[RowPair], expected: Sequence[GTLineItem], predicted: Sequence[LineItem]
) -> tuple[int, int, int, list[ErrorClass]]:
    """(compared cells, matching cells, perfect rows, cell errors) over matched pairs."""
    compared = correct = perfect = 0
    errors: list[ErrorClass] = []
    for pair in pairs:
        compared += pair.compared_cells
        correct += pair.compared_cells - pair.cost
        if pair.cost == 0:
            perfect += 1
        for cell, verdict in pair.cells.items():
            if verdict.matched:
                continue
            outcome = _cell_outcome(
                getattr(expected[pair.expected], cell),
                getattr(predicted[pair.predicted], cell),
                verdict.matched,
            )
            error = classify_field(
                outcome,
                CELL_KIND[cell],
                similarity=verdict.similarity,
                expected_absent=getattr(expected[pair.expected], cell) is None,
            )
            if error is not None:
                errors.append(error)
    return compared, correct, perfect, errors


def score_rows(expected: Any, predicted: Sequence[LineItem] | None) -> RowsScore:
    """Score a label's `line_items` (`expected`) against the predicted rows (`None` = not run)."""
    if is_unlabeled(expected):
        return RowsScore()
    rows: Sequence[LineItem] = predicted or []
    if expected is None:  # labeled: this document has no itemization
        return RowsScore(
            labeled=True,
            no_itemization=True,
            spurious=len(rows),
            errors=("row_spurious",) * len(rows),
        )
    if predicted is None:
        return RowsScore(labeled=True, missed=len(expected), errors=("row_missed",) * len(expected))
    match: RowMatch = match_rows(rows, expected)
    compared, correct, perfect, cell_errors = _score_pairs(match.pairs, expected, rows)
    errors: list[ErrorClass] = [
        *["row_missed"] * len(match.missed),
        *["row_spurious"] * len(match.spurious),
        *cell_errors,
    ]
    return RowsScore(
        labeled=True,
        matched=match.matched,
        missed=len(match.missed),
        spurious=len(match.spurious),
        perfect=perfect,
        cells_compared=compared,
        cells_correct=correct,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------- a whole document


@dataclass(frozen=True)
class DocumentScore:
    """One document's result: its labeled fields, its rows, and the counts derived from both."""

    doc_id: str
    owner: str
    status: str
    fields: list[FieldScore]
    rows: RowsScore
    errors: dict[str, int]

    @property
    def metrics(self) -> FieldMetrics:
        return FieldMetrics.from_outcomes(score.outcome for score in self.fields)


def score_document(
    label: GroundTruth,
    *,
    owner: str,
    status: str,
    document: DocumentExtraction | None,
) -> DocumentScore:
    """Score one labeled document. `document` is the ok record's `validated` (None = not run)."""
    fields = []
    for name in SCALAR_FIELD_NAMES:
        expected = getattr(label, name)
        if is_unlabeled(expected):
            continue
        prediction = None if document is None else getattr(document, name)
        fields.append(score_field(name, expected, prediction))
    rows = score_rows(label.line_items, None if document is None else list(document.line_items))
    errors = count_errors([*(score.error_class for score in fields), *rows.errors])
    return DocumentScore(
        doc_id=label.doc_id,
        owner=owner,
        status=status,
        fields=fields,
        rows=rows,
        errors=errors,
    )
