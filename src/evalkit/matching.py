"""Line-item matching between predicted and ground-truth rows.

Rows are paired by a hand-written Hungarian assignment (no scipy; a document has at most tens of
rows). The cost of pairing a predicted row with a labeled row is the number of cells that do not
agree, over the cells the label actually decided: a `"__unlabeled__"` cell is out of the cost, a
labeled `null` cell matches only a null cell, and the other cells go through
`evalkit.comparators.compare_cell`.

The assignment minimizes the total integer cost. It runs on a square, padded matrix (padding
entries cost 0 and are structural only: the number of real-real pairs a document can have is fixed
by its row counts, so padding can never change which rows pair). Equal-cost assignments always
resolve the same way: the algorithm's scan order is fixed and its comparisons are strict, which a
test pins on an all-equal-cost matrix.

After the assignment, a pair in which NO compared cell agrees is unwound into one missed row and
one spurious row, so a document's extra and missing rows are not confused with a row that was
found but misread.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from docextract.extract.schema import CELL_NAMES, LineItem
from evalkit.comparators import Comparison, compare_cell
from evalkit.groundtruth import GTLineItem, is_unlabeled


def cell_verdict(cell: str, expected: object, predicted: object) -> Comparison | None:
    """The verdict for one cell, or None when the label did not decide it (unlabeled)."""
    if is_unlabeled(expected):
        return None
    if expected is None or predicted is None:
        absent_matches = expected is None and predicted is None
        return Comparison(absent_matches, "exact" if absent_matches else None)
    return compare_cell(cell, expected, predicted)


def compare_rows(expected: GTLineItem, predicted: LineItem) -> dict[str, Comparison]:
    """The compared cells of one row pair, keyed by cell name (unlabeled cells omitted)."""
    verdicts: dict[str, Comparison] = {}
    for cell in CELL_NAMES:
        verdict = cell_verdict(cell, getattr(expected, cell), getattr(predicted, cell))
        if verdict is not None:
            verdicts[cell] = verdict
    return verdicts


def row_cost(cells: Mapping[str, Comparison]) -> int:
    """How many of the compared cells disagree."""
    return sum(1 for verdict in cells.values() if not verdict.matched)


def assign_min_cost(cost: Sequence[Sequence[int]]) -> list[int]:
    """The minimum-cost assignment of a square cost matrix: row index -> column index.

    The Jonker-Volgenant potentials form of the Hungarian algorithm, O(n^3). Ties are broken by
    the matrix order (the caller encodes them in the costs), so the result is deterministic.
    """
    size = len(cost)
    if size == 0:
        return []
    infinity = max(max(row) for row in cost) * size + 1
    row_potential = [0] * (size + 1)
    col_potential = [0] * (size + 1)
    row_of_column = [0] * (size + 1)  # column -> assigned row (1-based), 0 = free
    previous = [0] * (size + 1)

    for row in range(1, size + 1):
        row_of_column[0] = row
        column = 0
        cheapest = [infinity] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = row_of_column[column]
            delta = infinity
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    cost[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - col_potential[candidate]
                )
                if reduced < cheapest[candidate]:
                    cheapest[candidate] = reduced
                    previous[candidate] = column
                if cheapest[candidate] < delta:
                    delta = cheapest[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[row_of_column[candidate]] += delta
                    col_potential[candidate] -= delta
                else:
                    cheapest[candidate] -= delta
            column = next_column
            if row_of_column[column] == 0:
                break
        while column != 0:
            source = previous[column]
            row_of_column[column] = row_of_column[source]
            column = source

    assignment = [0] * size
    for column in range(1, size + 1):
        if row_of_column[column] != 0:
            assignment[row_of_column[column] - 1] = column - 1
    return assignment


def _padded_costs(predicted: Sequence[LineItem], expected: Sequence[GTLineItem]) -> list[list[int]]:
    """The square cost matrix: a real pair costs its disagreeing cells, padding costs 0."""
    size = max(len(predicted), len(expected))
    costs = []
    for row in range(size):
        line = []
        for column in range(size):
            if row < len(predicted) and column < len(expected):
                line.append(row_cost(compare_rows(expected[column], predicted[row])))
            else:
                line.append(0)
        costs.append(line)
    return costs


@dataclass(frozen=True)
class RowPair:
    """One matched pair of rows. `cells` holds only the cells the label decided."""

    predicted: int
    expected: int
    cells: dict[str, Comparison]

    @property
    def cost(self) -> int:
        return row_cost(self.cells)

    @property
    def compared_cells(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class RowMatch:
    """The pairing of a document's rows: matched pairs, plus the rows left on either side."""

    pairs: list[RowPair]
    missed: list[int]  # indices into the labeled rows
    spurious: list[int]  # indices into the predicted rows

    @property
    def matched(self) -> int:
        return len(self.pairs)


def match_rows(predicted: Sequence[LineItem], expected: Sequence[GTLineItem]) -> RowMatch:
    """Pair predicted rows with labeled rows by minimum total cell cost."""
    if not predicted or not expected:
        return RowMatch([], list(range(len(expected))), list(range(len(predicted))))
    costs = _padded_costs(predicted, expected)
    assignment = assign_min_cost(costs)
    pairs: list[RowPair] = []
    missed: list[int] = []
    spurious: list[int] = []
    for row, column in enumerate(assignment):
        if row >= len(predicted):
            if column < len(expected):
                missed.append(column)  # the padding row took a labeled row
            continue
        if column >= len(expected):
            spurious.append(row)  # this predicted row took a padding column
            continue
        cells = compare_rows(expected[column], predicted[row])
        pair = RowPair(predicted=row, expected=column, cells=cells)
        if pair.cost == pair.compared_cells and pair.compared_cells > 0:
            missed.append(column)  # nothing agrees: a missed row and a spurious one
            spurious.append(row)
        else:
            pairs.append(pair)
    return RowMatch(
        pairs=sorted(pairs, key=lambda pair: pair.expected),
        missed=sorted(missed),
        spurious=sorted(spurious),
    )
