from decimal import Decimal

import pytest

from docextract.extract.schema import LineItem
from evalkit.groundtruth import GTLineItem
from evalkit.matching import (
    RowMatch,
    assign_min_cost,
    cell_verdict,
    compare_rows,
    match_rows,
    row_cost,
)


def line(
    description: str | None,
    quantity: Decimal | None = None,
    unit_price: Decimal | None = None,
    amount: Decimal | None = None,
) -> LineItem:
    return LineItem(
        description=description,
        quantity=quantity,
        unit_price=unit_price,
        amount=amount,
        status="extracted",
        confidence=1.0,
    )


def gt_row(
    description: object = "__unlabeled__",
    quantity: object = "__unlabeled__",
    unit_price: object = "__unlabeled__",
    amount: object = "__unlabeled__",
) -> GTLineItem:
    return GTLineItem(
        description=description, quantity=quantity, unit_price=unit_price, amount=amount
    )


def item(index: int) -> tuple[GTLineItem, LineItem]:
    quantity = Decimal(index + 1)
    unit_price = Decimal("1000")
    amount = quantity * unit_price
    return (
        gt_row(f"Item {index:02d}", quantity, unit_price, amount),
        line(f"Item {index:02d}", quantity, unit_price, amount),
    )


class TestAssignMinCost:
    def test_empty_matrix_has_no_assignment(self) -> None:
        assert assign_min_cost([]) == []

    @pytest.mark.parametrize(
        ("cost", "expected"),
        [
            ([[0]], [0]),
            ([[1, 2], [2, 1]], [0, 1]),
            ([[2, 9, 3], [9, 2, 9], [3, 9, 2]], [0, 1, 2]),
        ],
    )
    def test_known_costs(self, cost: list[list[int]], expected: list[int]) -> None:
        assert assign_min_cost(cost) == expected

    def test_the_cheaper_assignment_wins(self) -> None:
        # pairing 0->1 and 1->0 costs 2; the identity costs 100
        assert assign_min_cost([[50, 1], [1, 50]]) == [1, 0]

    def test_equal_costs_resolve_the_same_way_every_time(self) -> None:
        costs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        first = assign_min_cost(costs)
        assert first == [0, 1, 2]  # pinned: the scan order decides ties, not chance
        assert all(assign_min_cost(costs) == first for _ in range(5))
        assert assign_min_cost([[0, 0], [0, 0]]) == [0, 1]


class TestCellVerdict:
    def test_an_unlabeled_cell_is_not_compared(self) -> None:
        assert cell_verdict("quantity", "__unlabeled__", Decimal("2")) is None

    def test_a_labeled_null_cell_matches_only_a_null_cell(self) -> None:
        assert cell_verdict("quantity", None, None).matched is True
        assert cell_verdict("quantity", None, Decimal("2")).matched is False

    def test_a_value_against_a_null_cell_is_a_mismatch(self) -> None:
        assert cell_verdict("amount", Decimal("5"), None).matched is False

    def test_values_go_through_the_cell_comparator(self) -> None:
        assert cell_verdict("amount", Decimal("5"), Decimal("5.00")).matched is True
        assert cell_verdict("description", "Kertas A4", "kertas a4").matched is True


class TestRowCost:
    def test_cost_counts_the_disagreeing_cells(self) -> None:
        expected_row = gt_row("A", Decimal("1"), amount=Decimal("9"))
        predicted = line("A", Decimal("2"), None, Decimal("9"))
        cells = compare_rows(expected_row, predicted)
        assert list(cells) == ["description", "quantity", "amount"]  # unit_price: unlabeled
        assert row_cost(cells) == 1


class TestMatchRows:
    def test_nothing_on_either_side(self) -> None:
        assert match_rows([], []) == RowMatch([], [], [])

    def test_no_predicted_rows_leaves_every_labeled_row_missed(self) -> None:
        result = match_rows([], [gt_row("A"), gt_row("B")])
        assert (result.pairs, result.missed, result.spurious) == ([], [0, 1], [])

    def test_no_labeled_rows_leaves_every_predicted_row_spurious(self) -> None:
        result = match_rows([line("A")], [])
        assert (result.pairs, result.missed, result.spurious) == ([], [], [0])

    def test_shuffled_rows_are_paired_correctly(self) -> None:
        pairs = [item(index) for index in range(4)]
        expected = [expected_row for expected_row, _ in pairs]
        predicted = [predicted_row for _, predicted_row in reversed(pairs)]
        result = match_rows(predicted, expected)
        assert result.missed == [] and result.spurious == []
        assert [(pair.predicted, pair.expected) for pair in result.pairs] == [
            (3, 0),
            (2, 1),
            (1, 2),
            (0, 3),
        ]

    def test_a_missing_row_is_missed_not_spurious(self) -> None:
        pairs = [item(index) for index in range(3)]
        expected = [expected_row for expected_row, _ in pairs]
        predicted = [pairs[0][1], pairs[2][1]]
        result = match_rows(predicted, expected)
        assert (len(result.pairs), result.missed, result.spurious) == (2, [1], [])

    def test_an_extra_row_is_spurious_not_missed(self) -> None:
        expected_row, predicted_row = item(0)
        result = match_rows([predicted_row, line("Extra", Decimal("1"))], [expected_row])
        assert (len(result.pairs), result.missed, result.spurious) == (1, [], [1])

    def test_a_row_that_shares_nothing_is_unwound(self) -> None:
        result = match_rows([line("Totally different")], [gt_row("Item 00")])
        assert (result.pairs, result.missed, result.spurious) == ([], [0], [0])

    def test_an_unlabeled_cell_costs_nothing(self) -> None:
        expected_row = gt_row("Item 00")  # only the description is labeled
        predicted = line("Item 00", Decimal("7"), Decimal("9"), Decimal("63"))
        result = match_rows([predicted], [expected_row])
        assert result.missed == [] and result.spurious == []
        pair = result.pairs[0]
        assert list(pair.cells) == ["description"]
        assert pair.cost == 0 and pair.compared_cells == 1

    def test_a_partial_row_still_pairs(self) -> None:
        expected_row = gt_row("Item 00", Decimal("1"), Decimal("1000"), Decimal("1000"))
        predicted = line("Item 00", Decimal("1"), Decimal("1000"), Decimal("1200"))
        result = match_rows([predicted], [expected_row])
        assert result.pairs[0].cost == 1  # the amount alone disagrees

    def test_print_rounding_inside_the_tolerance_is_not_an_error(self) -> None:
        expected_row = gt_row("Item 00", amount=Decimal("1000"))
        predicted = line("Item 00", None, None, Decimal("999"))
        assert match_rows([predicted], [expected_row]).pairs[0].cost == 0

    def test_a_forty_row_document_is_paired(self) -> None:
        pairs = [item(index) for index in range(40)]
        expected = [expected_row for expected_row, _ in pairs]
        predicted = [predicted_row for _, predicted_row in pairs]
        predicted.reverse()
        result = match_rows(predicted, expected)
        assert result.missed == [] and result.spurious == []
        assert len(result.pairs) == 40
        assert all(pair.cost == 0 for pair in result.pairs)
