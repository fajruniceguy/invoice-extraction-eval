import pytest

from evalkit.comparators import CELL_KIND, FIELD_KIND
from evalkit.taxonomy import (
    ERROR_CLASSES,
    NEAR_MISS_MIN,
    FieldOutcome,
    classify_field,
    count_errors,
)

WRONG_OUTCOMES: tuple[FieldOutcome, ...] = (
    "value_wrong",
    "should_be_absent",
    "should_be_value",
    "abstained",
    "no_prediction",
)


class TestVocabulary:
    def test_every_kind_has_a_wrong_answer_class(self) -> None:
        for kind in set(FIELD_KIND.values()) | set(CELL_KIND.values()):
            assert classify_field("value_wrong", kind) in ERROR_CLASSES

    def test_the_classes_are_unique(self) -> None:
        assert len(set(ERROR_CLASSES)) == len(ERROR_CLASSES)


class TestClassifyField:
    @pytest.mark.parametrize("outcome", WRONG_OUTCOMES)
    def test_every_wrong_outcome_gets_exactly_one_class(self, outcome: FieldOutcome) -> None:
        result = classify_field(outcome, "number")
        assert result in ERROR_CLASSES

    def test_a_correct_field_has_no_class(self) -> None:
        assert classify_field("correct", "number") is None

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("number", "numeric_mismatch"),
            ("date", "date_mismatch"),
            ("text", "string_mismatch"),
            ("code", "string_mismatch"),
            ("name", "string_mismatch"),
        ],
    )
    def test_a_wrong_value_is_classified_by_kind(self, kind: str, expected: str) -> None:
        assert classify_field("value_wrong", kind) == expected

    def test_a_near_miss_name_is_told_apart_from_a_plain_mismatch(self) -> None:
        near = classify_field("value_wrong", "name", similarity=NEAR_MISS_MIN)
        assert near == "name_near_miss"
        assert classify_field("value_wrong", "name", similarity=0.49) == "string_mismatch"
        assert classify_field("value_wrong", "name") == "string_mismatch"

    def test_absence_mistakes_are_named(self) -> None:
        assert classify_field("should_be_absent", "number") == "hallucinated_value"
        assert classify_field("should_be_value", "number") == "missing_value"
        assert classify_field("no_prediction", "number") == "no_prediction"

    def test_an_abstention_says_what_it_abstained_on(self) -> None:
        assert classify_field("abstained", "number") == "abstained_on_value"
        assert classify_field("abstained", "number", expected_absent=True) == "abstained_on_absent"


class TestCountErrors:
    def test_counts_sum_to_the_wrong_total_and_skip_the_empty_classes(self) -> None:
        counts = count_errors(["numeric_mismatch", None, "numeric_mismatch", "row_missed", None])
        assert counts == {"numeric_mismatch": 2, "row_missed": 1}
        assert sum(counts.values()) == 3

    def test_the_order_is_the_vocabulary_order(self) -> None:
        counts = count_errors(["row_spurious", "numeric_mismatch", "numeric_mismatch"])
        assert list(counts) == ["numeric_mismatch", "row_spurious"]

    def test_nothing_wrong_counts_nothing(self) -> None:
        assert count_errors([None, None]) == {}
