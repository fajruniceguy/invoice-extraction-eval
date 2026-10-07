from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher

import pytest

from docextract.extract.schema import CELL_NAMES, SCALAR_FIELD_NAMES
from evalkit.comparators import (
    CELL_KIND,
    FIELD_KIND,
    FUZZY_NAME_MIN,
    ComparatorError,
    compare_cell,
    compare_code,
    compare_date,
    compare_field,
    compare_name,
    compare_number,
    compare_text,
    digits_only,
    normalize_text,
)


def test_registries_cover_every_field_and_cell() -> None:
    assert set(FIELD_KIND) == set(SCALAR_FIELD_NAMES)
    assert set(CELL_KIND) == set(CELL_NAMES)


class TestNormalizeText:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("PT. Acme", "pt acme"),
            ("  ACME\tCorp.  ", "acme corp"),
            ("CV Maju-Jaya, Tbk", "cv maju jaya tbk"),
            ("", ""),
            ("---", ""),
        ],
    )
    def test_case_punctuation_and_spacing_are_taken_away(self, text: str, expected: str) -> None:
        assert normalize_text(text) == expected

    def test_digits_only_keeps_digits(self) -> None:
        assert digits_only("01.234.567.8-901.000") == "012345678901000"


class TestCompareName:
    def test_formatting_variants_match(self) -> None:
        result = compare_name("PT. Acme", "pt acme")
        assert (result.matched, result.how) == (True, "normalized")

    def test_identical_values_are_exact(self) -> None:
        assert compare_name("Acme Corp", "Acme Corp").how == "exact"

    def test_a_close_typo_matches_fuzzily(self) -> None:
        pair = ("acme corporation", "acme corporati")
        assert SequenceMatcher(None, *pair).ratio() >= FUZZY_NAME_MIN
        result = compare_name(*pair)
        assert (result.matched, result.how) == (True, "fuzzy")
        assert result.similarity is not None and result.similarity >= FUZZY_NAME_MIN

    def test_a_distant_string_does_not_match_but_reports_its_similarity(self) -> None:
        pair = ("acme corporation", "acme corporat")
        assert SequenceMatcher(None, *pair).ratio() < FUZZY_NAME_MIN
        result = compare_name(*pair)
        assert result.matched is False
        assert result.how is None
        assert result.similarity is not None and result.similarity < FUZZY_NAME_MIN

    def test_a_completely_different_name_does_not_match(self) -> None:
        assert compare_name("Acme", "Borneo Jaya").matched is False


class TestCompareText:
    @pytest.mark.parametrize(
        ("expected", "predicted", "matched"),
        [
            ("Jalan Sudirman 1", "Jalan Sudirman 1", True),
            ("Jalan Sudirman 1", "jalan  sudirman   1", True),
            ("Jalan Sudirman 1", "Jalan Sudirman 2", False),
            ("Transfer", "Tunai", False),
        ],
    )
    def test_normalized_exact(self, expected: str, predicted: str, matched: bool) -> None:
        assert compare_text(expected, predicted).matched is matched


class TestCompareCode:
    def test_tax_ids_match_on_digits_only(self) -> None:
        result = compare_code("01.234.567.8-901.000", "012345678901000")
        assert (result.matched, result.how) == (True, "normalized")

    def test_currency_case_does_not_matter(self) -> None:
        assert compare_code("idr", "IDR").matched is True

    def test_a_different_number_does_not_match(self) -> None:
        assert compare_code("012345678901000", "012345678901001").matched is False

    def test_digits_only_equality_needs_digits_on_both_sides(self) -> None:
        assert compare_code("A-123", "123").matched is False
        assert compare_code("...", "123").matched is False


class TestCompareNumber:
    def test_scale_difference_is_the_same_value(self) -> None:
        assert compare_number(Decimal("1250000"), Decimal("1250000.00")).matched is True

    def test_a_value_from_a_label_string_is_accepted(self) -> None:
        assert compare_number("1250000.00", Decimal("1250000")).matched is True

    def test_a_cent_of_rounding_matches(self) -> None:
        result = compare_number(Decimal("123874"), Decimal("123873.99"))
        assert (result.matched, result.how) == (True, "numeric")

    def test_a_real_difference_does_not_match(self) -> None:
        assert compare_number(Decimal("1000"), Decimal("1006")).matched is False

    def test_the_tolerance_is_the_larger_of_the_absolute_floor_and_the_relative_part(self) -> None:
        assert compare_number(Decimal("1000"), Decimal("1004.9")).matched is True
        assert compare_number(Decimal("0.001"), Decimal("0.010")).matched is True  # floor
        assert compare_number(Decimal("0.001"), Decimal("0.0111")).matched is False

    @pytest.mark.parametrize("bad", ["", "abc", "1.2.3", "NaN", "Infinity"])
    def test_a_corrupt_label_is_a_loud_error(self, bad: str) -> None:
        with pytest.raises(ComparatorError):
            compare_number(bad, Decimal("1"))

    def test_a_corrupt_prediction_is_a_loud_error(self) -> None:
        with pytest.raises(ComparatorError):
            compare_number(Decimal("1"), object())


class TestCompareDate:
    def test_a_printed_indonesian_date_matches_its_iso(self) -> None:
        result = compare_date(date(2024, 3, 5), "5 Maret 2024")
        assert (result.matched, result.how) == (True, "date")

    def test_a_different_day_does_not_match(self) -> None:
        assert compare_date(date(2024, 3, 5), "6 Maret 2024").matched is False

    def test_an_ambiguous_date_is_never_guessed(self) -> None:
        assert compare_date(date(2024, 3, 5), "05/03/2024").matched is False

    def test_an_unreadable_date_does_not_match(self) -> None:
        assert compare_date(date(2024, 3, 5), "next tuesday").matched is False
        assert compare_date(3, date(2024, 3, 5)).matched is False


class TestDispatch:
    def test_fields_use_the_kind_they_name(self) -> None:
        assert compare_field("vendor_name", "PT. Acme", "pt acme").matched is True
        assert compare_field("currency", "idr", "IDR").matched is True
        assert compare_field("total", Decimal("1250000"), Decimal("1250000.00")).matched is True
        assert compare_field("document_date", date(2024, 3, 5), "5 Maret 2024").matched is True

    def test_cells_use_the_kind_they_name(self) -> None:
        assert compare_cell("description", "Kertas A4", "kertas a4").matched is True
        assert compare_cell("amount", Decimal("450000"), Decimal("450000.00")).matched is True
