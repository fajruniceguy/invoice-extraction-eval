from datetime import date
from decimal import Decimal

import pytest

from docextract.normalize import convention_for, currency_marker, parse_date, parse_number

D = Decimal


def ok(text: str, **kwargs: str | None) -> str:
    """The normalized string of a number that must parse unambiguously."""
    result = parse_number(text, **kwargs)  # type: ignore[arg-type]
    assert result.status == "ok", f"{text!r}: {result}"
    assert result.text is not None
    return result.text


class TestIndonesianNumbers:
    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            ("1.250.000,00", "1250000.00"),
            ("9.116,8", "9116.8"),
            ("1.250.000", "1250000"),
            ("12.345.678.901", "12345678901"),
            ("123.456.789.012,34", "123456789012.34"),
            ("0,352", "0.352"),
            ("9,5", "9.5"),
            ("12,50", "12.50"),
            ("1234567,89", "1234567.89"),
            ("25.900", "25900") if False else ("25.900,00", "25900.00"),
            ("0,05", "0.05"),
            ("999.999.999", "999999999"),
        ],
    )
    def test_indonesian_format_with_dot_thousands_and_comma_decimals(
        self, printed: str, expected: str
    ) -> None:
        assert ok(printed) == expected

    def test_the_printed_scale_is_kept(self) -> None:
        with_decimals = parse_number("1.250.000,00")
        without = parse_number("1.250.000")
        assert with_decimals.value == D("1250000.00")
        assert with_decimals.value is not None
        assert with_decimals.value.as_tuple().exponent == -2
        assert without.value == D("1250000")
        assert without.value is not None
        assert without.value.as_tuple().exponent == 0

    def test_format_is_reported(self) -> None:
        assert parse_number("1.250.000,00").format == "indonesian"
        assert parse_number("1,250,000.00").format == "international"
        assert parse_number("1250000").format == "plain"


class TestInternationalNumbers:
    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            ("1,250,000.00", "1250000.00"),
            ("1,250,000", "1250000"),
            ("12.50", "12.50"),
            ("1,250.5", "1250.5"),
            ("1234.5", "1234.5"),
            ("0.352", "0.352"),
            ("1250000", "1250000"),
            ("17.28", "17.28"),
            ("1250000.00", "1250000.00"),
            ("0.0005", "0.0005"),
            ("12345.678", "12345.678"),  # 5 digits before: cannot be a thousands group
            ("1234,567", "1234.567"),
        ],
    )
    def test_international_and_plain_formats(self, printed: str, expected: str) -> None:
        assert ok(printed) == expected


class TestAmbiguousNumbers:
    """'1.250' / '1,250': thousands or decimal? Never guessed: context or ambiguous."""

    @pytest.mark.parametrize("printed", ["1.250", "1,250", "12.345", "999,999", "10.000", "1.000"])
    def test_no_currency_context_means_ambiguous_with_both_candidates(self, printed: str) -> None:
        result = parse_number(printed)
        assert result.status == "ambiguous"
        assert result.value is None
        assert result.text is None
        digits = printed.replace(".", "").replace(",", "")
        before, after = printed[:-4], printed[-3:]
        assert D(digits) in result.candidates  # the thousands reading
        assert D(f"{before}.{after}") in result.candidates  # the decimal reading
        assert len(result.candidates) == 2
        assert "no currency context" in result.reason

    def test_a_dot_is_thousands_in_an_idr_context(self) -> None:
        result = parse_number("1.250", currency="IDR")
        assert result.status == "ok"
        assert result.text == "1250"
        assert result.resolved_by == "currency_context"
        assert result.format == "indonesian"

    def test_a_dot_is_a_decimal_in_a_usd_context(self) -> None:
        result = parse_number("1.250", currency="USD")
        assert result.status == "ok"
        assert result.text == "1.250"
        assert result.value == D("1.250")
        assert result.resolved_by == "currency_context"

    def test_a_comma_is_thousands_in_a_usd_context(self) -> None:
        result = parse_number("1,250", currency="USD")
        assert result.status == "ok"
        assert result.text == "1250"
        assert result.resolved_by == "currency_context"

    def test_a_comma_stays_ambiguous_in_an_idr_context(self) -> None:
        # an Indonesian decimal comma (1.250) or an international thousands comma (1250)
        result = parse_number("1,250", currency="IDR")
        assert result.status == "ambiguous"
        assert set(result.candidates) == {D("1250"), D("1.250")}
        assert "Indonesian-style" in result.reason

    @pytest.mark.parametrize("currency", ["EUR", "eur", " IDR "])
    def test_eur_behaves_like_idr_and_case_does_not_matter(self, currency: str) -> None:
        assert ok("1.250", currency=currency) == "1250"

    def test_an_unknown_currency_gives_no_context(self) -> None:
        assert parse_number("1.250", currency="XYZ").status == "ambiguous"
        assert parse_number("1.250", currency="").status == "ambiguous"
        assert parse_number("1.250", currency=None).status == "ambiguous"

    def test_a_printed_marker_beats_the_callers_currency(self) -> None:
        # the text says Rp, the caller says USD: the printed evidence wins
        assert ok("Rp 1.250", currency="USD") == "1250"
        assert ok("$1.250", currency="IDR") == "1.250"

    @pytest.mark.parametrize(
        ("printed", "currency"),
        [("Rp 1.250", None), ("Rp. 1.250", None), ("IDR 1.250", None), ("1.250 Rp", None)],
    )
    def test_a_marker_in_the_text_supplies_the_context(
        self, printed: str, currency: str | None
    ) -> None:
        assert ok(printed, currency=currency) == "1250"

    def test_negative_ambiguous_numbers_keep_their_sign_in_the_candidates(self) -> None:
        result = parse_number("-1.250")
        assert result.status == "ambiguous"
        assert set(result.candidates) == {D("-1250"), D("-1.250")}

    @pytest.mark.parametrize(
        "printed",
        [
            "1.25",  # two digits after: a decimal
            "1,25",
            "0.250",  # integer part 0: only a decimal
            "0,352",
            ".250",  # a thousands group cannot lead
            "12345.678",  # more than 3 digits before: not a thousands group
            "1234,567",
            "1.250.000",  # repeated: only thousands
            "1.250,000",  # both kinds: the comma is the decimal mark
        ],
    )
    def test_these_look_similar_but_are_not_ambiguous(self, printed: str) -> None:
        assert parse_number(printed).status == "ok"

    def test_unambiguous_numbers_say_so(self) -> None:
        assert parse_number("12,50").resolved_by == "unambiguous"
        assert parse_number("1.250.000").resolved_by == "unambiguous"


class TestSignsMarkersAndSpacing:
    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            ("-9.116,8", "-9116.8"),
            ("(1.250.000,00)", "-1250000.00"),
            ("1.250.000,00-", "-1250000.00"),
            ("−9,5", "-9.5"),  # a typographic minus
            ("+12,5", "12.5"),
            ("-0", "0"),
        ],
    )
    def test_negative_forms(self, printed: str, expected: str) -> None:
        assert ok(printed) == expected

    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            ("Rp 1.250.000,00", "1250000.00"),
            ("Rp. 15.000,00", "15000.00"),
            ("Rp15.000,00", "15000.00"),
            ("IDR 2.500.000", "2500000"),
            ("$1,250.00", "1250.00"),
            ("USD 1,250,000", "1250000"),
            ("US$ 12.50", "12.50"),
            ("S$ 3.20", "3.20"),
            ("15.000,00 Rp", "15000.00"),
            ("-Rp 9.116,8", "-9116.8"),
            ("Rp -9.116,8", "-9116.8"),
            ("€ 1.234,56", "1234.56"),
        ],
    )
    def test_currency_markers_are_stripped(self, printed: str, expected: str) -> None:
        assert ok(printed) == expected

    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            ("1 250 000", "1250000"),
            ("1 250 000", "1250000"),
            ("1 250,50", "1250.50"),
            ("  17,28  ", "17.28"),
        ],
    )
    def test_spaces_that_group_thousands(self, printed: str, expected: str) -> None:
        assert ok(printed) == expected

    def test_leading_zeros_are_dropped_from_the_integer_part(self) -> None:
        assert ok("007") == "7"
        assert ok("000") == "0"

    def test_a_leading_separator_is_a_decimal(self) -> None:
        assert ok(",5") == "0.5"
        assert ok(".25") == "0.25"


class TestInvalidNumbers:
    @pytest.mark.parametrize(
        "printed",
        [
            "",
            "   ",
            "abc",
            "Rp",
            "-",
            ".",
            ",",
            "1.2.3",
            "1,2,3",
            "1.250.00",  # last group is not 3 digits
            "12.34.567",
            "1,250.000,5",  # decimal mark twice
            "1..2",
            "1 2",  # a space that does not group thousands
            "12 34",
            "1.",  # dangling separator
            "1.250.",
            "--5",
            "5%",
            "1e5",
            "١٢٣",  # non-ASCII digits are not silently accepted
            "012.345",  # leading zero before a separator
            "1,25.5,3",
            "0.250.000",  # a grouped integer cannot start with 0
        ],
    )
    def test_not_a_number(self, printed: str) -> None:
        result = parse_number(printed)
        assert result.status == "invalid"
        assert result.value is None
        assert result.text is None
        assert result.reason  # always says why

    def test_the_reason_quotes_the_input(self) -> None:
        assert "'1.2.3'" in parse_number("1.2.3").reason

    def test_ok_property(self) -> None:
        assert parse_number("1250000").ok
        assert not parse_number("1.250").ok
        assert not parse_number("x").ok


class TestConvention:
    @pytest.mark.parametrize(
        ("code", "expected"), [("IDR", "id"), ("eur", "id"), ("USD", "intl"), ("sgd", "intl")]
    )
    def test_known_currencies(self, code: str, expected: str) -> None:
        assert convention_for(code) == expected

    @pytest.mark.parametrize("code", [None, "", "XYZ", "  "])
    def test_unknown_currencies(self, code: str | None) -> None:
        assert convention_for(code) is None


# ------------------------------------------------------------------ dates


def iso(text: str, **kwargs: bool | None) -> str:
    result = parse_date(text, **kwargs)  # type: ignore[arg-type]
    assert result.status == "ok", f"{text!r}: {result}"
    assert result.iso is not None
    return result.iso


class TestDates:
    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            ("2024-03-05", "2024-03-05"),
            ("2024/03/05", "2024-03-05"),
            ("2024.3.5", "2024-03-05"),
            ("20240305", "2024-03-05"),
            ("2024-03-05T14:30:00Z", "2024-03-05"),
            ("2024-03-05 14:30", "2024-03-05"),
        ],
    )
    def test_year_first_formats(self, printed: str, expected: str) -> None:
        assert iso(printed) == expected

    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            ("5 Maret 2024", "2024-03-05"),
            ("05 MARET 2024", "2024-03-05"),
            ("5 Mar 2024", "2024-03-05"),
            ("Mar 5, 2024", "2024-03-05"),
            ("March 5th, 2024", "2024-03-05"),
            ("1st May 2024", "2024-05-01"),
            ("05-Mar-24", "2024-03-05"),
            ("5-Mar-2024", "2024-03-05"),
            ("Rabu, 5 Maret 2024", "2024-03-05"),
            ("Rabu, 5 Maret 2024 14:30", "2024-03-05"),
            ("Wednesday, March 5, 2024 2:30 PM", "2024-03-05"),
            ("5 Maret 2024 14:30 WIB", "2024-03-05"),
            ("17 Agu 2024", "2024-08-17"),
            ("17 Agt 2024", "2024-08-17"),
            ("17 Agustus 2024", "2024-08-17"),
            ("1 Okt 2024", "2024-10-01"),
            ("25 Des 2024", "2024-12-25"),
            ("25 December 2024", "2024-12-25"),
            ("Mei 5, 2024", "2024-05-05"),
            ("9 Feb 2024", "2024-02-09"),
            ("3 Juni 2024", "2024-06-03"),
            ("3 Juli 2024", "2024-07-03"),
            ("Sept 9, 2024", "2024-09-09"),
            ("11 Nopember 2024", "2024-11-11"),
        ],
    )
    def test_named_months_in_indonesian_and_english(self, printed: str, expected: str) -> None:
        assert iso(printed) == expected

    def test_named_months_ignore_the_day_first_hint(self) -> None:
        assert iso("5 Mar 2024", day_first=False) == "2024-03-05"
        assert iso("Mar 5, 2024", day_first=True) == "2024-03-05"

    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            ("13/03/2024", "2024-03-13"),  # 13 cannot be a month: day first
            ("03/13/2024", "2024-03-13"),  # the second number cannot be a month: month first
            ("31-12-2024", "2024-12-31"),
            ("12/31/2024", "2024-12-31"),
            ("05/05/2024", "2024-05-05"),  # the same either way round
            ("05.03.2024", None),  # genuinely ambiguous, covered below
        ],
    )
    def test_numeric_dates_that_are_not_ambiguous(self, printed: str, expected: str | None) -> None:
        if expected is None:
            assert parse_date(printed).status == "ambiguous"
        else:
            assert iso(printed) == expected

    def test_both_readings_are_candidates_when_day_first_is_unknown(self) -> None:
        result = parse_date("05/03/2024")
        assert result.status == "ambiguous"
        assert result.iso is None
        assert set(result.candidates) == {date(2024, 3, 5), date(2024, 5, 3)}
        assert "day_first" in result.reason

    def test_day_first_resolves_it(self) -> None:
        resolved = parse_date("05/03/2024", day_first=True)
        assert (resolved.status, resolved.iso, resolved.resolved_by) == (
            "ok",
            "2024-03-05",
            "day_first",
        )
        flipped = parse_date("05/03/2024", day_first=False)
        assert (flipped.status, flipped.iso, flipped.resolved_by) == (
            "ok",
            "2024-05-03",
            "day_first",
        )

    def test_unambiguous_dates_do_not_claim_the_hint_resolved_them(self) -> None:
        assert parse_date("13/03/2024", day_first=False).resolved_by == "unambiguous"
        assert parse_date("13/03/2024", day_first=False).iso == "2024-03-13"
        assert parse_date("2024-03-05").resolved_by == "unambiguous"

    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            ("5-3-24", "2024-03-05"),
            ("05/03/24", "2024-03-05"),
            ("01/01/69", "2069-01-01"),
            ("01/01/70", "1970-01-01"),
            ("99-12-31", "1999-12-31"),  # yy-mm-dd: 99 cannot be a day
            ("5 Mar 24", "2024-03-05"),
        ],
    )
    def test_two_digit_years(self, printed: str, expected: str) -> None:
        assert iso(printed, day_first=True) == expected

    def test_a_first_number_up_to_31_is_never_read_as_a_year(self) -> None:
        # documented rule: "24-03-05" is 24 March 2005 (day first), not 5 March 2024
        assert iso("24-03-05", day_first=True) == "2005-03-24"

    @pytest.mark.parametrize(
        "printed",
        [
            "",
            "   ",
            "yesterday",
            "31/02/2024",  # not a real date
            "32 Maret 2024",
            "30 Februari 2024",
            "Mar 2024",  # no day
            "5 Mar",  # no year
            "13/13/2024",  # neither can be a month
            "2024-13-01",
            "2024-02-30",
            "5 Marsh 2024",
            "1/2",
            "Rabu",  # a weekday alone
            "12345",
            "5 Mar Apr 2024",  # two month names
            "1850-01-01",  # outside the plausible range
            "01/01/2200",
        ],
    )
    def test_invalid_dates(self, printed: str) -> None:
        result = parse_date(printed, day_first=True)
        assert result.status == "invalid"
        assert result.value is None
        assert result.iso is None
        assert result.reason

    def test_leap_days(self) -> None:
        assert iso("29 Feb 2024") == "2024-02-29"
        assert parse_date("29 Feb 2023").status == "invalid"

    def test_ok_property_and_value(self) -> None:
        result = parse_date("5 Maret 2024")
        assert result.ok
        assert result.value == date(2024, 3, 5)
        assert not parse_date("05/03/2024").ok


class TestCurrencyMarker:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Rp 1.250.000", "IDR"),
            ("rp.12.000", "IDR"),
            ("US$ 12.00", "USD"),
            ("$12.00", "USD"),
            ("12.00 SGD", "SGD"),
            ("EUR 9,90", "EUR"),
            ("(Rp 12.000)", "IDR"),
            ("1.250.000,00", None),
            ("12 pcs", None),
            ("", None),
        ],
    )
    def test_the_marker_a_currency_names(self, text: str, expected: str | None) -> None:
        assert currency_marker(text) == expected
