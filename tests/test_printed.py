import pytest

from evalkit.printed import CURRENCY_NAMES, Parsed, currency_code, parse_amount, parse_printed_date


class TestCurrencyCode:
    @pytest.mark.parametrize(
        ("text", "code"),
        [
            ("Rp", "IDR"),
            ("Rp.", "IDR"),
            ("RP", "IDR"),
            (" rupiah ", "IDR"),
            ("IDR", "IDR"),
            ("$", "USD"),
            ("US$", "USD"),
            ("usd", "USD"),
            ("S$", "SGD"),
            ("€", "EUR"),
            ("£", "GBP"),
        ],
    )
    def test_known_markers(self, text: str, code: str) -> None:
        assert currency_code(text) == code

    @pytest.mark.parametrize(("text", "code"), [("CHF", "CHF"), ("aud", "AUD"), ("jpy.", "JPY")])
    def test_any_other_three_letters_are_taken_as_an_iso_code(self, text: str, code: str) -> None:
        assert currency_code(text) == code

    @pytest.mark.parametrize("text", [None, "", "  ", "12$", "ab", "abcd", "rupee-ish", "1ab"])
    def test_anything_else_is_not_a_currency(self, text: str | None) -> None:
        assert currency_code(text) is None

    def test_every_table_entry_maps_to_a_three_letter_code(self) -> None:
        assert all(len(code) == 3 and code.isupper() for code in CURRENCY_NAMES.values())
        assert all(key == key.lower() for key in CURRENCY_NAMES)


class TestParseAmount:
    @pytest.mark.parametrize(
        ("text", "currency", "value"),
        [
            ("Rp 1.250.000,00", None, "1250000.00"),  # the marker is its own context
            ("1.250.000,00", "IDR", "1250000.00"),
            ("1.250.000", "IDR", "1250000"),
            ("9.116,8", None, "9116.8"),
            ("1,250,000.00", "USD", "1250000.00"),
            ("1.250", "IDR", "1250"),
            ("1.250", "USD", "1.250"),
            ("2,5", None, "2.5"),
            ("(1.500)", "IDR", "-1500"),
            ("0", None, "0"),
        ],
    )
    def test_amounts_keep_their_printed_scale(
        self, text: str, currency: str | None, value: str
    ) -> None:
        parsed = parse_amount(text, currency)
        assert parsed == Parsed(value=value) and parsed.ok

    def test_what_the_context_cannot_settle_is_ambiguous_not_guessed(self) -> None:
        parsed = parse_amount("1.250", None)
        assert parsed.value is None and not parsed.ok
        assert parsed.kind == "ambiguous" and "'1.250'" in parsed.problem

    @pytest.mark.parametrize("text", ["abc", "", "1.2.3,4,5", "12,", "--5"])
    def test_garbage_is_invalid(self, text: str) -> None:
        parsed = parse_amount(text, "IDR")
        assert parsed.value is None and parsed.kind == "invalid" and parsed.problem


class TestParsePrintedDate:
    @pytest.mark.parametrize(
        ("text", "currency", "iso"),
        [
            ("5 Maret 2024", None, "2024-03-05"),
            ("2024-03-05", None, "2024-03-05"),
            ("05/03/2024", "IDR", "2024-03-05"),  # an IDR document is read day-first
            ("13/03/2024", None, "2024-03-13"),  # a day above 12 settles it
            ("5 March 2024", "USD", "2024-03-05"),
        ],
    )
    def test_dates_become_iso(self, text: str, currency: str | None, iso: str) -> None:
        assert parse_printed_date(text, currency) == Parsed(value=iso)

    def test_a_date_that_could_be_read_two_ways_is_ambiguous_without_a_convention(self) -> None:
        for currency in (None, "USD"):
            parsed = parse_printed_date("05/03/2024", currency)
            assert parsed.value is None and parsed.kind == "ambiguous", currency

    @pytest.mark.parametrize("text", ["31/02/2024", "someday", ""])
    def test_an_impossible_date_is_invalid(self, text: str) -> None:
        parsed = parse_printed_date(text, "IDR")
        assert parsed.value is None and parsed.kind == "invalid" and parsed.problem
