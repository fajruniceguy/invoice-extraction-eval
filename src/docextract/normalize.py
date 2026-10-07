"""Deterministic number and date normalization, shared by the extraction pipeline, the
pre-labeling script and the labeling app.

The contract is "never guess": a printed value is either converted by an unambiguous rule,
resolved by an explicit piece of context (the currency, a day-first hint), or reported as
`ambiguous` with every candidate so a caller can send it to a human or to a repair call.

Numbers
  - Both '.' and ',' present: the LAST one is the decimal separator ("1.250.000,00",
    "1,250,000.00", "9.116,8"); the other must group digits in threes.
  - One separator kind repeated: thousands ("1.250.000", "1,250,000").
  - One separator, once, not followed by exactly 3 digits: decimal ("9,5", "12.50"); an integer
    part of "0" or of more than 3 digits can only be decimal ("0,352", "12345.678").
  - One separator, once, followed by exactly 3 digits after 1-3 digits ("1.250", "1,250"):
    AMBIGUOUS unless the currency decides. Indonesian-style currencies (IDR, EUR, ...) use '.'
    for thousands and ',' for decimals, so "1.250" is 1250 but "1,250" stays ambiguous (an
    Indonesian decimal or an international thousands separator); international-style
    currencies (USD, SGD, ...) are the mirror image. A currency marker inside the text ("Rp",
    "$") beats the `currency` argument: it is printed evidence.
  - The result keeps the printed scale as a decimal string ("1.250.000,00" -> "1250000.00").

Dates
  - ISO (also with '/' or '.'), D/M/Y and M/D/Y with '/', '-' or '.', two- or four-digit years
    (two-digit: 00-69 -> 20xx, 70-99 -> 19xx; a first number above 31 is read as the year, a
    first number up to 31 is always a day or month, never a year), Indonesian and English
    month names and
    abbreviations, weekday names and times are ignored, "20240305".
  - "05/03/2024" is ambiguous unless a day is above 12 or `day_first` says which it is.
"""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

Status = Literal["ok", "ambiguous", "invalid"]
Convention = Literal["id", "intl"]
NumberFormat = Literal["indonesian", "international", "plain"]

# "id": '.' groups thousands and ',' is the decimal mark; "intl": the reverse.
_ID_STYLE = frozenset({"IDR", "EUR", "BRL", "VND", "TRY"})
_INTL_STYLE = frozenset(
    {"USD", "SGD", "AUD", "GBP", "MYR", "THB", "JPY", "CAD", "HKD", "CNY", "NZD", "INR"}
)

# longest first: "us$" before "s$" before "$", "rp." before "rp"
_MARKERS: tuple[tuple[str, str], ...] = (
    ("us$", "USD"),
    ("rp.", "IDR"),
    ("idr", "IDR"),
    ("usd", "USD"),
    ("sgd", "SGD"),
    ("eur", "EUR"),
    ("gbp", "GBP"),
    ("s$", "SGD"),
    ("rp", "IDR"),
    ("$", "USD"),
    ("€", "EUR"),
    ("£", "GBP"),
)

_GROUP_SPACE = re.compile(r"(?<=\d)[   '’](?=\d{3}(?!\d))")


def convention_for(currency: str | None) -> Convention | None:
    code = (currency or "").strip().upper()
    if code in _ID_STYLE:
        return "id"
    if code in _INTL_STYLE:
        return "intl"
    return None


# ---------------------------------------------------------------- numbers


@dataclass(frozen=True)
class NumberResult:
    status: Status
    value: Decimal | None = None
    text: str | None = None  # the value as a plain decimal string, printed scale kept
    candidates: tuple[Decimal, ...] = ()  # every reading, when status is "ambiguous"
    reason: str = ""
    resolved_by: Literal["unambiguous", "currency_context"] | None = None
    format: NumberFormat | None = None  # which convention the printed digits followed

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _invalid(text: str, reason: str) -> NumberResult:
    return NumberResult(status="invalid", reason=f"{text!r}: {reason}")


def _strip_decorations(text: str) -> tuple[str, bool, str | None]:
    """Remove sign forms and currency markers; return (core, negative, marker currency)."""
    remaining = text.strip().replace("−", "-")
    negative = False
    marker_currency: str | None = None
    for _ in range(4):
        remaining = remaining.strip()
        changed = False
        if len(remaining) > 1 and remaining.startswith("(") and remaining.endswith(")"):
            remaining, negative, changed = remaining[1:-1], True, True
        # at most ONE sign in total: a second one is left in place and fails validation
        if not negative and remaining.startswith("-"):
            remaining, negative, changed = remaining[1:], True, True
        elif not negative and remaining.startswith("+"):
            remaining, changed = remaining[1:], True
        if not negative and remaining.endswith("-"):
            remaining, negative, changed = remaining[:-1], True, True
        lowered = remaining.lower()
        for marker, code in _MARKERS:
            if lowered.startswith(marker):
                remaining, marker_currency, changed = (
                    remaining[len(marker) :],
                    marker_currency or code,
                    True,
                )
                break
            if lowered.endswith(marker):
                remaining, marker_currency, changed = (
                    remaining[: -len(marker)],
                    marker_currency or code,
                    True,
                )
                break
        if not changed:
            break
    return remaining.strip(), negative, marker_currency


def currency_marker(text: str) -> str | None:
    """The currency a printed marker in `text` names ("Rp 12.000" -> "IDR"), or None.

    The same markers `parse_number` strips, so a marker found here is printed evidence of
    which currency the document's money is in."""
    _, _, marker_currency = _strip_decorations(text)
    return marker_currency


def _grouped_integer(part: str, separator: str) -> str | None:
    """Digits of `part` if it is valid thousands grouping (1-3 digits, then groups of 3)."""
    groups = part.split(separator)
    if not groups[0] or not groups[0].isdigit() or len(groups[0]) > 3:
        return None
    if not all(len(group) == 3 and group.isdigit() for group in groups[1:]):
        return None
    if groups[0].startswith("0"):  # "0.250.000" or "012.000": not a grouped integer
        return None
    return "".join(groups)


def _result(
    negative: bool,
    integer: str,
    fraction: str | None,
    *,
    fmt: NumberFormat,
    resolved_by: Literal["unambiguous", "currency_context"],
) -> NumberResult:
    integer = integer.lstrip("0") or "0"
    text = integer if fraction is None else f"{integer}.{fraction}"
    value = Decimal(text)
    if negative and value != 0:
        text, value = f"-{text}", -value
    return NumberResult(status="ok", value=value, text=text, resolved_by=resolved_by, format=fmt)


def parse_number(text: str, *, currency: str | None = None) -> NumberResult:
    """Parse a printed number. See the module docstring for the rules."""
    original = text
    core, negative, marker_currency = _strip_decorations(text)
    # printed evidence (a "Rp" or "$" in the text) beats the caller's context
    convention = convention_for(marker_currency) or convention_for(currency)
    core = _GROUP_SPACE.sub("", core)
    if not core:
        return _invalid(original, "no digits")
    if not re.fullmatch(r"[0-9.,]+", core):
        return _invalid(original, "unexpected characters")
    if not any(character.isdigit() for character in core):
        return _invalid(original, "no digits")
    if core[-1] in ".,":
        return _invalid(original, "dangling separator")

    dots, commas = core.count("."), core.count(",")
    if dots == 0 and commas == 0:
        return _result(negative, core, None, fmt="plain", resolved_by="unambiguous")

    if dots and commas:
        decimal_mark = "." if core.rindex(".") > core.rindex(",") else ","
        grouping = "," if decimal_mark == "." else "."
        if core.count(decimal_mark) != 1:
            return _invalid(original, f"more than one {decimal_mark!r} decimal mark")
        integer_part, fraction = core.split(decimal_mark)
        if grouping not in integer_part or decimal_mark in integer_part:
            return _invalid(original, "separators in an impossible order")
        digits = _grouped_integer(integer_part, grouping)
        if digits is None or not fraction.isdigit():
            return _invalid(original, "thousands groups must have exactly 3 digits")
        fmt: NumberFormat = "indonesian" if decimal_mark == "," else "international"
        return _result(negative, digits, fraction, fmt=fmt, resolved_by="unambiguous")

    separator = "." if dots else ","
    count = dots + commas
    thousands_format: NumberFormat = "indonesian" if separator == "." else "international"
    decimal_format: NumberFormat = "indonesian" if separator == "," else "international"

    if count >= 2:  # a decimal mark can appear only once, so this is thousands grouping
        digits = _grouped_integer(core, separator)
        if digits is None:
            return _invalid(original, f"repeated {separator!r} is not valid thousands grouping")
        return _result(negative, digits, None, fmt=thousands_format, resolved_by="unambiguous")

    before, after = core.split(separator)
    if before and not before.isdigit() or not after.isdigit():
        return _invalid(original, "malformed number")
    if before.startswith("0") and before != "0":
        return _invalid(original, "leading zero")
    if len(after) != 3 or before in ("", "0") or len(before) > 3:
        # not exactly 3 digits after it, or an integer part that cannot be a thousands group
        return _result(
            negative, before or "0", after, fmt=decimal_format, resolved_by="unambiguous"
        )

    # exactly 3 digits after one separator, after 1-3 digits: "1.250" / "1,250"
    thousands_reading = before + after
    if convention == "id" and separator == ".":
        return _result(
            negative, thousands_reading, None, fmt="indonesian", resolved_by="currency_context"
        )
    if convention == "intl" and separator == ",":
        return _result(
            negative, thousands_reading, None, fmt="international", resolved_by="currency_context"
        )
    if convention == "intl" and separator == ".":
        return _result(negative, before, after, fmt="international", resolved_by="currency_context")
    sign = -1 if negative else 1
    candidates = (Decimal(thousands_reading) * sign, Decimal(f"{before}.{after}") * sign)
    if convention is None:
        why = "no currency context"
    else:  # Indonesian-style currency with a ',' : decimal comma or international thousands
        why = "an Indonesian-style currency can print either a decimal comma or a thousands comma"
    return NumberResult(
        status="ambiguous",
        candidates=candidates,
        reason=(
            f"{original!r} could be {thousands_reading} (thousands) or {before}.{after} "
            f"(decimal): {why}"
        ),
    )


# ---------------------------------------------------------------- dates


@dataclass(frozen=True)
class DateResult:
    status: Status
    value: date | None = None
    iso: str | None = None
    candidates: tuple[date, ...] = ()
    reason: str = ""
    resolved_by: Literal["unambiguous", "day_first"] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


_MONTHS: dict[str, int] = {}
for _number, _names in enumerate(
    [
        ("januari", "january", "jan"),
        ("februari", "february", "feb", "pebruari"),
        ("maret", "march", "mar"),
        ("april", "apr"),
        ("mei", "may"),
        ("juni", "june", "jun"),
        ("juli", "july", "jul"),
        ("agustus", "august", "agu", "agt", "agust", "ags", "aug"),
        ("september", "sep", "sept"),
        ("oktober", "october", "okt", "oct"),
        ("november", "nov", "nopember"),
        ("desember", "december", "des", "dec"),
    ],
    start=1,
):
    for _name in _names:
        _MONTHS[_name] = _number

_WEEKDAYS = frozenset(
    {
        # Indonesian, full and abbreviated
        "senin", "selasa", "rabu", "kamis", "jumat", "jum'at", "sabtu", "minggu", "ahad",
        "sen", "sel", "rab", "kam", "jum", "sab", "min",
        # English, full and abbreviated
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    }
)  # fmt: skip

_TIME = re.compile(
    r"(?:[tT]|\b)\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*(?:[AaPp][Mm])?\s*"
    r"(?:WIB|WITA|WIT|UTC|GMT|Z|[+-]\d{2}:?\d{2})?",
)


def _invalid_date(text: str, reason: str) -> DateResult:
    return DateResult(status="invalid", reason=f"{text!r}: {reason}")


def _make_date(text: str, year: int, month: int, day: int) -> date | DateResult:
    if not 1900 <= year <= 2100:
        return _invalid_date(text, f"year {year} is out of range")
    try:
        return date(year, month, day)
    except ValueError:
        return _invalid_date(text, f"{year:04d}-{month:02d}-{day:02d} is not a real date")


def _two_digit_year(value: int) -> int:
    return 2000 + value if value < 70 else 1900 + value


def _year(token: str) -> int:
    return int(token) if len(token) == 4 else _two_digit_year(int(token))


def _ok(value: date, by: Literal["unambiguous", "day_first"] = "unambiguous") -> DateResult:
    return DateResult(status="ok", value=value, iso=value.isoformat(), resolved_by=by)


def parse_date(text: str, *, day_first: bool | None = None) -> DateResult:
    """Parse a printed date into ISO. `day_first` resolves D/M vs M/D only when it is needed."""
    original = text
    cleaned = _TIME.sub(" ", text.strip())
    tokens = re.findall(r"[A-Za-z']+|\d+|[/.\-]", cleaned)
    # weekday names carry no information
    words = [token for token in tokens if token.isalpha() or "'" in token]
    kept = [t for t in tokens if t.lower() not in _WEEKDAYS]
    if not kept:
        return _invalid_date(original, "no date found")

    month_words = [w for w in words if w.lower().rstrip(".") in _MONTHS]
    numbers = [t for t in kept if t.isdigit()]
    if month_words:
        if len(month_words) != 1:
            return _invalid_date(original, "more than one month name")
        month = _MONTHS[month_words[0].lower().rstrip(".")]
        # ordinal suffixes ("5th", "1st") arrive as a stray word: ignore it
        if len(numbers) != 2:
            return _invalid_date(original, "expected a day and a year next to the month name")
        four_digit = [n for n in numbers if len(n) == 4]
        if four_digit:
            year_token = four_digit[0]
            day_token = next(n for n in numbers if n is not year_token)
        else:  # "5 Mar 24" / "Mar 5, 24": the year is the later number
            day_token, year_token = numbers
        made = _make_date(original, _year(year_token), month, int(day_token))
        return _ok(made) if isinstance(made, date) else made

    if any(token.isalpha() for token in kept):
        return _invalid_date(original, "unrecognized words in the date")

    joined = "".join(kept)
    if re.fullmatch(r"\d{8}", joined):  # 20240305
        made = _make_date(original, int(joined[:4]), int(joined[4:6]), int(joined[6:]))
        return _ok(made) if isinstance(made, date) else made

    parts = re.fullmatch(r"(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{1,4})", joined)
    if not parts:
        return _invalid_date(original, "unrecognized date format")
    first, second, third = parts.groups()

    if len(first) == 4:  # year first: 2024-03-05, 2024/3/5
        made = _make_date(original, int(first), int(second), int(third))
        return _ok(made) if isinstance(made, date) else made
    if len(third) not in (2, 4):
        return _invalid_date(original, "unrecognized date format")
    if len(third) == 2 and int(first) > 31:  # yy-mm-dd
        made = _make_date(original, _two_digit_year(int(first)), int(second), int(third))
        return _ok(made) if isinstance(made, date) else made

    year, a, b = _year(third), int(first), int(second)
    by: Literal["unambiguous", "day_first"] = "unambiguous"
    if a > 12 and b > 12:
        return _invalid_date(original, "neither number can be a month")
    if a > 12:  # the first number cannot be a month, so it is the day
        day, month = a, b
    elif b > 12:  # the second number cannot be a month, so the first is
        month, day = a, b
    elif a == b:  # 05/05/2024: the same either way round
        day = month = a
    elif day_first is not None:  # both could be either: the caller's hint decides
        by = "day_first"
        day, month = (a, b) if day_first else (b, a)
    else:
        # with both numbers <= 12 either reading is a real date, so both are candidates
        first_reading = _make_date(original, year, b, a)
        second_reading = _make_date(original, year, a, b)
        if not isinstance(first_reading, date) or not isinstance(second_reading, date):
            return _invalid_date(original, "year is out of range")
        return DateResult(
            status="ambiguous",
            candidates=(first_reading, second_reading),
            reason=f"{original!r} could be day-first or month-first and day_first is not known",
        )
    made = _make_date(original, year, month, day)
    return _ok(made, by) if isinstance(made, date) else made
