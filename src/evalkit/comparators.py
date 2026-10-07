"""Field-level value comparators: normalized strings, fuzzy names, numeric tolerance, dates.

The eval harness decides whether a predicted value matches the label through the comparator its
field kind names. Kinds (and the fields that use them):

  name    fuzzy: vendor_name, buyer_name, row description (>= FUZZY_NAME_MIN similarity)
  text    normalized exact: vendor_address, payment_method
  code    normalized exact, digits-only when neither side has letters: vendor_tax_id,
          document_number, currency
  number  decimal with a tolerance: subtotal, tax, discount, total, row quantity/unit_price/amount
  date    both sides resolved to a date and compared exactly: document_date, due_date

The three thresholds are documented defaults (chosen with the user, "balanced"), not calibrated: a
run is always scored with the thresholds it records. `similarity` is reported even on a failed
fuzzy match so the error taxonomy can tell a near miss from a plain mismatch. Nothing here guesses:
an ambiguous date, a corrupt label or a non-finite number is a mismatch or a loud
`ComparatorError`, never a coerced value.
"""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Literal

from docextract.normalize import parse_date

FUZZY_NAME_MIN = 0.90
NUMERIC_REL_TOL = Decimal("0.005")
NUMERIC_ABS_TOL = Decimal("0.01")

Kind = Literal["name", "text", "code", "number", "date"]
How = Literal["exact", "normalized", "fuzzy", "numeric", "date"]

FIELD_KIND: dict[str, Kind] = {
    "vendor_name": "name",
    "vendor_address": "text",
    "vendor_tax_id": "code",
    "buyer_name": "name",
    "document_number": "code",
    "document_date": "date",
    "due_date": "date",
    "currency": "code",
    "payment_method": "text",
    "subtotal": "number",
    "tax": "number",
    "discount": "number",
    "total": "number",
}

CELL_KIND: dict[str, Kind] = {
    "description": "name",
    "quantity": "number",
    "unit_price": "number",
    "amount": "number",
}

_WORDS = re.compile(r"[^\W_]+")  # unicode letters and digits, no underscore
_NON_DIGITS = re.compile(r"\D")


class ComparatorError(ValueError):
    """A value cannot be compared: a corrupt label or a non-finite number, not a mismatch."""


@dataclass(frozen=True)
class Comparison:
    """One value pair's verdict. The string-ish comparators always set `similarity`."""

    matched: bool
    how: How | None = None
    similarity: float | None = None


def normalize_text(text: str) -> str:
    """Case-folded words joined by single spaces: the document's own case, punctuation and
    spacing taken away ("PT. Acme  Corp." -> "pt acme corp")."""
    return " ".join(_WORDS.findall(text.casefold()))


def digits_only(text: str) -> str:
    return _NON_DIGITS.sub("", text)


def _has_letters(text: str) -> bool:
    return any(character.isalpha() for character in text)


def _ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def _strings_match(expected: str, predicted: str) -> Comparison:
    if expected == predicted:
        return Comparison(True, "exact")
    if normalize_text(expected) == normalize_text(predicted):
        return Comparison(True, "normalized")
    return Comparison(False)


def compare_name(expected: str, predicted: str) -> Comparison:
    """Names are allowed to differ in formatting and slightly in spelling, up to FUZZY_NAME_MIN."""
    if expected == predicted:
        return Comparison(True, "exact", 1.0)
    left, right = normalize_text(expected), normalize_text(predicted)
    if left == right:
        return Comparison(True, "normalized", 1.0)
    ratio = _ratio(left, right)
    matched = ratio >= FUZZY_NAME_MIN
    return Comparison(matched, "fuzzy" if matched else None, ratio)


def compare_text(expected: str, predicted: str) -> Comparison:
    return _strings_match(expected, predicted)


def compare_code(expected: str, predicted: str) -> Comparison:
    """Tax ids, document numbers, currency codes: normalized exact, plus digits-only equality when
    neither side has a letter ("01.234.567.8-901.000" == "012345678901000")."""
    comparison = _strings_match(expected, predicted)
    if comparison.matched:
        return comparison
    no_letters = not _has_letters(expected) and not _has_letters(predicted)
    if no_letters and digits_only(expected) and digits_only(expected) == digits_only(predicted):
        return Comparison(True, "normalized")
    return Comparison(False)


def _to_decimal(value: Any, *, source: str) -> Decimal:
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except InvalidOperation:
            raise ComparatorError(f"{source}: {value!r} is not a decimal") from None
    else:
        raise ComparatorError(f"{source}: {value!r} is not a decimal")
    if not number.is_finite():
        raise ComparatorError(f"{source}: {value!r} is not a finite decimal")
    return number


def compare_number(
    expected: Any,
    predicted: Any,
    *,
    rel_tol: Decimal = NUMERIC_REL_TOL,
    abs_tol: Decimal = NUMERIC_ABS_TOL,
) -> Comparison:
    """Equal as decimals, or within max(abs_tol, rel_tol * |expected|) of it (print rounding)."""
    left = _to_decimal(expected, source="label")
    right = _to_decimal(predicted, source="prediction")
    tolerance = max(abs_tol, rel_tol * abs(left))
    matched = abs(left - right) <= tolerance
    return Comparison(matched, "numeric" if matched else None)


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        result = parse_date(value)  # day_first=None: an ambiguous date is never guessed
        return result.value if result.ok else None
    return None


def compare_date(expected: Any, predicted: Any) -> Comparison:
    left, right = _as_date(expected), _as_date(predicted)
    if left is None or right is None:
        return Comparison(False)
    return Comparison(left == right, "date" if left == right else None)


def compare_values(kind: Kind, expected: Any, predicted: Any) -> Comparison:
    if kind == "name":
        return compare_name(str(expected), str(predicted))
    if kind == "text":
        return compare_text(str(expected), str(predicted))
    if kind == "code":
        return compare_code(str(expected), str(predicted))
    if kind == "number":
        return compare_number(expected, predicted)
    return compare_date(expected, predicted)


def compare_field(name: str, expected: Any, predicted: Any) -> Comparison:
    """Compare one document field (scalar) by the kind its name names."""
    return compare_values(FIELD_KIND[name], expected, predicted)


def compare_cell(cell: str, expected: Any, predicted: Any) -> Comparison:
    """Compare one line-item cell by the kind its name names."""
    return compare_values(CELL_KIND[cell], expected, predicted)
