"""Turning what a document PRINTS into label values: shared by the pre-labeler and the labeling app.

Both read a person's or a model's transcript of the page ("Rp 1.250.000,00", "5 Maret 2024") and
need the value that goes into the label ("1250000.00", "2024-03-05"). They must agree, so the rule
lives here once, on top of the shared normalizer (`docextract.normalize`), which never guesses:

  * an amount keeps its printed scale ("1.250.000,00" -> "1250000.00", "1.250.000" -> "1250000");
  * what the text or the currency context cannot settle ("1.250" with no currency, "05/03/2024"
    on a document with no known convention) is reported as a PROBLEM, never resolved by a guess;
  * an IDR document is read day-first.
"""

from dataclasses import dataclass
from typing import Literal

from docextract.normalize import parse_date, parse_number

CURRENCY_NAMES = {
    "rp": "IDR",
    "rupiah": "IDR",
    "idr": "IDR",
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "s$": "SGD",
    "sgd": "SGD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}


@dataclass(frozen=True)
class Parsed:
    """`value` is set when the text could be read without guessing; otherwise `problem` says why
    not and `kind` whether it was ambiguous or simply unreadable."""

    value: str | None = None
    kind: Literal["ok", "ambiguous", "invalid"] = "ok"
    problem: str = ""

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


def currency_code(text: str | None) -> str | None:
    """'Rp' / 'rupiah' / 'IDR' -> 'IDR'; any other three letters are kept as they are; else None."""
    if not text:
        return None
    key = text.strip().rstrip(".").lower()
    if key in CURRENCY_NAMES:
        return CURRENCY_NAMES[key]
    return key.upper() if len(key) == 3 and key.isalpha() else None


def parse_amount(text: str, currency: str | None) -> Parsed:
    """An amount or quantity as printed -> a plain decimal string at the printed scale."""
    result = parse_number(text, currency=currency)
    if result.ok:
        return Parsed(value=result.text)
    return Parsed(
        kind="ambiguous" if result.status == "ambiguous" else "invalid", problem=result.reason
    )


def parse_printed_date(text: str, currency: str | None) -> Parsed:
    """A date as printed -> ISO 8601. An IDR document is read day-first."""
    result = parse_date(text, day_first=True if currency == "IDR" else None)
    if result.ok:
        return Parsed(value=result.iso)
    return Parsed(
        kind="ambiguous" if result.status == "ambiguous" else "invalid", problem=result.reason
    )
