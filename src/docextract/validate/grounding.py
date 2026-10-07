"""Evidence self-consistency: an extracted value must be traceable in its own evidence quote.

The model is asked for a verbatim quote and a page for every value it extracts, and the
preprocess contract (`extract.preprocess.check_contract`, repairable) makes it produce one.
This module checks the stronger claim that the quote actually supports the value:

  - money, quantities and dates: some span of the quote parses, with the shared normalizer and
    the document's own currency / day-first context, to exactly the value. "Rp 1.250.000,00"
    grounds 1250000.00, "5 Maret 2024" grounds 2024-03-05. An ambiguous span is NOT grounding:
    a reading the normalizer refused to make cannot confirm a value.
  - strings (names, addresses, payment methods, document numbers, tax ids): the value must be
    contained in the quote once case, punctuation and whitespace are ignored (its letters and
    digits, in order), so a name printed in capitals or a number formatted differently still
    grounds.

A value that is not traceable is contradicted by its own evidence, so it is recorded as
`evidence_not_grounded` (HARD: the abstention policy abstains the field, keeping the quote) —
except a row `description`, which is recorded as `description_not_verbatim` (soft): a reworded
description must not kill an otherwise grounded row, only cap it.

What this cannot catch: a quote AND a value the model invented together. That needs the
against-page grounding of a later milestone (OCR or a text sidecar), not this one.

Copies only, never mutates the input; re-running is idempotent.
"""

import re
from datetime import date
from decimal import Decimal
from typing import Any

from docextract.extract.preprocess import DATE_FIELDS, MONEY_FIELDS
from docextract.extract.schema import SCALAR_FIELD_NAMES, DocumentExtraction
from docextract.normalize import parse_date, parse_number
from docextract.validate.codes import DESCRIPTION_NOT_VERBATIM, EVIDENCE_NOT_GROUNDED
from docextract.validate.reasons import with_document_reason, with_row, with_row_reason

_NUMERIC_CELLS: tuple[str, ...] = ("quantity", "unit_price", "amount")
_ALNUM = re.compile(r"[^\W_]+")  # unicode letters and digits, no underscore
_DATE_WINDOW = 4  # tokens: "5 Maret 2024", "05 Mar 2024 14:30"


def _compact(text: str) -> str:
    """Letters and digits only, case-folded: what is left of a quote or value once the
    document's own formatting (case, punctuation, spacing) is taken away."""
    return "".join(_ALNUM.findall(text.casefold()))


def _string_grounded(quote: str, value: str) -> bool:
    """The value, formatting aside, contained in the quote."""
    compact = _compact(value)
    return bool(compact) and compact in _compact(quote)


def _number_candidates(text: str) -> list[str]:
    """The whitespace tokens of a quote, each also without surrounding punctuation, so both
    "(1.250)" and "1.250" are offered to the parser."""
    candidates: list[str] = []
    for token in text.split():
        candidates.append(token)
        stripped = token.strip(".,;:!?()[]{}\"'")
        if stripped and stripped != token:
            candidates.append(stripped)
    return candidates


def _number_grounded(quote: str, value: Decimal, currency: str | None) -> bool:
    for token in _number_candidates(quote):
        if not any(character.isdigit() for character in token):
            continue
        result = parse_number(token, currency=currency)
        if result.ok and result.value == value:
            return True
    return False


def _date_grounded(quote: str, value: date, day_first: bool | None) -> bool:
    tokens = quote.split()
    for size in range(1, _DATE_WINDOW + 1):
        for start in range(len(tokens) - size + 1):
            result = parse_date(" ".join(tokens[start : start + size]), day_first=day_first)
            if result.ok and result.value == value:
                return True
    return False


def _scalar_grounded(
    name: str, value: Any, quote: str, *, currency: str | None, day_first: bool | None
) -> bool:
    if name in MONEY_FIELDS:
        return isinstance(value, Decimal) and _number_grounded(quote, value, currency)
    if name in DATE_FIELDS:
        return isinstance(value, date) and _date_grounded(quote, value, day_first)
    return isinstance(value, str) and _string_grounded(quote, value)


def _flag_cells(document: DocumentExtraction, index: int, cells: list[str]) -> DocumentExtraction:
    """Add `cells` to the row's flagged_cells (each at most once, order kept)."""
    row = document.line_items[index]
    flagged = list(row.flagged_cells)
    for cell in cells:
        if cell not in flagged:
            flagged.append(cell)
    return with_row(document, index, row.model_copy(update={"flagged_cells": flagged}, deep=True))


def validate_grounding(document: DocumentExtraction) -> DocumentExtraction:
    """Flag every extracted scalar and row cell whose own evidence quote does not support it."""
    currency_field = document.currency
    currency = (
        currency_field.value if currency_field.status == "extracted" else None
    )
    day_first = True if (currency or "").strip().upper() == "IDR" else None

    result = document
    for name in SCALAR_FIELD_NAMES:
        field = getattr(result, name)
        if field.status != "extracted" or field.value is None:
            continue
        if not _scalar_grounded(
            name, field.value, field.evidence or "", currency=currency, day_first=day_first
        ):
            result = with_document_reason(result, name, EVIDENCE_NOT_GROUNDED)

    for index in range(len(result.line_items)):
        row = result.line_items[index]
        if row.status != "extracted":
            continue
        quote = row.evidence or ""
        ungrounded = [
            cell
            for cell in _NUMERIC_CELLS
            if getattr(row, cell) is not None
            and not _number_grounded(quote, getattr(row, cell), currency)
        ]
        if row.description is not None and not _string_grounded(quote, row.description):
            result = _flag_cells(result, index, ["description"])
            result = with_row_reason(result, index, DESCRIPTION_NOT_VERBATIM)
        if ungrounded:
            result = _flag_cells(result, index, ungrounded)
            result = with_row_reason(result, index, EVIDENCE_NOT_GROUNDED)
    return result
