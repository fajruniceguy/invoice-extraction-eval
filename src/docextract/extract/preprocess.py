"""Deterministic clean-up of model output BEFORE it is validated, and therefore before any repair.

Models often print money and dates the way the document does ("1.250.000,00", "5 Maret 2024")
instead of the plain decimals and ISO dates the schema asks for. Re-asking the model for that
costs a call and may not even work, so the client runs this first (`chat(preprocess=...)`):

    parse JSON -> normalize (this module) -> validate -> repair ONLY if that still fails

What is normalized: string values of the money fields (subtotal, tax, discount, total), of the
date fields (document_date, due_date) and of the line-item cells (quantity, unit_price, amount).
Numbers and null are left alone. The currency context comes from the payload's own `currency`
field, and an IDR document is read day-first. Anything that cannot be resolved without guessing
(an ambiguous "1.250" with no currency, a malformed date) is LEFT UNCHANGED so validation fails
and the repair prompt can say exactly what the normalizer could not decide.
"""

import copy
from collections.abc import Callable
from typing import Any

from docextract.normalize import parse_date, parse_number
from docextract.providers.openai_compat import PreprocessResult

MONEY_FIELDS = ("subtotal", "tax", "discount", "total")
DATE_FIELDS = ("document_date", "due_date")
LINE_ITEM_CELLS = ("quantity", "unit_price", "amount")

Convert = Callable[[str], tuple[str | None, str | None]]  # text -> (normalized, problem)


def _currency(payload: dict[str, Any]) -> str | None:
    field = payload.get("currency")
    value = field.get("value") if isinstance(field, dict) else None
    return value.strip().upper() or None if isinstance(value, str) else None


def normalize_extraction_payload(data: dict[str, Any]) -> PreprocessResult:
    payload = copy.deepcopy(data)  # never mutate what the model said
    problems: list[str] = []
    changes: list[str] = []
    currency = _currency(payload)
    day_first = True if currency == "IDR" else None

    def number(text: str) -> tuple[str | None, str | None]:
        result = parse_number(text, currency=currency)
        return (result.text, None) if result.ok else (None, result.reason)

    def date_(text: str) -> tuple[str | None, str | None]:
        result = parse_date(text, day_first=day_first)
        return (result.iso, None) if result.ok else (None, result.reason)

    def apply(container: Any, key: str, label: str, convert: Convert) -> None:
        if not isinstance(container, dict):
            return
        value = container.get(key)
        if not isinstance(value, str):
            return  # numbers, null and anything else are not ours to touch
        normalized, problem = convert(value)
        if problem is not None:
            problems.append(f"{label}: {problem}")
        elif normalized != value:
            container[key] = normalized
            changes.append(f"{label}: {value!r} -> {normalized!r}")

    for name in MONEY_FIELDS:
        apply(payload.get(name), "value", f"{name}.value", number)
    for name in DATE_FIELDS:
        apply(payload.get(name), "value", f"{name}.value", date_)
    rows = payload.get("line_items")
    if isinstance(rows, list):
        for index, row in enumerate(rows, start=1):
            for cell in LINE_ITEM_CELLS:
                apply(row, cell, f"line_items[{index}].{cell}", number)

    return PreprocessResult(data=payload, problems=tuple(problems), changes=tuple(changes))
