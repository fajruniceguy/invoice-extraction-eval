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

The extractor's hook (`extraction_preprocess`) adds the evidence contract, which the schema cannot
check because it needs to know how many pages were sent: every `extracted` field and row must quote
its evidence and give its page, and any page given must be one of the pages sent. A breach is a
violation, so the model is re-asked (a repair) instead of the output being accepted uncited.
"""

import copy
import dataclasses
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from docextract.extract.schema import SCALAR_FIELD_NAMES
from docextract.normalize import parse_date, parse_number
from docextract.providers.openai_compat import Preprocess, PreprocessResult

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


# ---------------------------------------------------------------- the evidence contract


def _page_number(page: Any) -> int | None:
    """The page as validation will read it; None when it is not a whole number (validation then
    reports it itself)."""
    if isinstance(page, bool):
        return None
    if isinstance(page, int):
        return page
    if isinstance(page, str) and page.strip().isdigit():
        return int(page)
    if isinstance(page, Decimal) and page.is_finite() and page == page.to_integral_value():
        return int(page)  # model JSON is parsed with parse_float=Decimal: "page": 2.0
    if isinstance(page, float) and page.is_integer():
        return int(page)
    return None


def _labels(names: list[str], rows: list[int]) -> str:
    """'vendor_name, total, line_items[1-3,7]': rows as ranges, so 60 uncited rows stay short."""
    spans: list[str] = []
    for number in rows:
        if spans and spans[-1].split("-")[-1] == str(number - 1):
            spans[-1] = f"{spans[-1].split('-')[0]}-{number}"
        else:
            spans.append(str(number))
    return ", ".join([*names, *([f"line_items[{','.join(spans)}]"] if spans else [])])


def check_contract(payload: dict[str, Any], page_count: int) -> tuple[str, ...]:
    """What the payload breaks of the evidence contract for a document of `page_count` pages:
    an extracted field or row without evidence or page, or a page outside 1..page_count."""
    uncited: tuple[list[str], list[int]] = ([], [])
    unpaged: tuple[list[str], list[int]] = ([], [])
    off_page: tuple[list[str], list[int]] = ([], [])

    def inspect(item: Any, name: str | None, row: int | None) -> None:
        if not isinstance(item, dict):
            return  # not ours: validation reports a malformed field
        page = item.get("page")
        number = _page_number(page)
        targets = []
        if item.get("status") == "extracted":
            evidence = item.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                targets.append(uncited)
            if page is None:
                targets.append(unpaged)
        if number is not None and not 1 <= number <= page_count:
            targets.append(off_page)
        for names, rows in targets:
            if row is None:
                names.append(str(name))
            else:
                rows.append(row)

    for name in SCALAR_FIELD_NAMES:
        inspect(payload.get(name), name, None)
    rows = payload.get("line_items")
    if isinstance(rows, list):
        for index, row in enumerate(rows, start=1):
            inspect(row, None, index)

    found: list[str] = []
    if any(uncited):
        found.append(f"extracted without evidence (quote the printed text): {_labels(*uncited)}")
    if any(unpaged):
        found.append(f"extracted without a page: {_labels(*unpaged)}")
    if any(off_page):
        pages = f"{page_count} page{'s' if page_count != 1 else ''}"
        found.append(
            f"page outside 1..{page_count} (the document has {pages}): {_labels(*off_page)}"
        )
    return tuple(found)


def extraction_preprocess(page_count: int) -> Preprocess:
    """The extractor's hook for a document of `page_count` pages: the normalizer, then the
    evidence contract on what it produced."""
    if page_count < 1:
        raise ValueError("a document has at least one page")

    def hook(data: dict[str, Any]) -> PreprocessResult:
        cleaned = normalize_extraction_payload(data)
        return dataclasses.replace(cleaned, violations=check_contract(cleaned.data, page_count))

    return hook
