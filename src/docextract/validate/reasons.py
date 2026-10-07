"""Copying helpers that record a reason code on a field or row.

Every validator works the same way: it returns a copy and never mutates its input, and it
strips its own code before adding it, so validating twice gives the same result as once.
"""

from typing import Any

from docextract.extract.schema import DocumentExtraction, Field, LineItem


def with_reason(field: Field[Any], code: str) -> Field[Any]:
    """A copy of `field` with `code` appended to its reasons (once)."""
    reasons = [reason for reason in field.reasons if reason != code]
    return field.model_copy(update={"reasons": [*reasons, code]}, deep=True)


def with_document_reason(document: DocumentExtraction, name: str, code: str) -> DocumentExtraction:
    """A copy of `document` whose field `name` carries `code`."""
    field = with_reason(getattr(document, name), code)
    return document.model_copy(update={name: field}, deep=True)


def with_row_reason(document: DocumentExtraction, index: int, code: str) -> DocumentExtraction:
    """A copy of `document` whose line item at `index` (0-based) carries `code`."""
    rows = [*document.line_items]
    rows[index] = with_reason(document.line_items[index], code)
    return document.model_copy(update={"line_items": rows}, deep=True)


def with_row(document: DocumentExtraction, index: int, row: LineItem) -> DocumentExtraction:
    """A copy of `document` with `row` replacing the line item at `index` (0-based)."""
    rows = [*document.line_items]
    rows[index] = row
    return document.model_copy(update={"line_items": rows}, deep=True)
