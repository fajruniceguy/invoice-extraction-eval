"""Field checks: dates, currency and the tax id.

Plausibility checks on single fields. Most results are things a document can legitimately
print (an issue date in the future on a pre-dated invoice, a tax id from another country), so
they only cap confidence — the cap itself is applied by `docextract.confidence.formula`, this
module only records the code. One result is a contradiction and is abstained by the abstention
policy (`HARD_CODES`): a due date before the document date.

Every function here returns a copy and never mutates its input; re-running is idempotent
(reasons.py strips a code before adding it again).
"""

from datetime import date

from docextract.extract.preprocess import MONEY_FIELDS
from docextract.extract.schema import DocumentExtraction
from docextract.normalize import convention_for, currency_marker
from docextract.validate.codes import (
    CURRENCY_CONTRADICTS_EVIDENCE,
    CURRENCY_UNKNOWN,
    DOCUMENT_DATE_IN_FUTURE,
    DUE_DATE_BEFORE_DOCUMENT_DATE,
    TAX_ID_IMPLAUSIBLE,
    TAX_ID_UNUSUAL_SHAPE,
)
from docextract.validate.reasons import with_document_reason

# An Indonesian tax id (NPWP): 15 digits in the old format, 16 in the new (NIK-based) one.
NPWP_DIGIT_COUNTS = frozenset({15, 16})


def validate_fields(
    document: DocumentExtraction, *, today: date | None = None
) -> DocumentExtraction:
    """Dates, currency and tax-id checks. `today` (default: the real today) decides whether a
    document date is in the future."""
    result = _check_dates(document, today or date.today())
    result = _check_currency(result)
    result = _check_tax_id(result)
    return result


def _check_dates(document: DocumentExtraction, today: date) -> DocumentExtraction:
    result = document
    issued = result.document_date
    if issued.status == "extracted" and issued.value is not None and issued.value > today:
        result = with_document_reason(result, "document_date", DOCUMENT_DATE_IN_FUTURE)
    due = result.due_date
    if (
        due.status == "extracted"
        and issued.status == "extracted"
        and due.value is not None
        and issued.value is not None
        and due.value < issued.value
    ):
        result = with_document_reason(result, "due_date", DUE_DATE_BEFORE_DOCUMENT_DATE)
    return result


def _check_currency(document: DocumentExtraction) -> DocumentExtraction:
    currency = document.currency
    if currency.status != "extracted" or currency.value is None:
        return document
    code = currency.value.strip().upper()
    result = document
    if convention_for(code) is None:
        result = with_document_reason(result, "currency", CURRENCY_UNKNOWN)
    # A currency marker printed in a money field's evidence is evidence in the document's own
    # words; if it names another currency, the currency claim is contradicted.
    evidences = [
        getattr(result, name).evidence or ""
        for name in MONEY_FIELDS
        if getattr(result, name).status == "extracted"
    ]
    marker = currency_marker(" ".join(evidences))
    if marker is not None and marker.upper() != code:
        result = with_document_reason(result, "currency", CURRENCY_CONTRADICTS_EVIDENCE)
    return result


def _check_tax_id(document: DocumentExtraction) -> DocumentExtraction:
    field = document.vendor_tax_id
    if field.status != "extracted" or field.value is None:
        return document
    digits = [character for character in field.value if character.isdigit()]
    if not digits:
        return with_document_reason(document, "vendor_tax_id", TAX_ID_IMPLAUSIBLE)
    if len(digits) not in NPWP_DIGIT_COUNTS:
        return with_document_reason(document, "vendor_tax_id", TAX_ID_UNUSUAL_SHAPE)
    return document
