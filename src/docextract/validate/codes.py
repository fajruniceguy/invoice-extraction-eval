"""The reason-code vocabulary.

Every reason a validator, the confidence formula or the abstention policy can put on a field
or row is one of these codes, so a reason is machine-readable (M5's error taxonomy) and never
free text. Each code is owned by the module that checks for it; they live together here so
`docextract.confidence` can name them without importing `docextract.validate` (and `validate`
itself imports this leaf module, never the other way round).

HARD codes make the abstention policy (`docextract.confidence.abstain`) flip an `extracted`
field or row to `abstained`; every other code only caps confidence
(`docextract.confidence.formula`), so the field stays `extracted` unless the threshold abstains
it. `CONFIDENCE_BELOW_THRESHOLD` is neither: it is the outcome the policy itself records.
"""

# Row arithmetic (validate.arithmetic): quantity x unit_price should equal amount.
QTY_X_PRICE_NE_AMOUNT = "qty_x_price_ne_amount"

# Document reconciliation (validate.arithmetic): the printed summary should add up.
LINES_NE_SUBTOTAL = "lines_ne_subtotal"
COMPONENTS_NE_TOTAL = "components_ne_total"

# Field checks (validate.fields): dates, currency, tax id.
DUE_DATE_BEFORE_DOCUMENT_DATE = "due_date_before_document_date"
DOCUMENT_DATE_IN_FUTURE = "document_date_in_future"
CURRENCY_UNKNOWN = "currency_unknown"
CURRENCY_CONTRADICTS_EVIDENCE = "currency_contradicts_evidence"
TAX_ID_IMPLAUSIBLE = "tax_id_implausible"
TAX_ID_UNUSUAL_SHAPE = "tax_id_unusual_shape"

# Evidence grounding (validate.grounding): the value must be traceable in its own quote.
EVIDENCE_NOT_GROUNDED = "evidence_not_grounded"
DESCRIPTION_NOT_VERBATIM = "description_not_verbatim"

# Abstention (confidence.abstain): recorded when the policy flips a field itself.
CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"

REASON_CODES: tuple[str, ...] = (
    QTY_X_PRICE_NE_AMOUNT,
    LINES_NE_SUBTOTAL,
    COMPONENTS_NE_TOTAL,
    DUE_DATE_BEFORE_DOCUMENT_DATE,
    DOCUMENT_DATE_IN_FUTURE,
    CURRENCY_UNKNOWN,
    CURRENCY_CONTRADICTS_EVIDENCE,
    TAX_ID_IMPLAUSIBLE,
    TAX_ID_UNUSUAL_SHAPE,
    EVIDENCE_NOT_GROUNDED,
    DESCRIPTION_NOT_VERBATIM,
    CONFIDENCE_BELOW_THRESHOLD,
)

# A field or row carrying one of these is contradicted, not merely shaky: the abstention
# policy turns it into `abstained` (value null) whatever its confidence.
HARD_CODES: frozenset[str] = frozenset({DUE_DATE_BEFORE_DOCUMENT_DATE, EVIDENCE_NOT_GROUNDED})
