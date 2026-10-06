"""Pre-labeling: a model drafts a label for a document, a human verifies every field.

Why a draft at all: transcribing 25 documents field by field from scratch is slow and error
prone, and checking a draft against the page is faster and (with every field forced through an
explicit verified / absent / unlabeled decision in the labeling app) just as reliable. Why a
DIFFERENT model than the extractor: if the model that is later evaluated also wrote the draft,
a verifier who nods too fast turns the extractor's own mistakes into ground truth.

Hard rules, enforced here and not left to the caller:
  * BLIND documents are never drafted (`BlindDocumentError`): they exist to measure the labels
    without any model influence. Only label_mode=prelabel documents that a human hand-labels
    qualify; documents that borrow labels (gt_from) or whose labels are generated do not.
  * the draft model must not be the extractor's (same endpoint and model is refused);
  * the model is shown only the STORED, REDACTED pages (verified against ingest.json), never
    a raw file and never an unredacted preview;
  * the model is asked to transcribe what is PRINTED, as strings; the shared normalizer turns
    them into values afterwards, and what it cannot resolve without guessing ("1.250" with no
    currency, "05/03/2024") becomes "__unlabeled__" with a note, never a guessed value.

A draft is a ground-truth file (`label_mode: prelabel`, no `labeled_at`: it is not a label until a
human has verified it) in ground_truth/drafts/.
"""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from docextract.config import Settings
from docextract.extract.preprocess import DATE_FIELDS, LINE_ITEM_CELLS, MONEY_FIELDS
from docextract.ingest.manifest import ManifestRow
from docextract.ingest.stored import StoredPagesError
from docextract.ingest.stored import stored_pages as load_stored_pages
from docextract.providers.openai_compat import (
    LLMResponse,
    Message,
    OpenAICompatClient,
    build_image_message,
)
from evalkit.groundtruth import SCALAR_FIELD_NAMES, UNLABELED, GroundTruth
from evalkit.gt_store import write_label
from evalkit.printed import currency_code, parse_amount, parse_printed_date

MAX_IMAGE_BYTES = 4_500_000  # Anthropic refuses images over ~5 MB


class PrelabelRefused(Exception):
    """This document cannot be pre-labeled (and why)."""


class BlindDocumentError(PrelabelRefused):
    """A blind document must never get a model draft."""


# ---------------------------------------------------------------- what may be drafted


def check_eligible(row: ManifestRow) -> None:
    """Raise unless `row` is a document a human hand-labels with label_mode=prelabel."""
    if row.label_mode == "blind":
        raise BlindDocumentError(
            f"{row.doc_id} is a BLIND document: a human labels it from the page alone, "
            "so a model draft must never exist for it"
        )
    if row.gt_from is not None:
        raise PrelabelRefused(
            f"{row.doc_id} takes its labels from {row.gt_from}: draft and label that one"
        )
    if row.capture == "constructed":
        raise PrelabelRefused(f"{row.doc_id} is constructed: its label is generated, not drafted")
    if row.label_mode != "prelabel":
        raise PrelabelRefused(
            f"{row.doc_id} has no label_mode: run scripts/assign_split.py "
            "(only label_mode=prelabel documents are drafted)"
        )


def eligibility(row: ManifestRow) -> str | None:
    """None if `row` may be drafted, else the reason it may not."""
    try:
        check_eligible(row)
    except PrelabelRefused as exc:
        return str(exc)
    return None


def check_model_differs(extractor: Settings | None) -> None:
    if extractor is not None and extractor.same_model_as_prelabel():
        raise PrelabelRefused(
            f"the pre-labeling model ({extractor.prelabel_model}) is the extractor's "
            f"({extractor.llm_model}) on the same endpoint: a draft by the model that is later "
            "evaluated would turn its own mistakes into ground truth. Set PRELABEL_* to "
            "another model."
        )


def stored_pages(doc_id: str, pages_dir: Path) -> list[Path]:
    """The document's stored (redacted) pages, checked against ingest.json. Nothing else is read."""
    try:
        return load_stored_pages(doc_id, pages_dir)
    except StoredPagesError as exc:
        raise PrelabelRefused(str(exc)) from None


# ---------------------------------------------------------------- the prompt

SYSTEM_PROMPT = """\
You transcribe business documents: invoices, receipts, tax invoices ("faktur pajak"), e-commerce \
orders, utility bills and hotel reservations. They are usually Indonesian, sometimes English. \
The page images come in order.

Report what is PRINTED, as it is printed:
- Copy every value exactly as printed, as a string, keeping the printed number and date format \
("Rp 1.250.000,00", "9.116,8", "05/03/2024", "5 Maret 2024"). Do not convert, round, add up or \
correct anything, and never compute a value that is not printed.
- If the document does not print a field, use null. Never guess. A value hidden under a black \
box is null.
- vendor_name / vendor_address / vendor_tax_id: the seller as printed (on a faktur the seller's \
NPWP). buyer_name: the customer as printed.
- document_number: the document's own number: invoice or receipt number, booking id, and for a \
faktur pajak the nomor seri faktur.
- document_date: when the document was issued or the transaction took place. due_date only if \
printed. payment_method as printed.
- currency: as printed (Rp, IDR, USD ...), or null if no currency is printed.
- subtotal: the amount before tax as printed (on a faktur the DPP). tax: the tax amount as \
printed (on a faktur the PPN). discount: the total discount if it is printed as one amount. \
total: the grand total or amount to pay, ONLY if printed as such (a faktur without a printed \
grand total has total null).
- line_items: one entry per printed row, in order, across all pages. A service charge, delivery \
fee, admin fee or rounding row is a line item too. A cell the row does not show is null. \
line_items is null if the document has no item table.
- notes: anything a person checking your transcript should know (unreadable parts, ambiguous \
numbers); otherwise null.
Answer with one JSON object and nothing else."""


def user_text(pages: int) -> str:
    return f"Transcribe this document ({pages} page{'s' if pages != 1 else ''}, in order)."


def build_messages(pages: list[Path], settings: Settings) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        build_image_message(
            user_text(len(pages)),
            list(pages),
            max_long_edge=settings.llm_image_max_long_edge,
            max_bytes=MAX_IMAGE_BYTES,
        ),
    ]


# ---------------------------------------------------------------- the model's answer

Printed = str | int | float | Decimal | None  # strings are asked for; numbers are tolerated


class DraftLineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: Printed = None
    quantity: Printed = None
    unit_price: Printed = None
    amount: Printed = None


class PrelabelDraft(BaseModel):
    """What the model returns: every value as printed. A key the model left out is "not
    answered" (unlabeled), a key it set to null is "not printed" (absent)."""

    model_config = ConfigDict(extra="ignore")

    vendor_name: Printed = None
    vendor_address: Printed = None
    vendor_tax_id: Printed = None
    buyer_name: Printed = None
    document_number: Printed = None
    document_date: Printed = None
    due_date: Printed = None
    currency: Printed = None
    payment_method: Printed = None
    line_items: list[DraftLineItem] | None = None
    subtotal: Printed = None
    tax: Printed = None
    discount: Printed = None
    total: Printed = None
    notes: str | None = None


# ---------------------------------------------------------------- draft -> label


def _text(value: Printed) -> str | None:
    """The printed text, or None for null / blank."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        return format(Decimal(str(value)), "f")
    text = str(value).strip()
    return text or None


def draft_to_label(draft: PrelabelDraft, *, doc_id: str, model: str) -> GroundTruth:
    """Turn the model's transcript into a ground-truth draft. Values the normalizer cannot
    resolve without guessing become "__unlabeled__" with a field note."""
    given = draft.model_fields_set
    values: dict[str, Any] = {}
    notes: dict[str, str] = {}

    currency_text = _text(draft.currency) if "currency" in given else None
    currency = currency_code(currency_text)

    for name in SCALAR_FIELD_NAMES:
        if name not in given:
            values[name] = UNLABELED
            notes[name] = "the model did not answer this field"
            continue
        text = _text(getattr(draft, name))
        if text is None:
            values[name] = None  # not printed
        elif name in MONEY_FIELDS:
            number = parse_amount(text, currency)
            if number.ok:
                values[name] = number.value
            else:
                values[name] = UNLABELED
                kind = "ambiguous number" if number.kind == "ambiguous" else "unreadable number"
                notes[name] = f"{kind}: printed {text!r} ({number.problem})"
        elif name in DATE_FIELDS:
            when = parse_printed_date(text, currency)
            if when.ok:
                values[name] = when.value
            else:
                values[name] = UNLABELED
                kind = "ambiguous date" if when.kind == "ambiguous" else "unreadable date"
                notes[name] = f"{kind}: printed {text!r} ({when.problem})"
        elif name == "currency":
            code = currency_code(text)
            if code is None:
                values[name] = UNLABELED
                notes[name] = f"unknown currency: printed {text!r}"
            else:
                values[name] = code
        else:
            values[name] = text

    if "line_items" not in given:
        values["line_items"] = UNLABELED
        notes["line_items"] = "the model did not answer this field"
    elif draft.line_items is None:
        values["line_items"] = None  # no itemization
    else:
        rows, row_notes = [], []
        for number, item in enumerate(draft.line_items, start=1):
            cells: dict[str, Any] = {}
            item_given = item.model_fields_set
            for cell in ("description", *LINE_ITEM_CELLS):
                if cell not in item_given:
                    cells[cell] = UNLABELED
                    continue
                text = _text(getattr(item, cell))
                if text is None or cell == "description":
                    cells[cell] = text
                    continue
                parsed = parse_amount(text, currency)
                if parsed.ok:
                    cells[cell] = parsed.value
                else:
                    cells[cell] = UNLABELED
                    kind = "ambiguous" if parsed.kind == "ambiguous" else "unreadable"
                    row_notes.append(f"row {number} {cell}: {kind} {text!r} ({parsed.problem})")
            rows.append(cells)
        values["line_items"] = rows
        if row_notes:
            notes["line_items"] = "; ".join(row_notes)

    header = f"pre-label draft by {model}; unverified"
    model_notes = (draft.notes or "").strip()
    return GroundTruth.model_validate(
        {
            "doc_id": doc_id,
            "label_mode": "prelabel",
            "notes": f"{header}. {model_notes}" if model_notes else header,
            "field_notes": notes,
            **values,
        }
    )


# ---------------------------------------------------------------- running it


@dataclass(frozen=True)
class DraftResult:
    doc_id: str
    label: GroundTruth
    response: LLMResponse[PrelabelDraft]
    pages: int

    @property
    def unlabeled_fields(self) -> list[str]:
        data = self.label.model_dump()
        return [name for name in (*SCALAR_FIELD_NAMES, "line_items") if data[name] == UNLABELED]


async def prelabel_document(
    row: ManifestRow,
    *,
    client: OpenAICompatClient,
    settings: Settings,
    pages_dir: Path,
    extractor: Settings | None = None,
) -> DraftResult:
    """Ask the pre-labeling model for a draft of `row`. `settings` is `prelabel_settings()`;
    `extractor` is the extractor's own settings, used to refuse the same model."""
    check_eligible(row)
    check_model_differs(extractor)
    pages = stored_pages(row.doc_id, pages_dir)
    if len(pages) > settings.prelabel_max_pages:
        raise PrelabelRefused(
            f"{row.doc_id} has {len(pages)} pages (PRELABEL_MAX_PAGES is "
            f"{settings.prelabel_max_pages}): split the bundle with page_range first"
        )
    response = await client.chat(
        build_messages(pages, settings),
        PrelabelDraft,
        temperature=0.0,
        max_tokens=settings.prelabel_max_tokens,
    )
    assert response.parsed is not None
    label = draft_to_label(response.parsed, doc_id=row.doc_id, model=settings.llm_model)
    return DraftResult(row.doc_id, label, response, len(pages))


def save_draft(row: ManifestRow, label: GroundTruth, drafts_dir: Path) -> Path:
    """Write the draft, re-checking that the document may have one: a blind document cannot get
    a draft written even by a caller that skipped the earlier checks."""
    check_eligible(row)
    if label.doc_id != row.doc_id:
        raise PrelabelRefused(f"the draft is for {label.doc_id}, not {row.doc_id}")
    return write_label(drafts_dir, label)
