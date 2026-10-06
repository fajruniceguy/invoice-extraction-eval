"""The labeling form as a pure state machine: no UI in here, so every rule is unit-tested.

A label is built field by field, and every field starts PENDING. The person looking at the page
must decide each one: `verified` (the value in the box is what the page prints), `absent` (the
page does not print it: stored as null) or `unlabeled` (not labeled: stored as "__unlabeled__",
skipped in scoring). The label can be saved only when nothing is pending, so a draft can never
turn into ground truth by being left alone.

  * Values are typed AS PRINTED ("Rp 1.250.000,00", "5 Maret 2024") and normalized when the field
    is verified, with the shared normalizer; anything it cannot settle without guessing is an
    inline error, never a guess (`evalkit.printed`).
  * A BLIND document starts empty and is never given a draft (`BlindDraftError`); a pre-labeled
    one starts from the model's draft, whose every field still has to be verified.
  * Saving writes ground_truth/<doc_id>.json with `label_mode` and `labeled_at`, and appends its
    own ground_truth/CHANGELOG.md line(s): overwriting a finished label needs a reason, and a
    label is never changed without a log entry (if the log cannot be written, the label file is
    put back as it was).
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from docextract.ingest.manifest import ManifestRow
from evalkit.groundtruth import SCALAR_FIELD_NAMES, UNLABELED, GroundTruth, LabelMode, is_unlabeled
from evalkit.gt_store import append_changelog, label_path, read_label, to_json_dict, write_label
from evalkit.printed import currency_code, parse_amount, parse_printed_date

Status = Literal["pending", "verified", "absent", "unlabeled"]
DECIDED: tuple[Status, ...] = ("verified", "absent", "unlabeled")
FieldKind = Literal["text", "date", "currency", "money"]

FIELD_KINDS: dict[str, FieldKind] = {
    "vendor_name": "text",
    "vendor_address": "text",
    "vendor_tax_id": "text",
    "buyer_name": "text",
    "document_number": "text",
    "document_date": "date",
    "due_date": "date",
    "currency": "currency",
    "payment_method": "text",
    "subtotal": "money",
    "tax": "money",
    "discount": "money",
    "total": "money",
}
ROW_CELLS = ("description", "quantity", "unit_price", "amount")
UNLABELED_CELL = "?"  # typed in a line-item cell to leave that cell unlabeled


class LabelingError(Exception):
    """The form cannot be built, saved or changed that way."""


class BlindDraftError(LabelingError):
    """A blind document must never see a model draft."""


def _display(value: Any) -> str:
    if value is None or is_unlabeled(value):
        return ""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


@dataclass
class FieldState:
    text: str = ""
    status: Status = "pending"
    note: str = ""  # saved with the label as a field note
    error: str | None = None  # why the last attempt to verify failed
    proposal: str | None = None  # what the draft said ("absent", "unlabeled", or its text)
    hint: str = ""  # the draft's own remark about this field (shown, never saved)


@dataclass
class LabelForm:
    doc_id: str
    label_mode: LabelMode
    fields: dict[str, FieldState]
    rows: list[dict[str, str]] = field(default_factory=list)
    rows_status: Status = "pending"
    rows_errors: list[str] = field(default_factory=list)
    rows_note: str = ""
    rows_proposal: str | None = None
    rows_hint: str = ""
    rows_values: list[dict[str, Any]] = field(default_factory=list)  # normalized, once verified
    notes: str = ""
    draft_notes: str = ""  # the draft's own remarks about the document (shown, never saved)

    # ------------------------------------------------------------ construction

    @classmethod
    def start(cls, row: ManifestRow, draft: GroundTruth | None = None) -> "LabelForm":
        """A form with every field pending. `draft` pre-fills it (never for a blind document)."""
        mode = labelable_mode(row)
        if mode == "blind" and draft is not None:
            raise BlindDraftError(
                f"{row.doc_id} is a BLIND document: it is labeled from the page alone, "
                "a draft must not be shown for it"
            )
        form = cls(row.doc_id, mode, {name: FieldState() for name in SCALAR_FIELD_NAMES})
        if draft is not None:
            form._prefill(draft)
        return form

    @classmethod
    def from_label(cls, row: ManifestRow, label: GroundTruth) -> "LabelForm":
        """Re-open a finished label: every field starts decided; editing one makes it pending."""
        mode = labelable_mode(row)
        if label.label_mode != mode:
            raise LabelingError(
                f"{row.doc_id}: the label says label_mode={label.label_mode!r} but the manifest "
                f"says {mode!r}"
            )
        form = cls(row.doc_id, mode, {name: FieldState() for name in SCALAR_FIELD_NAMES})
        for name in SCALAR_FIELD_NAMES:
            value = getattr(label, name)
            state = form.fields[name]
            state.note = label.field_notes.get(name, "")
            if is_unlabeled(value):
                state.status = "unlabeled"
            elif value is None:
                state.status = "absent"
            else:
                state.text, state.status = _display(value), "verified"
        items = label.line_items
        form.rows_note = label.field_notes.get("line_items", "")
        if is_unlabeled(items):
            form.rows_status = "unlabeled"
        elif items is None:
            form.rows_status = "absent"
        else:
            assert isinstance(items, list)
            form.rows = [
                {cell: _cell_text(getattr(item, cell)) for cell in ROW_CELLS} for item in items
            ]
            form.rows_values = [{cell: getattr(item, cell) for cell in ROW_CELLS} for item in items]
            form.rows_status = "verified"
        form.notes = label.notes or ""
        return form

    def _prefill(self, draft: GroundTruth) -> None:
        for name in SCALAR_FIELD_NAMES:
            value = getattr(draft, name)
            state = self.fields[name]
            if is_unlabeled(value):
                state.proposal = "unlabeled"
            elif value is None:
                state.proposal = "absent"
            else:
                state.text = state.proposal = _display(value)
            state.hint = draft.field_notes.get(name, "")
        items = draft.line_items
        self.rows_hint = draft.field_notes.get("line_items", "")
        if is_unlabeled(items):
            self.rows_proposal = "unlabeled"
        elif items is None:
            self.rows_proposal = "absent"
        else:
            assert isinstance(items, list)
            self.rows = [
                {cell: _cell_text(getattr(item, cell)) for cell in ROW_CELLS} for item in items
            ]
            self.rows_proposal = f"{len(items)} row(s)"
        self.draft_notes = draft.notes or ""

    # ------------------------------------------------------------ scalar fields

    def _state(self, name: str) -> FieldState:
        if name not in self.fields:
            raise LabelingError(f"no such field: {name!r}")
        return self.fields[name]

    def set_text(self, name: str, text: str) -> None:
        """Edit the box. A decision already made no longer holds: the field is pending again."""
        state = self._state(name)
        if text != state.text:
            state.text, state.status, state.error = text, "pending", None

    def set_note(self, name: str, note: str) -> None:
        if name == "line_items":
            self.rows_note = note
        else:
            self._state(name).note = note

    def currency_context(self) -> str | None:
        """The currency typed so far (verified or not): an absent or unlabeled currency has an
        empty box, so it gives no context."""
        return currency_code(self.fields["currency"].text)

    def verify(self, name: str) -> bool:
        """Settle the field on what is in the box. False (with `error` set) if it cannot be read
        without guessing; the field stays as it was."""
        state, kind = self._state(name), FIELD_KINDS[name]
        text = state.text.strip()
        state.error = None
        if not text:
            state.error = "type the value as printed, or mark the field absent"
            return False
        value: str | None
        if kind == "text":
            value = text
        elif kind == "currency":
            value = currency_code(text)
            if value is None:
                state.error = f"{text!r} is not a currency I know: type an ISO code such as IDR"
        elif kind == "date":
            parsed = parse_printed_date(text, self.currency_context())
            value = parsed.value
            if value is None:
                state.error = f"{parsed.kind}: {parsed.problem}. Type it as YYYY-MM-DD."
        else:
            parsed = parse_amount(text, self.currency_context())
            value = parsed.value
            if value is None:
                state.error = (
                    f"{parsed.kind}: {parsed.problem}. Set the currency first, or type the plain "
                    "decimal (for example 1250000.00)."
                )
        if value is None:
            return False
        state.text, state.status = value, "verified"
        return True

    def mark_absent(self, name: str) -> None:
        state = self._state(name)
        state.text, state.status, state.error = "", "absent", None

    def mark_unlabeled(self, name: str) -> None:
        state = self._state(name)
        state.text, state.status, state.error = "", "unlabeled", None

    def reset(self, name: str) -> None:
        state = self._state(name)
        state.status, state.error = "pending", None

    # ------------------------------------------------------------ line items

    def _rows_changed(self) -> None:
        self.rows_status, self.rows_errors, self.rows_values = "pending", [], []

    def set_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        cleaned = [
            {
                cell: "" if row.get(cell) is None else str(row.get(cell)).strip()
                for cell in ROW_CELLS
            }
            for row in rows
        ]
        if cleaned != self.rows:
            self.rows = cleaned
            self._rows_changed()

    def add_row(self) -> None:
        self.rows.append(dict.fromkeys(ROW_CELLS, ""))
        self._rows_changed()

    def remove_row(self, index: int) -> None:
        if not 0 <= index < len(self.rows):
            raise LabelingError(f"no row {index + 1}")
        del self.rows[index]
        self._rows_changed()

    def verify_rows(self) -> bool:
        """Settle the table on its rows. A cell left empty is absent (the row does not show it),
        `?` leaves it unlabeled, and an amount is read like any other (errors say which cell)."""
        errors: list[str] = []
        values: list[dict[str, Any]] = []
        currency = self.currency_context()
        if not self.rows:
            self.rows_errors = ["add the rows, or mark the table absent (no itemization)"]
            return False
        for number, row in enumerate(self.rows, start=1):
            cells: dict[str, Any] = {}
            for cell in ROW_CELLS:
                text = row[cell].strip()
                if not text:
                    cells[cell] = None
                elif text in (UNLABELED_CELL, UNLABELED):
                    cells[cell] = UNLABELED
                elif cell == "description":
                    cells[cell] = text
                else:
                    parsed = parse_amount(text, currency)
                    if parsed.value is None:
                        errors.append(f"row {number} {cell}: {parsed.kind}: {parsed.problem}")
                    cells[cell] = parsed.value
            values.append(cells)
        self.rows_errors = errors
        if errors:
            return False
        for row, cells in zip(self.rows, values, strict=True):
            for cell in ROW_CELLS:
                row[cell] = (
                    UNLABELED_CELL
                    if cells[cell] == UNLABELED
                    else ""
                    if cells[cell] is None
                    else str(cells[cell])
                )
        self.rows_values, self.rows_status = values, "verified"
        return True

    def mark_rows_absent(self) -> None:
        """The document has no item table: stored as `line_items: null`."""
        self.rows_status, self.rows_errors, self.rows_values = "absent", [], []

    def mark_rows_unlabeled(self) -> None:
        self.rows_status, self.rows_errors, self.rows_values = "unlabeled", [], []

    # ------------------------------------------------------------ progress and the label

    def pending(self) -> list[str]:
        names = [name for name, state in self.fields.items() if state.status == "pending"]
        return [*names, *(["line_items"] if self.rows_status == "pending" else [])]

    def progress(self) -> tuple[int, int]:
        total = len(self.fields) + 1
        return total - len(self.pending()), total

    @property
    def can_save(self) -> bool:
        return not self.pending()

    def build_label(self, now: datetime | None = None) -> GroundTruth:
        """The finished label. Refused while any field is pending."""
        pending = self.pending()
        if pending:
            raise LabelingError(
                f"{len(pending)} field(s) still pending: {', '.join(pending)}. "
                "Decide each one: verified, absent or unlabeled."
            )
        stamp = now or datetime.now(UTC)
        if stamp.tzinfo is None:
            raise LabelingError("labeled_at needs a timezone")
        values: dict[str, Any] = {}
        for name, state in self.fields.items():
            values[name] = (
                UNLABELED
                if state.status == "unlabeled"
                else None
                if state.status == "absent"
                else state.text
            )
        if self.rows_status == "unlabeled":
            values["line_items"] = UNLABELED
        elif self.rows_status == "absent":
            values["line_items"] = None
        else:
            values["line_items"] = self.rows_values
        notes = {
            name: state.note.strip() for name, state in self.fields.items() if state.note.strip()
        }
        if self.rows_note.strip():
            notes["line_items"] = self.rows_note.strip()
        return GroundTruth.model_validate(
            {
                "doc_id": self.doc_id,
                "label_mode": self.label_mode,
                "labeled_at": stamp,
                "notes": self.notes.strip() or None,
                "field_notes": notes,
                **values,
            }
        )

    # ------------------------------------------------------------ keeping work in progress

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["rows_values"] = []  # recomputed by verify_rows when the work is reopened
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LabelForm":
        """Reopen saved work. Nothing in the file is trusted: its shape is checked, and every
        field it calls verified has to verify AGAIN, so a stale or hand-edited file cannot
        smuggle in a value that was never read."""
        try:
            mode = data["label_mode"]
            if mode not in ("prelabel", "blind"):
                raise ValueError(f"label_mode {mode!r}")
            saved = data.get("fields", {})
            if not isinstance(saved, Mapping):
                raise TypeError("fields must be a mapping")
            fields = {name: _saved_state(name, saved.get(name, {})) for name in SCALAR_FIELD_NAMES}
            saved_rows = data.get("rows", [])
            if not isinstance(saved_rows, list):
                raise TypeError("rows must be a list")
            rows_status = data.get("rows_status", "pending")
            if rows_status not in ("pending", *DECIDED):
                raise ValueError(f"rows_status {rows_status!r}")
            form = cls(
                doc_id=str(data["doc_id"]),
                label_mode=mode,
                fields=fields,
                rows=[_saved_row(row) for row in saved_rows],
                rows_status="pending",
                rows_note=_saved_text(data.get("rows_note", "")),
                rows_proposal=_saved_optional_text(data.get("rows_proposal")),
                rows_hint=_saved_text(data.get("rows_hint", "")),
                notes=_saved_text(data.get("notes", "")),
                draft_notes=_saved_text(data.get("draft_notes", "")),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise LabelingError(f"unusable saved work: {exc}") from exc
        for name, state in form.fields.items():
            if state.status == "verified":
                state.status = "pending"
                form.verify(name)
        if rows_status == "verified":
            form.verify_rows()
        elif rows_status != "pending":
            form.rows_status = rows_status
        return form


def _saved_text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected text, got {type(value).__name__}")
    return value


def _saved_optional_text(value: Any) -> str | None:
    return None if value is None else _saved_text(value)


def _saved_state(name: str, raw: Any) -> FieldState:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name}: expected a mapping")
    unknown = set(raw) - {"text", "status", "note", "error", "proposal", "hint"}
    if unknown:
        raise ValueError(f"{name}: unknown key(s) {sorted(unknown)}")
    status = raw.get("status", "pending")
    if status not in ("pending", *DECIDED):
        raise ValueError(f"{name}: status {status!r}")
    return FieldState(
        text=_saved_text(raw.get("text", "")),
        status=status,
        note=_saved_text(raw.get("note", "")),
        error=_saved_optional_text(raw.get("error")),
        proposal=_saved_optional_text(raw.get("proposal")),
        hint=_saved_text(raw.get("hint", "")),
    )


def _saved_row(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise TypeError("a row must be a mapping")
    return {cell: _saved_text(raw.get(cell, "")) for cell in ROW_CELLS}


def labelable_mode(row: ManifestRow) -> LabelMode:
    """The label_mode of a document a person labels by hand, or a LabelingError saying why not."""
    if row.gt_from is not None:
        raise LabelingError(f"{row.doc_id} takes its labels from {row.gt_from}: label that one")
    if row.capture == "constructed":
        raise LabelingError(f"{row.doc_id} is constructed: its label is generated")
    if row.label_mode is None:
        raise LabelingError(
            f"{row.doc_id} has no label_mode: run scripts/assign_split.py before labeling"
        )
    return row.label_mode


def _cell_text(value: Any) -> str:
    if is_unlabeled(value):
        return UNLABELED_CELL
    return _display(value)


# ---------------------------------------------------------------- the changelog


def _shown(value: Any, limit: int = 60) -> str:
    if value is None:
        text = "null"
    elif isinstance(value, list):
        text = f"{len(value)} rows"
    elif isinstance(value, dict):
        text = ", ".join(f"{k}: {v}" for k, v in value.items()) or "{}"
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def changelog_lines(
    old: GroundTruth | None, new: GroundTruth, *, today: date, reason: str | None
) -> list[str]:
    """One line per change, in the CHANGELOG format `date | file | field | old -> new | reason`."""
    file = f"{new.doc_id}.json"
    if old is None:
        mode = f"label_mode={new.label_mode}"
        return [f"{today.isoformat()} | {file} | (new file) | - -> {mode} | labeled by hand"]
    before, after = to_json_dict(old), to_json_dict(new)
    lines = []
    for key in [*before, *(k for k in after if k not in before)]:
        if key == "labeled_at" or before.get(key) == after.get(key):
            continue
        lines.append(
            f"{today.isoformat()} | {file} | {key} | "
            f"{_shown(before.get(key))} -> {_shown(after.get(key))} | {reason}"
        )
    return lines


# ---------------------------------------------------------------- saving


@dataclass(frozen=True)
class SaveResult:
    path: Path
    status: Literal["created", "replaced", "unchanged"]
    changelog: list[str]


def save_label(
    label: GroundTruth,
    row: ManifestRow,
    *,
    ground_truth_dir: Path,
    changelog: Path,
    drafts_dir: Path | None = None,
    reason: str | None = None,
    today: date | None = None,
) -> SaveResult:
    """Write the label and log it. Refused for a blind document that has a draft, and (without a
    `reason`) for a label that would replace a finished one. An identical label is not rewritten."""
    mode = labelable_mode(row)
    if label.doc_id != row.doc_id:
        raise LabelingError(f"the label is for {label.doc_id}, not {row.doc_id}")
    if label.label_mode != mode:
        raise LabelingError(
            f"the label says label_mode={label.label_mode!r} but the manifest says {mode!r}"
        )
    if mode == "blind" and drafts_dir is not None and label_path(drafts_dir, row.doc_id).exists():
        raise BlindDraftError(
            f"{row.doc_id} is a BLIND document but has a model draft "
            f"({label_path(drafts_dir, row.doc_id)}): delete it before saving a blind label"
        )
    target = label_path(ground_truth_dir, row.doc_id)
    old = read_label(target) if target.is_file() else None
    stamp = today or (label.labeled_at or datetime.now(UTC)).date()
    if old is not None:
        if to_json_dict(old) | {"labeled_at": None} == to_json_dict(label) | {"labeled_at": None}:
            return SaveResult(target, "unchanged", [])
        if not (reason or "").strip():
            raise LabelingError(
                f"{target.name} already exists: say why it is being replaced (the reason goes "
                "into CHANGELOG.md)"
            )
    lines = changelog_lines(old, label, today=stamp, reason=(reason or "").strip() or None)
    previous = target.read_bytes() if old is not None else None
    write_label(ground_truth_dir, label)
    try:
        append_changelog(changelog, lines)
    except BaseException:
        # a label must never change without its log line: put the old file back
        if previous is None:
            target.unlink(missing_ok=True)
        else:
            target.write_bytes(previous)
        raise
    return SaveResult(target, "created" if old is None else "replaced", lines)
