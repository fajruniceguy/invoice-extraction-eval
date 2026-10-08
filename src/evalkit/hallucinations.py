"""Hallucinated values taken apart: what they are, where they sit, and whether the label holds.

`hallucinated_value` is the scorer's class for "the model returned a value where the label says
the field is absent". It is one number on the scorecard; this module says what is behind it:

  * every case, scalar field or row cell, with its document, label owner, capture, confidence,
    the validator codes on it and the model's own note;
  * the denominators (labeled-absent fields, null label cells in matched rows), so it is a rate;
  * an audit of documents that SHARE a label through `gt_from`: copies of one page should read
    alike, so a copy whose exact-match rate is far below its owner's, while the run reads
    different values in each, is flagged for a human to look at the pages.

The audit only flags. Whether the label or the model is wrong is a judgement about the page that
a tool cannot make; the report says so, and the verdict is written by whoever looked.

Offline and deterministic. Nothing here changes a score, a label or a record. The committed
report (`to_markdown`) holds values and the model's one-line notes but never an evidence quote;
`review_markdown` is the list for a human, with the quotes, and belongs outside the repository's
tracked folders.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from docextract.extract.runner import DocRecord, RunRecord
from docextract.extract.schema import CELL_NAMES, SCALAR_FIELD_NAMES
from docextract.ingest.manifest import ManifestRow, read_manifest
from docextract.validate.codes import REASON_CODES
from evalkit.groundtruth import is_unlabeled
from evalkit.gt_store import ResolvedLabel, load_labels
from evalkit.matching import match_rows
from evalkit.metrics import score_field

# The audit's rule, fixed before the result was looked at (the first look: owner 0.8, copies 0.2).
LINEAGE_MIN_FIELDS = 3  # labeled non-null scalar fields needed on BOTH sides to say anything
LINEAGE_GAP = Fraction(1, 2)  # a member this far (or more) below its owner is suspect
IDENTIFIER_FIELDS = ("document_number", "document_date", "total")
CONFIDENCE_BANDS: tuple[tuple[str, float, float], ...] = (
    (">= 0.90", 0.9, 1.01),
    ("0.70 - 0.89", 0.7, 0.9),
    ("< 0.70", 0.0, 0.7),
)


class HallucinationError(Exception):
    """The run folder cannot be analysed (a missing or unreadable record)."""


@dataclass(frozen=True)
class Hallucination:
    """One value returned where the label says absent."""

    doc_id: str
    owner: str  # the document whose label file was used
    level: str  # "field" or "cell"
    name: str  # the field, or the row cell
    row: int | None  # the predicted row (0-based) for a cell
    value: str
    confidence: float
    page: int | None
    codes: tuple[str, ...]  # validator reason codes on the field or row
    note: str  # the model's own free-text reasons, joined
    quote: str  # the evidence quote (review list only)
    raw_also: bool  # the model's own answer (`parsed`) had a value there too
    capture: str
    doc_type: str
    label_mode: str


@dataclass(frozen=True)
class Member:
    """A document of a shared-label group, scored on the labeled non-null scalar fields."""

    doc_id: str
    labeled: int
    correct: int
    suspect: bool = False

    @property
    def rate(self) -> Fraction | None:
        return Fraction(self.correct, self.labeled) if self.labeled else None


@dataclass(frozen=True)
class LineageGroup:
    owner: Member
    members: tuple[Member, ...]
    reads: dict[str, tuple[str, ...]]  # identifier field -> distinct things the run read

    @property
    def suspects(self) -> tuple[Member, ...]:
        return tuple(member for member in self.members if member.suspect)


@dataclass(frozen=True)
class HallucinationReport:
    run_id: str
    model: str
    prompt_sha256: str
    documents: int
    items: tuple[Hallucination, ...]
    absent_fields: int  # labeled scalar fields whose label is null
    null_cells: int  # compared row cells whose label is null
    lineage: tuple[LineageGroup, ...]

    @property
    def fields(self) -> tuple[Hallucination, ...]:
        return tuple(item for item in self.items if item.level == "field")

    @property
    def cells(self) -> tuple[Hallucination, ...]:
        return tuple(item for item in self.items if item.level == "cell")


def is_suspect(owner: Member, member: Member) -> bool:
    """The audit's rule: both sides have enough labeled fields and the member is far below."""
    if owner.labeled < LINEAGE_MIN_FIELDS or member.labeled < LINEAGE_MIN_FIELDS:
        return False
    assert owner.rate is not None and member.rate is not None
    return owner.rate - member.rate >= LINEAGE_GAP


def _note(reasons: list[str]) -> str:
    """The model's own words: the reasons that are not validator codes."""
    return " | ".join(reason for reason in reasons if reason not in REASON_CODES)


def _codes(reasons: list[str]) -> tuple[str, ...]:
    return tuple(reason for reason in reasons if reason in REASON_CODES)


def _read_record(run_dir: Path, doc_id: str) -> DocRecord:
    path = run_dir / "docs" / f"{doc_id}.json"
    try:
        record = DocRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HallucinationError(f"{path}: cannot be read ({exc})") from exc
    except ValidationError as exc:
        raise HallucinationError(f"{path}: not a document record ({exc})") from exc
    if record.doc_id != doc_id:
        raise HallucinationError(f"{path}: holds doc_id {record.doc_id!r}, not {doc_id!r}")
    return record


def _read_text(value: Any, status: str) -> str:
    if status == "extracted":
        return str(value)
    return f"({status})"


def _member(doc_id: str, record: DocRecord | None, resolved: ResolvedLabel) -> Member:
    if record is None or record.status != "ok" or record.validated is None:
        return Member(doc_id, 0, 0)
    labeled = correct = 0
    for name in SCALAR_FIELD_NAMES:
        expected = getattr(resolved.label, name)
        if expected is None or is_unlabeled(expected):
            continue
        labeled += 1
        score = score_field(name, expected, getattr(record.validated, name))
        correct += score.outcome == "correct"
    return Member(doc_id, labeled, correct)


def _lineage(
    rows: list[ManifestRow],
    labels: dict[str, ResolvedLabel],
    records: dict[str, DocRecord],
) -> tuple[LineageGroup, ...]:
    members_of: dict[str, list[str]] = {}
    for row in rows:
        if row.gt_from and row.doc_id in labels and row.gt_from in labels:
            members_of.setdefault(row.gt_from, []).append(row.doc_id)
    groups: list[LineageGroup] = []
    for owner_id in sorted(members_of):
        if owner_id not in records:
            continue  # the owner was not part of this run: nothing to compare against
        owner = _member(owner_id, records.get(owner_id), labels[owner_id])
        members = []
        for doc_id in sorted(members_of[owner_id]):
            if doc_id not in records:
                continue
            member = _member(doc_id, records.get(doc_id), labels[doc_id])
            members.append(
                Member(member.doc_id, member.labeled, member.correct, is_suspect(owner, member))
            )
        reads: dict[str, tuple[str, ...]] = {}
        for name in IDENTIFIER_FIELDS:
            seen: list[str] = []
            for doc_id in [owner_id, *(m.doc_id for m in members)]:
                record = records.get(doc_id)
                if record is None or record.status != "ok" or record.validated is None:
                    continue
                field = getattr(record.validated, name)
                seen.append(_read_text(field.value, field.status))
            reads[name] = tuple(seen)
        groups.append(LineageGroup(owner, tuple(members), reads))
    return tuple(groups)


def collect(run_dir: Path, *, manifest_path: Path, labels_dir: Path) -> HallucinationReport:
    """Every hallucinated value of the run in `run_dir`, with denominators and the lineage audit."""
    run_path = run_dir / "run.json"
    try:
        run = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HallucinationError(f"{run_path}: cannot be read ({exc})") from exc
    except ValidationError as exc:
        raise HallucinationError(f"{run_path}: not a run record ({exc})") from exc
    try:
        manifest = read_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        raise HallucinationError(f"{manifest_path}: cannot be read as a manifest ({exc})") from exc
    labels, _ = load_labels(manifest, labels_dir)
    by_id = {row.doc_id: row for row in manifest}

    records: dict[str, DocRecord] = {
        doc_id: _read_record(run_dir, doc_id) for doc_id in sorted(run.statuses) if doc_id in labels
    }
    items: list[Hallucination] = []
    absent_fields = null_cells = documents = 0
    for doc_id, record in records.items():
        if record.status != "ok" or record.validated is None or record.parsed is None:
            continue
        documents += 1
        resolved = labels[doc_id]
        row = by_id[doc_id]
        context = {
            "doc_id": doc_id,
            "owner": resolved.owner,
            "capture": row.capture or "",
            "doc_type": row.doc_type or "",
            "label_mode": resolved.label.label_mode or "",
        }
        for name in SCALAR_FIELD_NAMES:
            expected = getattr(resolved.label, name)
            if is_unlabeled(expected) or expected is not None:
                continue
            absent_fields += 1
            field = getattr(record.validated, name)
            if score_field(name, None, field).outcome != "should_be_absent":
                continue
            raw = getattr(record.parsed, name)
            items.append(
                Hallucination(
                    level="field",
                    name=name,
                    row=None,
                    value=str(field.value),
                    confidence=field.confidence,
                    page=field.page,
                    codes=_codes(field.reasons),
                    note=_note(field.reasons),
                    quote=field.evidence or "",
                    raw_also=raw.status == "extracted",
                    **context,
                )
            )
        expected_rows = resolved.label.line_items
        if is_unlabeled(expected_rows) or expected_rows is None:
            continue
        predicted = record.validated.line_items
        for pair in match_rows(predicted, expected_rows).pairs:
            labeled_row = expected_rows[pair.expected]
            returned = predicted[pair.predicted]
            raw_rows = record.parsed.line_items
            for cell in CELL_NAMES:
                if cell not in pair.cells or getattr(labeled_row, cell) is not None:
                    continue
                null_cells += 1
                value = getattr(returned, cell)
                if value is None:
                    continue
                raw_also = (
                    pair.predicted < len(raw_rows)
                    and getattr(raw_rows[pair.predicted], cell) is not None
                )
                items.append(
                    Hallucination(
                        level="cell",
                        name=cell,
                        row=pair.predicted,
                        value=str(value),
                        confidence=returned.confidence,
                        page=returned.page,
                        codes=_codes(returned.reasons),
                        note=_note(returned.reasons),
                        quote=returned.evidence or "",
                        raw_also=raw_also,
                        **context,
                    )
                )
    return HallucinationReport(
        run_id=run.run_id,
        model=run.model,
        prompt_sha256=run.prompt_sha256,
        documents=documents,
        items=tuple(items),
        absent_fields=absent_fields,
        null_cells=null_cells,
        lineage=_lineage(manifest, labels, records),
    )


# ---------------------------------------------------------------- tables


def _percent(part: int, whole: int) -> str:
    return "n/a" if whole == 0 else f"{100 * part / whole:.1f}%"


def band_of(confidence: float) -> str:
    for name, low, high in CONFIDENCE_BANDS:
        if low <= confidence < high:
            return name
    return CONFIDENCE_BANDS[-1][0]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _counts(items: tuple[Hallucination, ...], key: str) -> list[tuple[str, int]]:
    tally = Counter(getattr(item, key) for item in items)
    return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def to_markdown(report: HallucinationReport, *, notes: str | None = None) -> str:
    items = report.items
    fields, cells = report.fields, report.cells
    lines = [
        f"# Hallucinated values: `{report.run_id}`",
        "",
        f"Model `{report.model}`, prompt `{report.prompt_sha256[:12]}`, "
        f"{report.documents} documents scored. Generated by "
        "`scripts/analyze_hallucinations.py`; the tables are data, the reading at the end is "
        "hand-written. `hallucinated_value` = a value returned where the label says the field "
        "is absent.",
        "",
        "## How many",
        "",
        *_table(
            ["level", "hallucinated", "label says absent", "rate"],
            [
                [
                    "scalar fields",
                    str(len(fields)),
                    str(report.absent_fields),
                    _percent(len(fields), report.absent_fields),
                ],
                [
                    "row cells (matched rows)",
                    str(len(cells)),
                    str(report.null_cells),
                    _percent(len(cells), report.null_cells),
                ],
                ["**all**", f"**{len(items)}**", "", ""],
            ],
        ),
        "",
        "The scalar denominator is every labeled field whose label is null; the cell "
        "denominator is every compared cell of a matched row pair whose label cell is null. The "
        "count equals the scorecard's `hallucinated_value`.",
        "",
    ]
    if not items:
        lines += ["No value was returned where the label says absent.", ""]
    else:
        lines += ["## By field or cell", ""]
        lines += _table(
            ["level", "name", "count"],
            [
                [level, name, str(count)]
                for (level, name), count in sorted(
                    Counter((i.level, i.name) for i in items).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ],
        )
        lines += ["", "## By label owner", ""]
        lines += [
            "Documents that share a label (`gt_from`) share its mistakes: the owner column "
            "groups them, so the count of independent labels behind the errors is visible.",
            "",
        ]
        owner_docs: dict[str, set[str]] = {}
        for item in items:
            owner_docs.setdefault(item.owner, set()).add(item.doc_id)
        lines += _table(
            ["label owner", "documents", "count", "share"],
            [
                [
                    owner,
                    ", ".join(sorted(owner_docs[owner])),
                    str(count),
                    _percent(count, len(items)),
                ]
                for owner, count in _counts(items, "owner")
            ],
        )
        lines += ["", "## By document", ""]
        lines += _table(
            ["document", "scalar", "cell"],
            [
                [
                    doc_id,
                    str(sum(1 for i in fields if i.doc_id == doc_id)),
                    str(sum(1 for i in cells if i.doc_id == doc_id)),
                ]
                for doc_id, _ in _counts(items, "doc_id")
            ],
        )
        lines += ["", "## By capture and document type", ""]
        lines += _table(
            ["capture", "count"], [[name, str(count)] for name, count in _counts(items, "capture")]
        )
        lines += [""]
        lines += _table(
            ["doc type", "count"],
            [[name, str(count)] for name, count in _counts(items, "doc_type")],
        )
        lines += ["", "## Confidence the model had", ""]
        lines += [
            "The returned (capped) confidence of the field or row. A hallucination the model is "
            "sure of is one the abstention policy cannot see.",
            "",
        ]
        lines += _table(
            ["band", "scalar", "cell"],
            [
                [
                    name,
                    str(sum(1 for i in fields if band_of(i.confidence) == name)),
                    str(sum(1 for i in cells if band_of(i.confidence) == name)),
                ]
                for name, _, _ in CONFIDENCE_BANDS
            ],
        )
        raw_also = sum(1 for i in items if i.raw_also)
        coded = sum(1 for i in items if i.codes)
        lines += [
            "",
            f"The model's own answer (before the validators) had a value in {raw_also} of "
            f"{len(items)} cases (a validator never turns an absent field into a value, so this "
            f"should be all of them). {coded} of {len(items)} carry a validator code "
            f"({', '.join(sorted({c for i in items for c in i.codes})) or 'none'}): the guards "
            "noticed something and the value was still kept.",
            "",
        ]
    lines += ["## Do the copies of a document read alike? (lineage audit)", ""]
    if not report.lineage:
        lines += ["No document borrows a label (`gt_from`) in this run.", ""]
    else:
        lines += [
            "A document with `gt_from` is scored against another document's label. Copies of "
            "one page should read alike. Rule, fixed in code: a member is **suspect** when it "
            f"has at least {LINEAGE_MIN_FIELDS} labeled non-null scalar fields, so does its "
            f"owner, and its exact-match rate is at least {float(LINEAGE_GAP):.2f} below the "
            "owner's. This only flags: whether the label or the model is wrong takes a person "
            "looking at the pages.",
            "",
        ]
        rows = []
        for group in report.lineage:
            for member in (group.owner, *group.members):
                rate = member.rate
                rows.append(
                    [
                        group.owner.doc_id,
                        member.doc_id + (" (owner)" if member is group.owner else ""),
                        f"{member.correct}/{member.labeled}",
                        "n/a" if rate is None else f"{float(rate):.2f}",
                        "**suspect**" if member.suspect else "",
                    ]
                )
        lines += _table(["label", "document", "correct / labeled non-null", "rate", "audit"], rows)
        lines += ["", "What the run read in each group (distinct values over owner + members):", ""]
        for group in report.lineage:
            ids = ", ".join([group.owner.doc_id, *(m.doc_id for m in group.members)])
            lines += [f"- **{group.owner.doc_id}** ({ids}):"]
            for name, seen in group.reads.items():
                distinct = sorted(set(seen))
                lines += [
                    f"  - `{name}`: {len(distinct)} distinct of {len(seen)} reads: "
                    + ", ".join(f"`{_short(v, 40)}`" for v in distinct)
                ]
        lines += [""]
    if items:
        lines += ["## Every case", ""]
        lines += _table(
            ["document", "level", "name", "row", "value", "conf", "codes", "model's note"],
            [
                [
                    i.doc_id,
                    i.level,
                    i.name,
                    "" if i.row is None else str(i.row),
                    _short(i.value, 30),
                    f"{i.confidence:.2f}",
                    ", ".join(i.codes),
                    _short(i.note),
                ]
                for i in sorted(items, key=lambda i: (i.doc_id, i.level, i.row or 0, i.name))
            ],
        )
        lines += [""]
    if notes:
        lines += ["## Reading", "", notes.strip(), ""]
    return "\n".join(lines)


def review_markdown(report: HallucinationReport) -> str:
    """The list for a human: what the label says, what the model said, and its quote. Evidence
    quotes come from (redacted) real pages, so this belongs outside the tracked folders."""
    lines = [
        f"# Review list: hallucinated values of `{report.run_id}`",
        "",
        "For each case, look at the page and write one of: INVENTED (the model made it up), "
        "PRINTED (the value is on the page, so the label should not say absent), PRINTED "
        "ELSEWHERE (it is on the page but is not this field), POLICY (a row-definition "
        "question: fee, voucher, discount or free item counted as a line item).",
        "",
    ]
    suspects = [m for g in report.lineage for m in g.suspects]
    if suspects:
        lines += [
            "## First, the lineage audit",
            "",
            "These documents borrow a label (`gt_from`) but the run reads them very "
            "differently from the label's owner. Check whether each one really is a copy of "
            "the owner's page:",
            "",
        ]
        for group in report.lineage:
            for member in group.suspects:
                lines += [
                    f"- {member.doc_id} is scored against {group.owner.doc_id}'s label: "
                    "SAME PAGE / DIFFERENT DOCUMENT?"
                ]
        lines += [""]
    for item in sorted(report.items, key=lambda i: (i.doc_id, i.level, i.row or 0, i.name)):
        where = f"row {item.row}, cell {item.name}" if item.level == "cell" else item.name
        lines += [
            f"### {item.doc_id}: {where}",
            "",
            f"- label ({item.owner}, {item.label_mode or 'no label_mode'}): absent",
            f"- model: `{item.value}` (confidence {item.confidence:.2f}, page {item.page})",
            f"- the model's note: {item.note or '(none)'}",
            f"- its quote: {item.quote or '(none)'}",
            "- verdict: ",
            "",
        ]
    return "\n".join(lines)
