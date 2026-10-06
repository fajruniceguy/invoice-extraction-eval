"""The dataset manifest: one row per LOGICAL document, in data/manifest.csv.

A logical document is what gets one ground truth: one receipt, one faktur, one hotel booking.
It may be one file, several files (a receipt photographed in two shots) or a page range of a
bigger PDF (a scanned bundle holding several fakturs becomes several rows over one file).

Rows are written by `scripts/manifest_init.py` with everything that can be inferred filled in;
the rest is filled by hand, so a row may legitimately be incomplete. Completeness is checked
per stage by `missing_fields` rather than enforced by the model.
"""

import csv
import io
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

DocType = Literal["receipt", "faktur", "invoice", "ecommerce", "bill", "hotel_reservation"]
Capture = Literal[
    "born_digital", "phone_photo", "screen_capture", "synthetic_degraded", "constructed"
]
Split = Literal["dev", "test"]
LabelMode = Literal["prelabel", "blind"]
Variant = Literal["light", "medium", "heavy"]

# the committed column order: the 14 columns of the brief, then the two optional extras
REQUIRED_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "raw_files",
    "page_range",
    "doc_type",
    "template_group",
    "content_group",
    "capture",
    "original_format",
    "language",
    "source_url",
    "license_note",
    "split",
    "label_mode",
    "needs_redaction",
)
OPTIONAL_COLUMNS: tuple[str, ...] = ("gt_from", "variant")
COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

_DOC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")
_PAGE_RANGE = re.compile(r"\d+(-\d+)?(,\d+(-\d+)?)*")
_TRUE = {"y", "yes", "true", "1"}
_FALSE = {"n", "no", "false", "0"}


class ManifestError(ValueError):
    """The manifest is unreadable or inconsistent. `problems` lists every issue found."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


class ManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    doc_id: str
    raw_files: tuple[str, ...] = ()  # relative to data/private/raw, '/' separators
    page_range: str | None = None  # 1-based inclusive: "3", "2-4", "1,3-4"
    doc_type: DocType | None = None
    template_group: str | None = None  # same layout/vendor template, for generalization analysis
    content_group: str | None = None  # same content = same GT; never split across dev/test
    capture: Capture | None = None
    original_format: str | None = None  # as the original files are: pdf, jpg, jfif, png, heic...
    language: str | None = None  # "id", "en", ... (';' separated when mixed)
    source_url: str | None = None
    license_note: str | None = None
    split: Split | None = None
    label_mode: LabelMode | None = None  # blank for docs that are never hand-labeled
    needs_redaction: bool | None = None  # blank = not decided yet, and ingestion refuses that
    gt_from: str | None = None  # take the ground truth of this doc (variants, duplicate photos)
    variant: Variant | None = None  # light|medium|heavy, for synthetic_degraded docs

    @field_validator("doc_id")
    @classmethod
    def _safe_doc_id(cls, value: str) -> str:
        if not _DOC_ID.fullmatch(value):
            raise ValueError(
                f"doc_id {value!r} must be letters, digits, '_', '-' or '.', starting with a "
                "letter or digit (it names a directory and a file)"
            )
        return value

    @field_validator("raw_files", mode="before")
    @classmethod
    def _split_raw_files(cls, value: Any) -> Any:
        if value is None or value == "":
            return ()
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(";") if part.strip())
        return value

    @field_validator("raw_files")
    @classmethod
    def _safe_raw_files(cls, files: tuple[str, ...]) -> tuple[str, ...]:
        for name in files:
            if "\\" in name:
                raise ValueError(f"raw file {name!r} must use '/' separators")
            parts = name.split("/")
            if (
                name.startswith("/")
                or re.match(r"^[A-Za-z]:", name)
                or ".." in parts
                or "" in parts
            ):
                raise ValueError(f"raw file {name!r} must be a relative path inside the raw folder")
        return files

    @field_validator("page_range")
    @classmethod
    def _page_range_grammar(cls, value: str | None) -> str | None:
        if value is None:
            return None
        compact = value.replace(" ", "")
        if not _PAGE_RANGE.fullmatch(compact):
            raise ValueError(f"page_range {value!r} must look like '3', '2-4' or '1,3-4'")
        for chunk in compact.split(","):
            low, _, high = chunk.partition("-")
            if int(low) < 1 or (high and int(high) < int(low)):
                raise ValueError(f"page_range {value!r}: pages start at 1 and ranges must ascend")
        return compact

    @field_validator("needs_redaction", mode="before")
    @classmethod
    def _yes_no(cls, value: Any) -> Any:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _TRUE:
                return True
            if lowered in _FALSE:
                return False
            if lowered == "":
                return None
            raise ValueError(f"needs_redaction {value!r} must be y or n")
        return value

    @model_validator(mode="before")
    @classmethod
    def _blank_is_none(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                key: (None if isinstance(value, str) and value.strip() == "" else value)
                for key, value in data.items()
            }
        return data

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.page_range is not None and len(self.raw_files) != 1:
            raise ValueError("page_range needs exactly one raw file (it slices a PDF)")
        derived = self.capture == "synthetic_degraded" or self.variant is not None
        if derived:
            if self.capture is not None and self.capture != "synthetic_degraded":
                raise ValueError("a variant must have capture=synthetic_degraded")
            if self.variant is None or self.gt_from is None:
                raise ValueError("a synthetic_degraded doc needs both variant and gt_from")
            if self.raw_files:
                raise ValueError("a synthetic_degraded doc is generated from pages: no raw_files")
        elif not self.raw_files:
            raise ValueError("raw_files is required (only synthetic_degraded docs have none)")
        if self.gt_from == self.doc_id:
            raise ValueError("gt_from cannot point at the doc itself")
        return self

    @property
    def is_derived(self) -> bool:
        return self.variant is not None

    def to_cells(self) -> dict[str, str]:
        """The CSV representation: blanks for None, 'y'/'n' for the redaction flag."""
        cells: dict[str, str] = {}
        for column in COLUMNS:
            value = getattr(self, column)
            if column == "raw_files":
                cells[column] = ";".join(value)
            elif column == "needs_redaction":
                cells[column] = "" if value is None else ("y" if value else "n")
            else:
                cells[column] = "" if value is None else str(value)
        return cells


# ---------------------------------------------------------------- page ranges


def parse_page_range(spec: str | None, page_count: int) -> list[int]:
    """The 1-based pages a row selects, ascending and unique. None selects every page."""
    if spec is None:
        return list(range(1, page_count + 1))
    pages: list[int] = []
    for chunk in spec.replace(" ", "").split(","):
        low, _, high = chunk.partition("-")
        first, last = int(low), int(high) if high else int(low)
        pages.extend(range(first, last + 1))
    if len(set(pages)) != len(pages) or pages != sorted(pages):
        raise ValueError(f"page_range {spec!r} repeats or reorders pages")
    outside = [page for page in pages if page > page_count]
    if outside:
        raise ValueError(
            f"page_range {spec!r} asks for page {outside[0]} of a {page_count}-page file"
        )
    return pages


# ---------------------------------------------------------------- csv io


def _detect_delimiter(header_line: str) -> str:
    """Excel in some locales saves CSV with ';' instead of ','."""
    if header_line.startswith("doc_id,"):
        return ","
    if header_line.startswith("doc_id;"):
        return ";"
    raise ManifestError(
        [f"the header must start with 'doc_id' then ',' or ';', got {header_line[:40]!r}"]
    )


def parse_manifest(text: str) -> list[ManifestRow]:
    text = text.lstrip("\N{ZERO WIDTH NO-BREAK SPACE}")  # Excel's byte-order mark
    lines = text.splitlines()
    if not lines or not lines[0].strip():
        raise ManifestError(["the manifest is empty (no header row)"])
    delimiter = _detect_delimiter(lines[0].strip())
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    header = [name.strip() for name in (reader.fieldnames or [])]
    unknown = [name for name in header if name not in COLUMNS]
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    problems: list[str] = []
    if unknown:
        problems.append(f"unknown column(s): {', '.join(unknown)}")
    if missing:
        problems.append(f"missing column(s): {', '.join(missing)}")
    if problems:
        raise ManifestError(problems)

    rows: list[ManifestRow] = []
    for number, record in enumerate(reader, start=2):  # row 1 is the header
        cells = {(key or "").strip(): (value or "") for key, value in record.items() if key}
        if None in record or not any(cells.values()):
            if None in record:
                problems.append(f"row {number}: more cells than columns")
            continue  # a fully blank line
        try:
            rows.append(ManifestRow.model_validate(cells))
        except ValidationError as exc:
            label = cells.get("doc_id") or "?"
            for error in exc.errors():
                where = ".".join(str(part) for part in error["loc"])
                problems.append(
                    f"row {number} ({label}) {where}: {error['msg']}".replace("Value error, ", "")
                )
    problems.extend(check_manifest(rows))
    if problems:
        raise ManifestError(problems)
    return rows


def read_manifest(path: Path) -> list[ManifestRow]:
    return parse_manifest(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row.to_cells())
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(buffer.getvalue())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------- cross-row checks


def check_manifest(rows: Sequence[ManifestRow]) -> list[str]:
    """Problems that only show up across rows (never raises)."""
    problems: list[str] = []
    by_id = {row.doc_id: row for row in rows}
    for doc_id, count in Counter(row.doc_id for row in rows).items():
        if count > 1:
            problems.append(f"doc_id {doc_id!r} appears {count} times")
    for row in rows:
        if row.gt_from is None:
            continue
        target = by_id.get(row.gt_from)
        if target is None:
            problems.append(f"{row.doc_id}: gt_from {row.gt_from!r} is not in the manifest")
        elif target.gt_from is not None:
            problems.append(
                f"{row.doc_id}: gt_from {row.gt_from!r} itself takes its GT from "
                f"{target.gt_from!r} (point at the doc that owns the labels)"
            )
    problems.extend(content_group_split_conflicts(rows))
    return problems


def content_group_split_conflicts(rows: Sequence[ManifestRow]) -> list[str]:
    """A content_group must never straddle dev and test: the same content on both sides leaks."""
    splits: dict[str, set[str]] = {}
    for row in rows:
        if row.content_group and row.split:
            splits.setdefault(row.content_group, set()).add(row.split)
    return [
        f"content_group {group!r} straddles splits {sorted(found)}"
        for group, found in splits.items()
        if len(found) > 1
    ]


def missing_fields(row: ManifestRow, stage: Literal["ingest", "split", "label"]) -> list[str]:
    """What still has to be filled in by hand before `row` can go through `stage`."""
    needed: list[str] = []
    if stage == "ingest" and row.needs_redaction is None:
        needed.append("needs_redaction")
    if stage in ("split", "label"):
        for column in ("doc_type", "content_group"):
            if getattr(row, column) is None:
                needed.append(column)
    if stage == "label":
        for column in ("capture", "split"):
            if getattr(row, column) is None:
                needed.append(column)
        if not row.is_derived and row.gt_from is None and row.label_mode is None:
            needed.append("label_mode")
    return needed
