"""Manifest row -> stored pages, idempotently.

    raw files -> pages (EXIF upright, PDF at INGEST_DPI, size cap) -> redaction -> encode ->
    data/pages/<doc_id>/p<N>.png|jpg  +  ingest.json

The work is skipped when nothing changed: the sidecar `ingest.json` records a hash of every input
that affects the output (the raw bytes, dpi, cap, quality, page_range, this document's redaction
boxes and INGEST_VERSION) and the sha256 of each stored page. A run is a no-op only if the inputs
hash matches AND every recorded page file is still there and unmodified. The sidecar is written
last, so an interrupted run is simply redone. Outputs are deterministic: the same inputs give
byte-identical files.

The sidecar holds sizes, hashes and page mapping but NO raw file names (they can contain names).
"""

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from docextract.config import IngestSettings
from docextract.fsutil import long_path
from docextract.imaging import encode_page, storage_format
from docextract.ingest.manifest import ManifestRow
from docextract.ingest.pages import PageLoadError, load_pages
from docextract.ingest.redact import Box, RedactionError, apply_boxes, validate_boxes

INGEST_VERSION = 1  # bump when the pipeline's output would change for the same inputs
SIDECAR = "ingest.json"
_PAGE_FILE = re.compile(r"p\d+\.(png|jpg)")
_EXTENSION = {"png": "png", "jpeg": "jpg"}


class IngestError(Exception):
    def __init__(self, doc_id: str, message: str) -> None:
        super().__init__(f"{doc_id}: {message}")
        self.doc_id = doc_id
        self.message = message


@dataclass(frozen=True)
class PageRecord:
    n: int  # 1-based page number within the document
    file: str  # p<N>.png or p<N>.jpg
    format: str  # "png" or "jpeg"
    size: tuple[int, int]  # (width, height) of the stored page
    sha256: str
    bytes: int
    source_file: int  # index into the manifest row's raw_files
    source_page: int  # page within that raw file

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "file": self.file,
            "format": self.format,
            "size": list(self.size),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "source": {"file": self.source_file, "page": self.source_page},
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "PageRecord":
        source = raw["source"]
        return cls(
            n=int(raw["n"]),
            file=str(raw["file"]),
            format=str(raw["format"]),
            size=(int(raw["size"][0]), int(raw["size"][1])),
            sha256=str(raw["sha256"]),
            bytes=int(raw["bytes"]),
            source_file=int(source["file"]),
            source_page=int(source["page"]),
        )


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    status: Literal["ingested", "unchanged", "would_ingest"]
    pages: tuple[PageRecord, ...]
    inputs_hash: str


def check_redaction_decision(row: ManifestRow, boxes: Mapping[int, Sequence[Any]] | None) -> None:
    """Fail loudly unless the redaction decision and the boxes agree."""
    has_boxes = boxes is not None and any(len(page_boxes) for page_boxes in boxes.values())
    if row.needs_redaction is None:
        raise RedactionError(
            f"{row.doc_id}: needs_redaction is blank in the manifest; "
            "decide y or n before ingesting"
        )
    if row.needs_redaction and not has_boxes:
        raise RedactionError(
            f"{row.doc_id}: needs_redaction=y but there are no boxes in redactions.yaml: "
            "refusing to store an unredacted page"
        )
    if not row.needs_redaction and has_boxes:
        raise RedactionError(
            f"{row.doc_id}: redactions.yaml has boxes but the manifest says needs_redaction=n "
            "(set it to y, or remove the boxes)"
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_inputs_hash(
    raw_hashes: Sequence[str],
    settings: IngestSettings,
    page_range: str | None,
    boxes: Mapping[int, Sequence[Sequence[Any]]],
) -> str:
    def canonical(value: Any) -> Any:  # 10.0 and 10 are the same pixel
        return int(value) if isinstance(value, float) and value.is_integer() else value

    recipe = {
        "version": INGEST_VERSION,
        "dpi": settings.dpi,
        "max_long_edge": settings.max_long_edge,
        "jpeg_quality": settings.jpeg_quality,
        "page_range": page_range,
        "raw": list(raw_hashes),
        "boxes": {
            str(page): [[canonical(v) for v in box] for box in found]
            for page, found in sorted(boxes.items())
        },
    }
    return _sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())


def read_sidecar(out_dir: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads((out_dir / SIDECAR).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _current_pages(
    out_dir: Path, sidecar: dict[str, Any] | None, inputs_hash: str
) -> tuple[PageRecord, ...] | None:
    """The recorded pages, if the sidecar matches `inputs_hash` and every file is intact."""
    if not sidecar or sidecar.get("ingest_version") != INGEST_VERSION:
        return None
    if sidecar.get("inputs_hash") != inputs_hash:
        return None
    try:
        records = tuple(PageRecord.from_json(entry) for entry in sidecar["pages"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    for record in records:
        try:
            if _sha256((out_dir / record.file).read_bytes()) != record.sha256:
                return None
        except OSError:
            return None
    return records


def _write_atomic(path: Path, data: bytes) -> None:
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def ingest_document(
    row: ManifestRow,
    *,
    raw_dir: Path,
    out_dir: Path,
    settings: IngestSettings,
    boxes: Mapping[int, Sequence[Sequence[Any]]] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> IngestResult:
    """Ingest one document. `boxes` maps a 1-based stored page number to redaction boxes."""
    doc_id = row.doc_id
    if row.is_derived:
        raise IngestError(
            doc_id, "a degraded variant is generated from its source's pages, not ingested"
        )
    try:
        check_redaction_decision(row, boxes)
    except RedactionError as exc:
        raise IngestError(doc_id, str(exc).removeprefix(f"{doc_id}: ")) from exc

    files = [raw_dir / name for name in row.raw_files]
    missing = [
        name
        for name, path in zip(row.raw_files, files, strict=True)
        if not long_path(path).is_file()
    ]
    if missing:
        raise IngestError(doc_id, f"raw file not found: {missing[0]}")
    raw_hashes = [_sha256(long_path(path).read_bytes()) for path in files]

    wanted_boxes: dict[int, list[Box]] = {}
    inputs_hash = compute_inputs_hash(
        raw_hashes,
        settings,
        row.page_range,
        boxes or {},
    )
    sidecar = read_sidecar(out_dir)
    if not force:
        recorded = _current_pages(out_dir, sidecar, inputs_hash)
        if recorded is not None:
            return IngestResult(doc_id, "unchanged", recorded, inputs_hash)
    if dry_run:
        return IngestResult(doc_id, "would_ingest", (), inputs_hash)

    try:
        rendered = load_pages(
            files,
            page_range=row.page_range,
            dpi=settings.dpi,
            max_long_edge=settings.max_long_edge or None,
        )
    except PageLoadError as exc:
        raise IngestError(doc_id, str(exc)) from exc
    if not rendered:
        raise IngestError(doc_id, "the selection contains no pages")

    for page in sorted(boxes or {}):
        if not 1 <= page <= len(rendered):
            raise IngestError(
                doc_id,
                f"redactions.yaml has boxes for page {page}, "
                f"but the document has {len(rendered)} page(s)",
            )
    for number, page in enumerate(rendered, start=1):
        try:
            wanted_boxes[number] = validate_boxes(
                (boxes or {}).get(number, []), page.image.size, where=f"{doc_id} page {number}"
            )
        except RedactionError as exc:
            raise IngestError(doc_id, str(exc)) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[PageRecord] = []
    for number, page in enumerate(rendered, start=1):
        fmt = storage_format(page.source_format)
        redacted = apply_boxes(page.image, wanted_boxes[number])  # BEFORE encoding
        data = encode_page(redacted, fmt, jpeg_quality=settings.jpeg_quality)
        name = f"p{number}.{_EXTENSION[fmt]}"
        _write_atomic(out_dir / name, data)
        records.append(
            PageRecord(
                n=number,
                file=name,
                format=fmt,
                size=redacted.size,
                sha256=_sha256(data),
                bytes=len(data),
                source_file=page.source_file,
                source_page=page.source_page,
            )
        )

    keep = {record.file for record in records}
    for stale in out_dir.iterdir():
        if _PAGE_FILE.fullmatch(stale.name) and stale.name not in keep:
            stale.unlink()  # a shorter document, or a page that changed format
    sidecar_data = {
        "ingest_version": INGEST_VERSION,
        "inputs_hash": inputs_hash,
        "settings": {
            "dpi": settings.dpi,
            "max_long_edge": settings.max_long_edge,
            "jpeg_quality": settings.jpeg_quality,
        },
        "page_range": row.page_range,
        "raw_sha256": raw_hashes,
        "redactions": {
            str(n): [list(b) for b in found] for n, found in wanted_boxes.items() if found
        },
        "pages": [record.to_json() for record in records],
    }
    _write_atomic(
        out_dir / SIDECAR, (json.dumps(sidecar_data, indent=2, sort_keys=True) + "\n").encode()
    )
    return IngestResult(doc_id, "ingested", tuple(records), inputs_hash)
