"""Unredacted previews: the pages exactly as ingestion would store them, minus the black boxes.

Redaction boxes are pixel coordinates on the STORED page, so you have to look at that page to
measure them. But a document with needs_redaction=y and no boxes yet is refused, so its stored page
cannot be made. A preview is the way out: the same loading, EXIF rotation, rendering and size cap,
so a coordinate read off a preview is the coordinate on the stored page, written to
data/private/preview/ (gitignored) and never to the pages directory.

These files are the unredacted originals in another encoding. They never go anywhere that is
committed, which is why the preview directory may not be (or contain, or sit inside) the pages
directory, and why a preview carries no ingest.json.
"""

import os
from pathlib import Path

from docextract.config import IngestSettings
from docextract.ingest.manifest import ManifestRow
from docextract.ingest.pages import PageLoadError, load_pages
from docextract.ingest.pipeline import (
    SIDECAR,
    IngestError,
    PageRecord,
    raw_paths,
    store_pages,
)


class PreviewError(Exception):
    """The preview would land somewhere it must not."""


def _normalized(path: Path) -> Path:
    return Path(os.path.normcase(str(path.resolve())))


def check_preview_dir(preview_dir: Path, pages_dir: Path) -> None:
    """Refuse a preview directory that is, contains or sits inside the stored-pages directory."""
    preview, pages = _normalized(preview_dir), _normalized(pages_dir)
    if preview == pages or preview.is_relative_to(pages) or pages.is_relative_to(preview):
        raise PreviewError(
            f"the preview directory {preview_dir} overlaps the stored-pages directory "
            f"{pages_dir}: unredacted pages must not be written next to the committed ones"
        )


def preview_document(
    row: ManifestRow,
    *,
    raw_dir: Path,
    preview_dir: Path,
    pages_dir: Path,
    settings: IngestSettings,
) -> tuple[PageRecord, ...]:
    """Write the document's UNREDACTED normalized pages to `preview_dir/<doc_id>/`."""
    check_preview_dir(preview_dir, pages_dir)
    if row.is_derived:
        raise IngestError(row.doc_id, "a degraded variant has no raw pages to preview")
    out_dir = preview_dir / row.doc_id
    if (out_dir / SIDECAR).exists():
        raise PreviewError(
            f"{out_dir} holds an {SIDECAR}, so it is a stored-pages directory, not a preview"
        )
    files = raw_paths(row, raw_dir)
    try:
        rendered = load_pages(
            files,
            page_range=row.page_range,
            dpi=settings.dpi,
            max_long_edge=settings.max_long_edge or None,
        )
    except PageLoadError as exc:
        raise IngestError(row.doc_id, str(exc)) from exc
    if not rendered:
        raise IngestError(row.doc_id, "the selection contains no pages")
    return tuple(store_pages(out_dir, rendered, {}, settings))
