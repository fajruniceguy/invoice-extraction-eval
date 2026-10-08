"""Turn uploaded files into the page images the extractor sends.

An upload goes through the SAME steps as ingestion (`load_pages` then `store_pages`, with the same
`IngestSettings`) minus redaction, so the pages are byte-identical to what `scripts/ingest.py` would
store for the same file. That is what makes a replay from the response cache cost nothing: the cache
key hashes the page images.

Uploads are not trusted:
  * only bytes come in, never a file name: the files are written as f0.bin, f1.bin... inside the
    caller's work directory, so a name like "../../x" has nothing to act on;
  * a PDF is page-counted BEFORE it is rendered, so a 500-page upload costs one header read;
  * size is checked first; Pillow's decompression-bomb guard is left on.

Everything is synchronous and CPU-bound: the API runs it in a thread.
"""

from collections.abc import Sequence
from pathlib import Path

import pypdfium2 as pdfium

from docextract.config import IngestSettings
from docextract.ingest.manifest import parse_page_range
from docextract.ingest.pages import PageLoadError, load_pages
from docextract.ingest.pipeline import store_pages
from docextract.ingest.probe import sniff_format


class UploadError(Exception):
    """An upload that cannot be turned into pages. `code` is the API's error code."""

    code = "bad_upload"


class UnreadableUpload(UploadError):
    code = "unreadable_file"


class FileTooLarge(UploadError):
    code = "file_too_large"


class TooManyPages(UploadError):
    code = "too_many_pages"


def _page_total(files: Sequence[bytes], page_range: str | None) -> int:
    """How many pages these files would render, without rendering any."""
    total = 0
    for index, data in enumerate(files, start=1):
        if sniff_format(data[:16]) != "pdf":
            total += 1  # an image is one page (page_range on an image fails when it is loaded)
            continue
        try:
            document = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as exc:
            raise UnreadableUpload(f"file {index}: cannot open the PDF: {exc}") from exc
        try:
            count = len(document)
        finally:
            document.close()
        try:
            total += len(parse_page_range(page_range, count))
        except ValueError as exc:
            raise UnreadableUpload(f"file {index}: {exc}") from exc
    return total


def pages_from_uploads(
    files: Sequence[bytes],
    workdir: Path,
    *,
    max_pages: int,
    page_range: str | None = None,
    max_bytes: int | None = None,
    settings: IngestSettings | None = None,
) -> list[Path]:
    """The page files of ONE document made of `files` (in order), written under `workdir`."""
    if not files:
        raise UnreadableUpload("no files were uploaded")
    if page_range is not None and len(files) != 1:
        raise UnreadableUpload("page_range needs exactly one file (it slices a PDF)")
    for index, data in enumerate(files, start=1):
        if max_bytes is not None and len(data) > max_bytes:
            raise FileTooLarge(f"file {index} is {len(data)} bytes; the limit is {max_bytes}")
    if (pages := _page_total(files, page_range)) > max_pages:
        raise TooManyPages(f"{pages} pages, but at most {max_pages} per document")

    config = settings or IngestSettings()
    raw = workdir / "in"
    raw.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, data in enumerate(files):
        path = raw / f"f{index}.bin"
        path.write_bytes(data)
        paths.append(path)
    try:
        rendered = load_pages(
            paths, page_range=page_range, dpi=config.dpi, max_long_edge=config.max_long_edge
        )
    except PageLoadError as exc:
        raise UnreadableUpload(str(exc)) from exc
    out = workdir / "pages"
    records = store_pages(out, rendered, {}, config)
    return [out / record.file for record in records]
