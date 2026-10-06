"""Raw file -> normalized page images (in memory).

Normalized means: EXIF orientation applied, PDF pages rasterized at the configured DPI, and the
size rule of `docextract.imaging` applied. Metadata is dropped later, by `encode_page`.
Redaction happens after this and before encoding.
"""

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps

from docextract.fsutil import long_path
from docextract.imaging import flatten, resize_to, target_size
from docextract.ingest.manifest import parse_page_range
from docextract.ingest.probe import register_heif, sniff_format


class PageLoadError(Exception):
    """A raw file could not be turned into pages."""


@dataclass(frozen=True)
class RenderedPage:
    image: Image.Image
    source_format: str  # "PDF" or the Pillow format name ("JPEG", "MPO", "HEIF", "PNG"...)
    source_file: int  # index into the document's raw_files
    source_page: int  # 1-based page within that file (always 1 for an image)


def _render_pdf(
    data: bytes, file_index: int, *, page_range: str | None, dpi: int, max_long_edge: int | None
) -> list[RenderedPage]:
    try:
        document = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise PageLoadError(f"cannot open the PDF: {exc}") from exc
    try:
        try:
            selected = parse_page_range(page_range, len(document))
        except ValueError as exc:
            raise PageLoadError(str(exc)) from exc
        pages: list[RenderedPage] = []
        for number in selected:
            page = document[number - 1]
            try:
                width_pt, height_pt = page.get_size()
                scale = dpi / 72
                wanted = target_size(
                    max(1, round(width_pt * scale)), max(1, round(height_pt * scale)), max_long_edge
                )
                # render straight at the final size: sharper than render-then-shrink, and a
                # poster-sized page never has to exist at full resolution in memory
                bitmap = page.render(scale=wanted[0] / width_pt)
                image = resize_to(bitmap.to_pil().convert("RGB"), wanted)
            finally:
                page.close()
            pages.append(RenderedPage(image, "PDF", file_index, number))
        return pages
    finally:
        document.close()


def _load_image(data: bytes, file_index: int, *, max_long_edge: int | None) -> RenderedPage:
    register_heif()
    try:
        with Image.open(io.BytesIO(data)) as opened:
            source_format = opened.format or ""
            opened.seek(0)  # multi-frame files (GIF, TIFF, MPO): the first frame only
            image = ImageOps.exif_transpose(opened)  # upright; returns a new image
            image.load()
            # flatten BEFORE resampling: transparency must become white, not black, and
            # resizing an RGBA image would otherwise bleed the transparent colour into the edges
            image = flatten(image)
    except Exception as exc:
        hint = " (is pillow-heif installed?)" if sniff_format(data[:16]) == "heic" else ""
        raise PageLoadError(f"cannot open the image: {type(exc).__name__}: {exc}{hint}") from exc
    image = resize_to(image, target_size(image.width, image.height, max_long_edge))
    return RenderedPage(image, source_format, file_index, 1)


def load_pages(
    files: Sequence[Path],
    *,
    page_range: str | None,
    dpi: int,
    max_long_edge: int | None,
) -> list[RenderedPage]:
    """Every page of every file, in order. `page_range` slices a (single) PDF."""
    pages: list[RenderedPage] = []
    for index, path in enumerate(files):
        data = long_path(path).read_bytes()
        if sniff_format(data[:16]) == "pdf":
            pages.extend(
                _render_pdf(
                    data, index, page_range=page_range, dpi=dpi, max_long_edge=max_long_edge
                )
            )
        else:
            if page_range is not None:
                raise PageLoadError("page_range applies to PDFs only, but this file is an image")
            pages.append(_load_image(data, index, max_long_edge=max_long_edge))
    return pages
