"""Builders for synthetic raw files: PDFs (with or without a text layer, any page size), JPEGs with
EXIF, PNGs with metadata, HEIC. Used by the manifest, probe and ingestion tests so that the real
code paths run without ever touching real documents."""

import io
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, PngImagePlugin
from reportlab.pdfgen import canvas

from docextract.fsutil import long_path

A4 = (595.276, 841.89)  # points


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_text_pdf(path: Path, text: str, pages: int = 1) -> Path:
    _ensure_parent(path)
    pdf = canvas.Canvas(str(path))
    for _ in range(pages):
        words = text.split()
        y = 800
        for start in range(0, len(words), 8):
            pdf.drawString(40, y, " ".join(words[start : start + 8]))
            y -= 16
        pdf.showPage()
    pdf.save()
    return path


def make_image_pdf(path: Path, pages: int = 2) -> Path:
    _ensure_parent(path)
    images = [Image.new("RGB", (120, 160), (240 - 20 * i,) * 3) for i in range(pages)]
    images[0].save(path, "PDF", save_all=True, append_images=images[1:])
    return path


def make_sized_pdf(path: Path, sizes: Sequence[tuple[float, float]]) -> Path:
    """A PDF with one page per (width, height) in points, each with a black bar near the top
    so that a rendered page is not blank. reportlab output is made deterministic."""
    _ensure_parent(path)
    pdf = canvas.Canvas(str(path), invariant=1, pageCompression=0)
    for width, height in sizes:
        pdf.setPageSize((width, height))
        pdf.rect(20, height - 60, width - 40, 30, fill=1)
        pdf.showPage()
    pdf.save()
    return path


def make_jpeg(
    path: Path,
    *,
    make: str | None = None,
    model: str | None = None,
    orientation: int | None = None,
    gps: bool = False,
    comment: str | None = None,
    size: tuple[int, int] = (60, 40),
    color: tuple[int, int, int] | str = "white",
    quadrants: bool = False,
) -> Path:
    """A JPEG with the requested EXIF. `quadrants` paints the top-left quarter red so that a
    rotation can be seen in the pixels."""
    _ensure_parent(path)
    exif = Image.Exif()
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    if orientation:
        exif[0x0112] = orientation
    if gps:
        ifd = exif.get_ifd(0x8825)
        ifd[1] = "S"
        ifd[2] = (6.0, 12.0, 30.0)
    image = Image.new("RGB", size, color)
    if quadrants:
        image.paste((255, 0, 0), (0, 0, size[0] // 2, size[1] // 2))
    extra = {"comment": comment.encode()} if comment else {}
    image.save(path, "JPEG", exif=exif.tobytes(), quality=95, **extra)
    return path


def make_png(
    path: Path, *, size: tuple[int, int] = (30, 20), color: tuple[int, int, int] | str = "white"
) -> Path:
    _ensure_parent(path)
    Image.new("RGB", size, color).save(path, "PNG")
    return path


def make_png_with_metadata(path: Path) -> Path:
    """A PNG carrying a text chunk, a DPI and an ICC-looking profile that must not survive."""
    _ensure_parent(path)
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "secret-comment")
    info.add_text("Author", "someone")
    Image.new("RGB", (40, 30), (120, 120, 120)).save(path, "PNG", pnginfo=info, dpi=(300, 300))
    return path


def make_rgba_png(path: Path) -> Path:
    _ensure_parent(path)
    Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(path, "PNG")
    return path


def make_heic(path: Path, *, size: tuple[int, int] = (64, 48)) -> Path:
    """A real HEIC (needs pillow-heif); the caller skips the test if that is unavailable."""
    import pillow_heif

    pillow_heif.register_heif_opener()
    _ensure_parent(path)
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format="HEIF")
    path.write_bytes(buffer.getvalue())
    return path


def write_long_named(directory: Path, name: str, content: bytes) -> Path:
    """Write a file whose full path may exceed the classic Windows limit."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    long_path(target).write_bytes(content)
    return target
