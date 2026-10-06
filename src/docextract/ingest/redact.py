"""Redaction: solid black rectangles burned into a page BEFORE it is encoded and stored.

Boxes are `[x, y, w, h]` in the pixel space of the STORED page: after EXIF rotation, PDF
rasterization and the size cap, i.e. exactly the pixels you see in data/pages (or in the
unredacted preview). A box covers the pixels `x <= px < x + w` and `y <= py < y + h`.

Boxes that do not fit the page are an error, never silently clipped: a clipped box usually
means the coordinates were taken from a different resolution, and the real target would then
be left uncovered.
"""

from collections.abc import Sequence
from typing import Any

from PIL import Image, ImageDraw

from docextract.imaging import flatten

Box = tuple[int, int, int, int]  # x, y, w, h


class RedactionError(ValueError):
    """The redaction boxes are inconsistent with the manifest or with the page."""


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RedactionError(f"{label}: {value!r} is not a number")
    if isinstance(value, float) and not value.is_integer():
        raise RedactionError(f"{label}: {value!r} must be a whole number of pixels")
    return int(value)


def validate_boxes(
    boxes: Sequence[Sequence[Any]], size: tuple[int, int], *, where: str
) -> list[Box]:
    """Check every box against the page size and return them as integer tuples."""
    width, height = size
    checked: list[Box] = []
    for index, raw in enumerate(boxes, start=1):
        label = f"{where} box {index}"
        if len(raw) != 4:
            raise RedactionError(f"{label}: expected [x, y, w, h], got {list(raw)!r}")
        x, y, w, h = (_as_int(v, label) for v in raw)
        if w < 1 or h < 1:
            raise RedactionError(f"{label}: width and height must be positive, got [{w}, {h}]")
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise RedactionError(
                f"{label}: [{x}, {y}, {w}, {h}] does not fit the {width}x{height} page "
                "(coordinates taken at another resolution?)"
            )
        checked.append((x, y, w, h))
    return checked


def apply_boxes(image: Image.Image, boxes: Sequence[Box]) -> Image.Image:
    """A copy of `image` with every box painted solid black. The input is never modified."""
    redacted = flatten(image)  # always a new image; transparency goes to white, not black
    if boxes:
        draw = ImageDraw.Draw(redacted)
        for x, y, w, h in boxes:
            draw.rectangle([x, y, x + w - 1, y + h - 1], fill=(0, 0, 0))
    return redacted
