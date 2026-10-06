"""Page-image geometry and encoding, shared by ingestion and the model request.

Two different sizes exist and must not be confused:

  - the STORED size is a ceiling applied once at ingestion (INGEST_MAX_LONG_EDGE, 2400 px),
    so the committed dataset stays a sane size;
  - the SENT size is chosen per request (LLM_IMAGE_MAX_LONG_EDGE) from the stored page, in
    memory, so a resolution sweep needs no re-ingest. The stored file is never touched.

Both use the same shape rule so they cannot disagree:

  - normal pages: the long edge is capped at E;
  - elongated pages (long/short > 2.5, e.g. thermal receipts): capping the long edge would
    squeeze the width and blur small print, so the AREA is capped instead, at ~6 MP when
    E = 2400 and scaling with E squared;
  - nothing is ever upscaled, and no side exceeds 8000 px (a common provider limit).
"""

import io
import math
from pathlib import Path
from typing import Literal

from PIL import Image

REFERENCE_EDGE = 2400  # the stored-page cap; the area budget below is defined at this edge
ELONGATED_RATIO = 2.5
ELONGATED_AREA_AT_REFERENCE = 6_000_000
HARD_MAX_SIDE = 8000
DEFAULT_JPEG_QUALITY = 92

PageFormat = Literal["png", "jpeg"]

# Pillow format names of sources that are ALREADY lossy: re-encoding them as PNG would add
# megabytes and no fidelity, so they are stored as JPEG. Everything else is stored as PNG.
_LOSSY_SOURCES = frozenset({"JPEG", "MPO", "HEIF", "HEIC", "AVIF"})


def target_size(width: int, height: int, max_long_edge: int | None) -> tuple[int, int]:
    """Size after the cap rule (see module docstring). `None`/0 means only the hard limit."""
    if width < 1 or height < 1:
        raise ValueError(f"invalid image size {width}x{height}")
    long_side, short_side = max(width, height), min(width, height)
    scale = 1.0
    if max_long_edge:
        if long_side / short_side > ELONGATED_RATIO:
            max_area = ELONGATED_AREA_AT_REFERENCE * (max_long_edge / REFERENCE_EDGE) ** 2
            if width * height > max_area:
                scale = math.sqrt(max_area / (width * height))
        elif long_side > max_long_edge:
            scale = max_long_edge / long_side
    if long_side * scale > HARD_MAX_SIDE:
        scale = HARD_MAX_SIDE / long_side
    if scale >= 1.0:
        return width, height
    return max(1, int(width * scale + 0.5)), max(1, int(height * scale + 0.5))


def storage_format(source_format: str | None) -> PageFormat:
    """PNG for lossless sources (PDF renders, PNG screenshots), JPEG for lossy photos."""
    if source_format and source_format.upper() in _LOSSY_SOURCES:
        return "jpeg"
    return "png"


def flatten(image: Image.Image) -> Image.Image:
    """A new RGB image, with any transparency composited on WHITE (receipts are dark ink on
    white). Plain `convert("RGB")` would drop the alpha channel and turn transparent pixels
    black, which can make black text unreadable."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def encode_page(
    image: Image.Image, fmt: PageFormat, *, jpeg_quality: int = DEFAULT_JPEG_QUALITY
) -> bytes:
    """Encode deterministically with NO metadata (no EXIF, ICC, DPI or text chunks)."""
    buffer = io.BytesIO()
    flat = flatten(image)  # always a new object, so clearing .info never touches the caller's
    # Pillow's encoders re-emit entries of `.info` (comment text, ICC profile, DPI) even when no
    # metadata is passed to save(): clear it so NOTHING from the source survives
    flat.info = {}
    if fmt == "png":
        flat.save(buffer, format="PNG", compress_level=9)
    else:
        flat.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return buffer.getvalue()


def resize_to(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def _mime(fmt: PageFormat) -> str:
    return "image/png" if fmt == "png" else "image/jpeg"


def prepare_for_model(
    source: Path | bytes,
    *,
    max_long_edge: int | None = None,
    max_bytes: int | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> tuple[bytes, str]:
    """Bytes and MIME type to send to a model for one stored page, derived in memory.

    When the page already fits (size rule and `max_bytes`) the stored bytes are returned
    untouched, so request payloads and cache keys stay stable. Otherwise the page is resized
    and re-encoded in the same family (PNG stays PNG, JPEG stays JPEG); if `max_bytes` is still
    exceeded it falls back to JPEG with shrinking dimensions. Never writes anything.
    """
    data = source.read_bytes() if isinstance(source, Path) else source
    with Image.open(io.BytesIO(data)) as opened:
        opened.load()
        width, height = opened.size
        family: PageFormat = "png" if opened.format == "PNG" else "jpeg"
        size = target_size(width, height, max_long_edge)
        within_bytes = max_bytes is None or len(data) <= max_bytes
        if size == (width, height) and within_bytes and opened.format in ("PNG", "JPEG"):
            return data, _mime(family)

        resized = resize_to(opened, size)
        encoded = encode_page(resized, family, jpeg_quality=jpeg_quality)
        if max_bytes is None or len(encoded) <= max_bytes:
            return encoded, _mime(family)

        # still too large: JPEG at lower quality, then progressively smaller
        current = resized
        quality = min(jpeg_quality, 85)
        for _ in range(8):
            encoded = encode_page(current, "jpeg", jpeg_quality=quality)
            if len(encoded) <= max_bytes:
                return encoded, "image/jpeg"
            smaller = (max(1, int(current.width * 0.85)), max(1, int(current.height * 0.85)))
            current = resize_to(current, smaller)
        raise ValueError(f"cannot fit the image into {max_bytes} bytes")
