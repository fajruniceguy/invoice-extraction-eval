"""Cheap, local facts about a raw file: what it really is, how many pages, whether it has a text
layer or camera EXIF.

Used to pre-fill the manifest skeleton. It reads only METADATA and counts: the text layer of a
PDF is used to count characters and score Indonesian-vs-English stopwords and is then thrown
away; pixels are never interpreted. Nothing here calls the network.
"""

import hashlib
import io
import re
from dataclasses import dataclass, replace
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

SUPPORTED_EXTENSIONS = frozenset(
    {"pdf", "jpg", "jpeg", "jfif", "png", "heic", "heif", "webp", "bmp", "tif", "tiff"}
)

_EXIF_MAKE, _EXIF_MODEL, _EXIF_ORIENTATION, _EXIF_GPS_IFD = 0x010F, 0x0110, 0x0112, 0x8825
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}

# Disjoint stopword lists: a decisive majority of hits, not a single word, picks the language.
_ID_WORDS = frozenset(
    {
        "yang", "dan", "untuk", "dengan", "tidak", "dari", "pada", "ini", "itu", "atau",
        "adalah", "jumlah", "harga", "tanggal", "nama", "alamat", "pembayaran", "faktur",
        "pajak", "kepada", "terima", "kasih", "nomor", "barang", "pelanggan", "tagihan",
        "sebesar", "dibayar",
    }
)  # fmt: skip
_EN_WORDS = frozenset(
    {
        "the", "and", "for", "with", "not", "from", "this", "that", "you", "your", "are", "is",
        "date", "name", "address", "payment", "booking", "confirmation", "thank", "number",
        "items", "customer", "price", "amount", "tax", "invoice", "receipt",
    }
)  # fmt: skip
_MIN_LANGUAGE_HITS = 4
_TEXT_LAYER_MIN_CHARS = 50
_LANGUAGE_SAMPLE_CHARS = 20_000


def register_heif() -> bool:
    """Let Pillow open HEIC/HEIF if pillow-heif is installed. Returns whether it is."""
    try:
        import pillow_heif
    except ImportError:
        return False
    pillow_heif.register_heif_opener()
    return True


def sniff_format(head: bytes) -> str | None:
    """The real format from the first bytes, whatever the extension says."""
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[4:8] == b"ftyp" and head[8:12] in _HEIF_BRANDS:
        return "heic"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if head.startswith(b"BM"):
        return "bmp"
    return None


@dataclass(frozen=True)
class FileProbe:
    name: str  # path relative to the raw folder, '/' separators
    extension: str  # lowercase, no dot
    size_bytes: int
    sha256: str
    content_format: str | None = None  # from the bytes, not the extension
    page_count: int | None = None
    image_size: tuple[int, int] | None = None  # (width, height) of an image, before EXIF rotation
    text_chars: int | None = None  # PDF text-layer characters, all pages
    language: str | None = None  # "id" / "en" when decisive, from a text layer only
    has_camera_exif: bool = False  # EXIF Make/Model present (a camera or phone wrote it)
    has_gps: bool = False
    exif_orientation: int | None = None
    error: str | None = None  # why the file could not be read

    @property
    def has_text_layer(self) -> bool:
        return (self.text_chars or 0) >= _TEXT_LAYER_MIN_CHARS

    @property
    def extension_mismatch(self) -> bool:
        """True when the bytes are not what the extension claims (a .jpg that is a PNG...)."""
        if self.content_format is None:
            return False
        expected = {"jpg": "jpeg", "jfif": "jpeg", "heif": "heic", "tif": "tiff"}.get(
            self.extension, self.extension
        )
        return expected != self.content_format


def guess_language(text: str) -> str | None:
    words = re.findall(r"[a-z']+", text.lower())
    indonesian = sum(word in _ID_WORDS for word in words)
    english = sum(word in _EN_WORDS for word in words)
    if indonesian >= _MIN_LANGUAGE_HITS and indonesian >= 2 * english:
        return "id"
    if english >= _MIN_LANGUAGE_HITS and english >= 2 * indonesian:
        return "en"
    return None  # mixed or too little text: say nothing rather than guess


def _probe_pdf(data: bytes, base: FileProbe) -> FileProbe:
    document = pdfium.PdfDocument(data)
    try:
        pages = len(document)
        chars = 0
        sample: list[str] = []
        sampled = 0
        for index in range(pages):
            textpage = document[index].get_textpage()
            chars += textpage.count_chars()
            if sampled < _LANGUAGE_SAMPLE_CHARS:
                chunk = textpage.get_text_range()[: _LANGUAGE_SAMPLE_CHARS - sampled]
                sample.append(chunk)
                sampled += len(chunk)
    finally:
        document.close()
    language = guess_language(" ".join(sample)) if chars >= _TEXT_LAYER_MIN_CHARS else None
    return _with(base, page_count=pages, text_chars=chars, language=language)


def _probe_image(data: bytes, base: FileProbe) -> FileProbe:
    register_heif()
    with Image.open(io.BytesIO(data)) as image:
        exif = image.getexif()
        gps = bool(exif.get_ifd(_EXIF_GPS_IFD)) if _EXIF_GPS_IFD in exif else False
        orientation = exif.get(_EXIF_ORIENTATION)
        return _with(
            base,
            page_count=1,
            image_size=image.size,
            has_camera_exif=bool(exif.get(_EXIF_MAKE) or exif.get(_EXIF_MODEL)),
            has_gps=gps,
            exif_orientation=int(orientation) if isinstance(orientation, int) else None,
        )


def _with(probe: FileProbe, **changes: object) -> FileProbe:
    return replace(probe, **changes)  # type: ignore[arg-type]


def probe_file(path: Path, name: str | None = None) -> FileProbe:
    """Probe one file. Never raises: a file that cannot be read comes back with `error` set."""
    data = path.read_bytes()
    base = FileProbe(
        name=name or path.name,
        extension=path.suffix.lower().lstrip("."),
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        content_format=sniff_format(data[:16]),
    )
    try:
        if base.content_format == "pdf":
            return _probe_pdf(data, base)
        if base.content_format is None and base.extension == "pdf":
            return _with(base, error="not a PDF (no %PDF header)")
        return _probe_image(data, base)
    except Exception as exc:  # corrupt, encrypted, truncated, or HEIC without pillow-heif
        message = f"{type(exc).__name__}: {exc}"
        if base.content_format == "heic":
            message += " (install pillow-heif to read HEIC)"
        return _with(base, error=message)
