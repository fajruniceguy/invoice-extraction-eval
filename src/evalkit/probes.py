"""The hallucination probe set (S4): copies of constructed invoices with fields digitally erased.

A probe is a copy of a document the dataset already has (cons_001..003, regenerated from the same
`Invoice`), with a value erased from the page, or the last page dropped. The truth is known by
construction: the erased field is not on the page any more, so the expected answer is `absent`
(or `abstained`), and any value the system returns for it is a hallucination, whatever a human
label says.

How a probe is made, all of it deterministic:

  1. render the invoice's PDF and every page of it exactly as ingestion does (`load_pages`);
  2. find every occurrence of the erased VALUE in the PDF text layer (the printed label next to
     it, "NPWP:" or "Pembeli:", stays, so the page still invites an answer) and paint over it
     with the colour found around it (the modern layout prints its header in white on a dark
     band); a box that held no ink, a value found a different number of times than the layout
     prints it, or a box that is not one flat colour afterwards makes the probe an error;
  3. save the pages as PNG and ingest them with the real `ingest_document`, so a probe is stored
     like any other document (`pN.png` + `ingest.json`);
  4. derive the label from the same `Invoice`: erased fields are null, a dropped last page nulls
     subtotal, discount, tax and total and keeps only the rows of the remaining pages, and what
     the remaining pages no longer decide (the currency) is `__unlabeled__`, never guessed.

No real page is read: the sources are synthetic. The controls (c01..c03) are the unmodified
documents, re-ingested, and their pages must equal the dataset's byte for byte.
"""

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from docextract.config import IngestSettings
from docextract.ingest.manifest import ManifestRow, write_manifest
from docextract.ingest.pages import load_pages
from docextract.ingest.pipeline import SIDECAR, ingest_document, sha256_hex
from evalkit.construct import (
    DEFAULT_SEED,
    LAYOUTS,
    Invoice,
    format_date,
    format_money,
    generate_invoice,
    ground_truth,
    render_invoice,
)
from evalkit.construct_io import LAYOUT_ORDER
from evalkit.groundtruth import UNLABELED, GroundTruth
from evalkit.gt_store import label_path, write_label

PAD = 3  # pixels added around a found text box
MIN_INK = 20  # pixels that must differ from the background in a box, or nothing was printed there
INK_DELTA = 40  # a pixel is ink when some channel differs from the fill by more than this
TOTALS = ("subtotal", "discount", "tax", "total")
PROBES_JSON = "probes.json"

# where a field is printed: every page (the number repeats in each continuation header), the
# first page (the header block), or the last page (the totals block)
PLACEMENT: dict[str, str] = {
    "document_number": "all",
    "vendor_name": "first",
    "vendor_address": "first",
    "vendor_tax_id": "first",
    "buyer_name": "first",
    "document_date": "first",
    "due_date": "first",
    "payment_method": "first",
    "subtotal": "last",
    "discount": "last",
    "tax": "last",
    "total": "last",
}


class ProbeError(Exception):
    """A probe cannot be made as specified (and why)."""


@dataclass(frozen=True)
class ProbeSpec:
    probe_id: str
    source: str  # cons_001..cons_003
    erase: tuple[str, ...] = ()  # field values erased from the page
    drop_last_page: bool = False

    @property
    def is_control(self) -> bool:
        return not self.erase and not self.drop_last_page


PROBES: tuple[ProbeSpec, ...] = (
    ProbeSpec("p01", "cons_001", erase=("document_number",)),
    ProbeSpec("p02", "cons_001", erase=("vendor_tax_id",)),
    ProbeSpec("p03", "cons_002", erase=("document_date",)),
    ProbeSpec("p04", "cons_002", erase=("vendor_name", "vendor_address")),
    ProbeSpec("p05", "cons_003", erase=("buyer_name",)),
    ProbeSpec("p06", "cons_003", erase=("due_date", "payment_method")),
    ProbeSpec("p07", "cons_001", drop_last_page=True),
    ProbeSpec("p08", "cons_002", erase=("subtotal", "tax", "total")),
    ProbeSpec("p09", "cons_003", drop_last_page=True),
    ProbeSpec("c01", "cons_001"),
    ProbeSpec("c02", "cons_002"),
    ProbeSpec("c03", "cons_003"),
)


@dataclass(frozen=True)
class Erasure:
    field: str
    value: str  # the text that was printed and is now painted over
    pages: tuple[int, ...]  # 0-based pages where it was erased (one box each)


@dataclass
class BuiltProbe:
    spec: ProbeSpec
    layout: str
    images: list[Image.Image]
    label: dict[str, Any]
    original: dict[str, Any]  # the unmodified label of the source, to tell recovered from invented
    erased: tuple[str, ...]  # fields whose truth is now "absent" (erasures + dropped totals)
    erasures: list[Erasure] = field(default_factory=list)
    source_pages: int = 0


# ---------------------------------------------------------------- the invoice


def invoice_for(source: str, seed: str = DEFAULT_SEED) -> Invoice:
    """The constructed invoice `source` (cons_001..cons_003), exactly as the dataset has it."""
    try:
        layout = LAYOUT_ORDER[int(source.rsplit("_", 1)[1]) - 1]
    except (ValueError, IndexError):
        raise ProbeError(f"{source}: not a constructed invoice (cons_001..cons_003)") from None
    return generate_invoice(source, layout, seed)


def target_text(invoice: Invoice, name: str) -> str:
    """The text the layout prints for field `name` (what a text search must find)."""
    layout = LAYOUTS[invoice.layout]
    decimals = layout.money_decimals
    simple = {
        "document_number": invoice.number,
        "vendor_name": invoice.vendor_name,
        "vendor_address": invoice.vendor_address,
        "vendor_tax_id": invoice.vendor_tax_id,
        "buyer_name": invoice.buyer_name,
        "document_date": format_date(invoice.issued, layout.date_style),
        "payment_method": layout.payment_method,
        "subtotal": format_money(invoice.subtotal, decimals),
        "tax": format_money(invoice.tax, decimals),
        "total": format_money(invoice.total, decimals),
    }
    if name == "due_date" and invoice.due is not None:
        return format_date(invoice.due, layout.date_style)
    if name == "discount" and invoice.discount is not None:
        return format_money(invoice.discount, decimals)
    if name not in simple:
        raise ProbeError(f"{invoice.doc_id}: the {invoice.layout} layout does not print {name}")
    return simple[name]


def _printed(invoice: Invoice, name: str) -> bool:
    layout = LAYOUTS[invoice.layout]
    return {
        "vendor_tax_id": layout.prints_tax_id,
        "due_date": layout.prints_due_date,
        "payment_method": layout.prints_payment_method,
        "discount": layout.discount is not None,
    }.get(name, True)


# ---------------------------------------------------------------- finding and painting


def _find(textpage: Any, value: str) -> list[tuple[float, float, float, float]]:
    """Every occurrence of `value` in the page's text layer, as (left, bottom, right, top) in
    PDF points (the union of the character boxes of the match)."""
    searcher = textpage.search(value, match_case=True)
    found: list[tuple[float, float, float, float]] = []
    while True:
        hit = searcher.get_next()
        if hit is None:
            break
        index, count = hit
        boxes = [textpage.get_charbox(i) for i in range(index, index + count)]
        found.append(
            (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
        )
    searcher.close()
    return found


def _ring(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, ...]:
    """The colour around `box`: the median of the one-pixel ring outside it, per channel."""
    left, top, right, bottom = box
    width, height = image.size
    pixels: list[Any] = []
    for x in range(max(left - 1, 0), min(right + 1, width)):
        for y in (top - 1, bottom):
            if 0 <= y < height:
                pixels.append(image.getpixel((x, y)))
    for y in range(max(top, 0), min(bottom, height)):
        for x in (left - 1, right):
            if 0 <= x < width:
                pixels.append(image.getpixel((x, y)))
    if not pixels:
        raise ProbeError("a box with no surroundings")
    channels = len(pixels[0]) if isinstance(pixels[0], tuple) else 1
    if channels == 1:
        return (sorted(pixels)[len(pixels) // 2],)
    return tuple(sorted(p[c] for p in pixels)[len(pixels) // 2] for c in range(channels))


def _ink(image: Image.Image, box: tuple[int, int, int, int], colour: tuple[int, ...]) -> int:
    """How many pixels of `box` differ from `colour`."""
    count = 0
    crop = image.crop(box)
    pixels = getattr(crop, "get_flattened_data", None) or crop.getdata
    for pixel in pixels():
        values = pixel if isinstance(pixel, tuple) else (pixel,)
        if any(abs(a - b) > INK_DELTA for a, b in zip(values, colour, strict=True)):
            count += 1
    return count


def erase_box(image: Image.Image, box: tuple[float, float, float, float]) -> None:
    """Paint over the box (pixels: left, top, right, bottom) with its surrounding colour, in
    place. Refused when nothing was printed there, or when the result is not one flat colour."""
    left, top, right, bottom = box
    width, height = image.size
    rect = (
        max(int(left) - PAD, 0),
        max(int(top) - PAD, 0),
        min(int(right) + 1 + PAD, width),
        min(int(bottom) + 1 + PAD, height),
    )
    colour = _ring(image, rect)
    if _ink(image, rect, colour) < MIN_INK:
        raise ProbeError(f"nothing is printed in the box {rect}: the text position is wrong")
    ImageDraw.Draw(image).rectangle(
        [rect[0], rect[1], rect[2] - 1, rect[3] - 1], fill=colour if len(colour) > 1 else colour[0]
    )
    extrema = image.crop(rect).getextrema()
    bands = extrema if isinstance(extrema[0], tuple) else (extrema,)
    if any(low != high for low, high in bands):
        raise ProbeError(f"the box {rect} is not one flat colour after painting")


def _pixel_box(
    box: tuple[float, float, float, float], page_size: tuple[float, float], image: Image.Image
) -> tuple[float, float, float, float]:
    """PDF points (origin bottom-left) to pixels (origin top-left) on the rendered image."""
    sx, sy = image.width / page_size[0], image.height / page_size[1]
    left, bottom, right, top = box
    return (left * sx, (page_size[1] - top) * sy, right * sx, (page_size[1] - bottom) * sy)


def _pages_for(name: str, total: int) -> tuple[int, ...]:
    place = PLACEMENT[name]
    if place == "all":
        return tuple(range(total))
    return (0,) if place == "first" else (total - 1,)


# ---------------------------------------------------------------- building one probe


def erased_fields(spec: ProbeSpec, invoice: Invoice) -> tuple[str, ...]:
    """The fields whose truth is `absent` once the probe is made."""
    if spec.drop_last_page:
        return tuple(name for name in TOTALS if _printed(invoice, name))
    return spec.erase


def label_for(spec: ProbeSpec, invoice: Invoice, plan: list[int]) -> dict[str, Any]:
    label = ground_truth(invoice)
    label["doc_id"] = spec.probe_id
    erased = erased_fields(spec, invoice)
    for name in erased:
        label[name] = None
    if spec.drop_last_page:
        keep = sum(plan[:-1])
        label["line_items"] = label["line_items"][:keep]
        label["currency"] = UNLABELED  # "Rp" is printed in the totals block, which is gone
        what = f"the last page was removed (rows 1-{keep} remain; {', '.join(erased)} are gone)"
    elif erased:
        what = f"the value of {', '.join(erased)} was painted over (its printed label stays)"
    else:
        what = "control: the unmodified document"
    label["notes"] = f"probe {spec.probe_id} from {spec.source}: {what}"
    return label


def build_probe(
    spec: ProbeSpec, *, seed: str = DEFAULT_SEED, settings: IngestSettings | None = None
) -> BuiltProbe:
    config = settings or IngestSettings(_env_file=None)
    invoice = invoice_for(spec.source, seed)
    layout = LAYOUTS[invoice.layout]
    for name in erased_fields(spec, invoice):
        if not _printed(invoice, name):
            raise ProbeError(f"{spec.probe_id}: {spec.source} does not print {name}")
    with tempfile.TemporaryDirectory(prefix="probe-") as work:
        pdf_path = Path(work) / f"{spec.source}.pdf"
        plan = render_invoice(invoice, pdf_path)
        rendered = load_pages(
            [pdf_path], page_range=None, dpi=config.dpi, max_long_edge=config.max_long_edge
        )
        if len(rendered) != len(plan):
            raise ProbeError(
                f"{spec.probe_id}: {len(rendered)} pages rendered, {len(plan)} laid out"
            )
        images = [page.image.copy() for page in rendered]
        erasures: list[Erasure] = []
        document = pdfium.PdfDocument(pdf_path.read_bytes())
        try:
            for name in spec.erase:
                value = target_text(invoice, name)
                pages = _pages_for(name, len(plan))
                for index in pages:
                    page = document[index]
                    found = _find(page.get_textpage(), value)
                    if len(found) != 1:
                        raise ProbeError(
                            f"{spec.probe_id}: {name} ({value!r}) is found {len(found)} time(s) on "
                            f"page {index + 1}; the {layout.name} layout prints it once"
                        )
                    erase_box(images[index], _pixel_box(found[0], page.get_size(), images[index]))
                erasures.append(Erasure(name, value, pages))
        finally:
            document.close()
    if spec.drop_last_page:
        images = images[:-1]
    return BuiltProbe(
        spec=spec,
        layout=layout.name,
        images=images,
        label=label_for(spec, invoice, plan),
        original=ground_truth(invoice),
        erased=erased_fields(spec, invoice),
        erasures=erasures,
        source_pages=len(plan),
    )


# ---------------------------------------------------------------- writing the probe set


def _png_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def manifest_row(probe: BuiltProbe, names: tuple[str, ...]) -> ManifestRow:
    return ManifestRow(
        doc_id=probe.spec.probe_id,
        raw_files=names,
        doc_type="invoice",
        template_group=f"probe_{probe.layout}",
        content_group=probe.spec.probe_id,
        capture="constructed",
        original_format="png",
        language="id",
        license_note="synthetic: generated by scripts/make_probes.py",
        split="test",
        needs_redaction=False,
    )


def _ingest(probe: BuiltProbe, pages_dir: Path, settings: IngestSettings) -> ManifestRow:
    with tempfile.TemporaryDirectory(prefix="probe-raw-") as work:
        raw = Path(work)
        names = []
        for number, image in enumerate(probe.images, start=1):
            name = f"{probe.spec.probe_id}_p{number}.png"
            (raw / name).write_bytes(_png_bytes(image))
            names.append(name)
        row = manifest_row(probe, tuple(names))
        out = pages_dir / probe.spec.probe_id
        if out.exists():
            shutil.rmtree(out)
        ingest_document(row, raw_dir=raw, out_dir=out, settings=settings)
    return row


def page_hashes(pages_dir: Path, probe_id: str) -> list[str]:
    sidecar = json.loads((pages_dir / probe_id / SIDECAR).read_text(encoding="utf-8"))
    return [str(page["sha256"]) for page in sidecar["pages"]]


def probe_entry(probe: BuiltProbe, hashes: list[str]) -> dict[str, Any]:
    return {
        "id": probe.spec.probe_id,
        "source": probe.spec.source,
        "layout": probe.layout,
        "kind": "control"
        if probe.spec.is_control
        else ("drop_last_page" if probe.spec.drop_last_page else "erase"),
        "erased": list(probe.erased),
        "erasures": [
            {"field": e.field, "value": e.value, "pages": [p + 1 for p in e.pages]}
            for e in probe.erasures
        ],
        "original": {name: probe.original.get(name) for name in probe.erased},
        "source_pages": probe.source_pages,
        "pages": hashes,
    }


def write_probe_set(
    probes_dir: Path,
    *,
    specs: tuple[ProbeSpec, ...] = PROBES,
    seed: str = DEFAULT_SEED,
    dataset_pages: Path | None = None,
) -> dict[str, Any]:
    """Build every probe and write `probes_dir` (manifest, probes.json, labels/, pages/).
    `dataset_pages` (data/pages) is where the controls and the kept pages are checked against."""
    settings = IngestSettings(_env_file=None)
    pages_dir = probes_dir / "pages"
    labels_dir = probes_dir / "labels"
    rows: list[ManifestRow] = []
    entries: list[dict[str, Any]] = []
    for spec in specs:
        probe = build_probe(spec, seed=seed, settings=settings)
        rows.append(_ingest(probe, pages_dir, settings))
        hashes = page_hashes(pages_dir, spec.probe_id)
        if dataset_pages is not None:
            _check_against_dataset(probe, hashes, dataset_pages)
        write_label(labels_dir, GroundTruth.model_validate(probe.label))
        entries.append(probe_entry(probe, hashes))
    write_manifest(probes_dir / "manifest.csv", rows)
    document = {"seed": seed, "probes": entries}
    (probes_dir / PROBES_JSON).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return document


def _check_against_dataset(probe: BuiltProbe, hashes: list[str], dataset_pages: Path) -> None:
    """The controls, and the pages a drop keeps, must be the dataset's own pages."""
    source = dataset_pages / probe.spec.source / SIDECAR
    if not source.is_file():
        return
    theirs = [
        str(page["sha256"]) for page in json.loads(source.read_text(encoding="utf-8"))["pages"]
    ]
    expected = theirs[: len(hashes)] if probe.spec.drop_last_page else theirs
    if (probe.spec.is_control or probe.spec.drop_last_page) and hashes != expected:
        raise ProbeError(
            f"{probe.spec.probe_id}: its pages are not byte-identical to {probe.spec.source}'s "
            "stored pages: the probe is not a copy of the dataset's document"
        )


def tree_hashes(root: Path) -> dict[str, str]:
    """Every file under `root`, by relative path -> sha256 (to compare two generations)."""
    return {
        path.relative_to(root).as_posix(): sha256_hex(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def read_probes(probes_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads((probes_dir / PROBES_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProbeError(f"{probes_dir / PROBES_JSON}: cannot be read ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("probes"), list):
        raise ProbeError(f"{probes_dir / PROBES_JSON}: not a probe set")
    return data


def label_file(probes_dir: Path, probe_id: str) -> Path:
    return label_path(probes_dir / "labels", probe_id)


def check_sources(raw_dir: Path, seed: str = DEFAULT_SEED) -> list[str]:
    """Sources whose regenerated PDF differs from the dataset's own (an empty list = the probes
    are copies of the documents the dataset has). A missing raw file is not a difference."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="probe-src-") as work:
        for source in sorted({spec.source for spec in PROBES}):
            theirs = raw_dir / "constructed" / f"{source}.pdf"
            if not theirs.is_file():
                continue
            mine = Path(work) / f"{source}.pdf"
            render_invoice(invoice_for(source, seed), mine)
            if mine.read_bytes() != theirs.read_bytes():
                problems.append(f"{source}: the regenerated PDF differs from {theirs}")
    return problems
