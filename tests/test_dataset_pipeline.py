"""The whole dataset pipeline, end to end, with the real scripts on a synthetic dataset.

Every step has its own tests; this one proves they fit together, in the order a person runs them:

    manifest_init -> (fill the manifest by hand) -> construct_invoices -> ingest (with redaction)
    -> assign_split -> degrade_docs -> assign_split again -> prelabel -> label -> check_ground_truth

and that the promises the pieces make to each other hold at the end: no content group straddles
dev and test, blind documents never have a draft, a redacted page stays redacted in its degraded
variants, variants resolve to the label of their source, and nothing but pages is stored."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from docextract.ingest.manifest import ManifestRow, check_manifest, read_manifest, write_manifest
from evalkit.groundtruth import SCALAR_FIELD_NAMES
from evalkit.gt_store import check_labels, label_path, load_labels
from evalkit.labeling import LabelForm, save_label
from evalkit.prelabel import eligibility
from fakes.fake_llm import FakeLLMServer, chat_reply
from files import make_jpeg, make_text_pdf
from ingest_support import ROOT
from ingest_support import pixels as pixels_of

pytestmark = pytest.mark.slow

DRAFT = {
    "vendor_name": "PT Contoh Makmur",
    "vendor_address": "Jl. Merdeka 12",
    "vendor_tax_id": None,
    "buyer_name": "CV Pembeli",
    "document_number": "INV/1",
    "document_date": "5 Maret 2024",
    "due_date": None,
    "currency": "Rp",
    "payment_method": None,
    "line_items": None,
    "subtotal": "450.000,00",
    "tax": "49.500,00",
    "discount": None,
    "total": "499.500,00",
    "notes": None,
}
TEXT = (
    "Faktur penjualan nomor 0001 tanggal lima Maret dua ribu dua puluh empat dari PT Contoh "
    "Makmur kepada CV Pembeli rincian barang kertas A4 sepuluh rim jumlah total seluruhnya "
    "empat ratus sembilan puluh sembilan ribu lima ratus rupiah termasuk PPN sebelas persen"
)


class Dataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw = root / "data" / "private" / "raw"
        self.pages = root / "data" / "pages"
        self.manifest = root / "data" / "manifest.csv"
        self.redactions = root / "data" / "redactions.yaml"
        self.gt = root / "ground_truth"
        self.drafts = self.gt / "drafts"
        self.changelog = self.gt / "CHANGELOG.md"

    def rows(self) -> list[ManifestRow]:
        return read_manifest(self.manifest)

    def by_file(self, name: str) -> ManifestRow:
        return next(r for r in self.rows() if r.raw_files and r.raw_files[0] == name)

    def script(
        self, name: str, *args: str, env: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        base = {k: v for k, v in os.environ.items() if not k.startswith(("LLM_", "PRELABEL_"))}
        base.update(env or {})
        done = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=base,
            cwd=ROOT,
            timeout=300,
        )
        return done.returncode, done.stdout, done.stderr

    def ok(self, name: str, *args: str, env: dict[str, str] | None = None) -> str:
        code, out, err = self.script(name, *args, env=env)
        assert code == 0, f"{name} {args} exited {code}\n{out}\n{err}"
        return out


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return Dataset(tmp_path_factory.mktemp("dataset"))


def fill_by_hand(data: Dataset) -> str:
    """What a person does after manifest_init: decide the types, the redaction and the
    duplicates. Returns the doc_id of the document that needs redaction."""
    rows = data.rows()
    updated: list[ManifestRow] = []
    redacted = ""
    first_receipt = ""
    for row in rows:
        name = row.raw_files[0]
        changes: dict[str, Any] = {"needs_redaction": False}
        if name.startswith("faktur_"):
            changes["doc_type"] = "faktur"
        elif name.startswith("invoice_"):
            changes["doc_type"] = "invoice"
        else:
            changes["doc_type"] = "receipt"
        if name == "invoice_1.pdf":
            changes["needs_redaction"] = True
            redacted = row.doc_id
        if name == "receipt_1.jpg":
            first_receipt = row.doc_id
        if name == "receipt_1_again.jpg":  # a second photo of the same receipt
            changes.update(gt_from=first_receipt, content_group=first_receipt)
        updated.append(row.model_copy(update=changes))
    write_manifest(data.manifest, updated)
    return redacted


def blank_pixel_count(path: Path, box: tuple[int, int, int, int]) -> float:
    """The share of the box that is black."""
    with Image.open(path) as image:
        crop = image.convert("L").crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
    pixels = pixels_of(crop)
    return sum(1 for p in pixels if p < 20) / len(pixels)


def test_the_whole_pipeline(dataset: Dataset, fake_llm: FakeLLMServer) -> None:
    data = dataset

    # ---- raw files and the manifest skeleton
    for index in range(1, 4):
        make_text_pdf(data.raw / f"faktur_{index}.pdf", TEXT, pages=1 + index % 2)
        make_text_pdf(data.raw / f"invoice_{index}.pdf", TEXT, pages=1)
    for index in range(1, 5):
        make_jpeg(data.raw / f"receipt_{index}.jpg", make="Xiaomi", model="Redmi", size=(300, 400))
    make_jpeg(data.raw / "receipt_1_again.jpg", make="Xiaomi", model="Redmi", size=(300, 400))

    out = data.ok("manifest_init.py", "--raw-dir", str(data.raw), "--manifest", str(data.manifest))
    assert "11" in out
    rows = data.rows()
    assert len(rows) == 11
    assert {r.capture for r in rows if r.raw_files[0].endswith(".pdf")} == {"born_digital"}
    assert {r.capture for r in rows if r.raw_files[0].endswith(".jpg")} == {"phone_photo"}
    redacted = fill_by_hand(data)

    # ---- constructed invoices join the manifest, with their labels and log lines
    data.ok(
        "construct_invoices.py",
        "--raw-dir", str(data.raw),
        "--ground-truth-dir", str(data.gt),
        "--manifest", str(data.manifest),
        "--changelog", str(data.changelog),
    )  # fmt: skip
    assert len(data.rows()) == 14
    assert data.changelog.read_text(encoding="utf-8").count("(new file)") == 3

    # ---- ingestion, with one document redacted
    data.redactions.write_text(f"{redacted}:\n  1:\n    - [10, 10, 80, 40]\n", encoding="utf-8")
    common = [
        "--manifest",
        str(data.manifest),
        "--raw-dir",
        str(data.raw),
        "--pages-dir",
        str(data.pages),
    ]
    out = data.ok("ingest.py", *common, "--redactions", str(data.redactions), "--dpi", "60")
    assert "14 ingested" in out
    again = data.ok("ingest.py", *common, "--redactions", str(data.redactions), "--dpi", "60")
    assert "0 ingested, 14 unchanged" in again  # idempotent
    page = data.pages / redacted / "p1.png"
    assert blank_pixel_count(page, (10, 10, 80, 40)) == 1.0  # burned in before storing
    assert sorted(p.name for p in data.pages.iterdir()) == sorted(r.doc_id for r in data.rows())

    # ---- the split and the blind picks
    split_args = ["--manifest", str(data.manifest), "--ground-truth-dir", str(data.gt)]
    out = data.ok("assign_split.py", *split_args, "--blind-count", "3")
    assert "blind: 3 document(s)" in out
    rows = data.rows()
    assert all(r.split for r in rows) and check_manifest(rows) == []
    blind = {r.doc_id for r in rows if r.label_mode == "blind"}
    assert len(blind) == 3
    assert all(r.split == "test" for r in rows if r.doc_id in blind)
    assert all(r.capture != "constructed" and r.gt_from is None for r in rows if r.doc_id in blind)
    assert all(r.label_mode is None for r in rows if r.capture == "constructed")
    duplicate, owner = data.by_file("receipt_1_again.jpg"), data.by_file("receipt_1.jpg")
    assert duplicate.split == owner.split and duplicate.label_mode is None

    # ---- degraded variants of the born-digital documents, then the split again
    data.ok(
        "degrade_docs.py", "--all", "--levels", "light", "--backend", "opencv",
        "--manifest", str(data.manifest), "--pages-dir", str(data.pages),
    )  # fmt: skip
    rows = data.rows()
    variants = [r for r in rows if r.is_derived]
    assert len(variants) == 6 and all(
        r.split is None or r.split in ("dev", "test") for r in variants
    )
    out = data.ok("assign_split.py", *split_args, "--blind-count", "3")
    rows = data.rows()
    by_id = {r.doc_id: r for r in rows}
    assert check_manifest(rows) == []
    assert all(r.split == by_id[str(r.gt_from)].split for r in rows if r.is_derived)
    assert {r.doc_id for r in rows if r.label_mode == "blind"} == blind  # the picks did not move
    redacted_variant = data.pages / f"{redacted}-deg-light" / "p1.png"
    assert blank_pixel_count(redacted_variant, (30, 30, 40, 20)) > 0.9  # still a black box
    with Image.open(redacted_variant) as variant, Image.open(page) as source:
        assert variant.size == source.size

    # ---- pre-labeling: every eligible document, and not one blind document
    env_file = data.root / "test.env"
    env_file.write_text(
        "\n".join(
            [
                f"LLM_BASE_URL={fake_llm.base_url}",
                "LLM_MODEL=extractor-model",
                f"PRELABEL_BASE_URL={fake_llm.base_url}",
                "PRELABEL_MODEL=prelabel-model",
                "PRELABEL_PRICE_INPUT_PER_MTOK=1",
                "PRELABEL_PRICE_OUTPUT_PER_MTOK=5",
                "LLM_MAX_RETRIES=0",
                "LLM_CACHE=off",
            ]
        ),
        encoding="utf-8",
    )
    eligible = [r.doc_id for r in rows if eligibility(r) is None]
    assert len(eligible) == 7  # 10 hand-labeled documents, minus the 3 blind ones
    assert blind.isdisjoint(eligible)
    fake_llm.enqueue(*[chat_reply(DRAFT) for _ in eligible])
    out = data.ok(
        "prelabel.py", "--all",
        "--manifest", str(data.manifest), "--pages-dir", str(data.pages),
        "--drafts-dir", str(data.drafts), "--ground-truth-dir", str(data.gt),
        "--env-file", str(env_file),
    )  # fmt: skip
    assert "7 drafted, 0 failed" in out
    assert sorted(p.stem for p in data.drafts.iterdir()) == sorted(eligible)
    assert len(fake_llm.requests) == 7
    assert check_labels(rows, data.gt, data.drafts) == []  # no blind document has a draft

    # ---- labeling, as the app does it: drafts for prelabel documents, from scratch for blind
    for row in rows:
        if row.gt_from is not None or row.capture == "constructed":
            continue
        from evalkit.gt_store import read_label

        draft = (
            read_label(label_path(data.drafts, row.doc_id))
            if row.label_mode == "prelabel"
            else None
        )
        form = LabelForm.start(row, draft)
        for name in SCALAR_FIELD_NAMES:
            if name == "vendor_name":
                form.set_text(name, "PT Contoh Makmur")
                assert form.verify(name)
            else:
                form.mark_absent(name)
        form.mark_rows_absent()
        save_label(
            form.build_label(),
            row,
            ground_truth_dir=data.gt,
            changelog=data.changelog,
            drafts_dir=data.drafts,
        )
    log = data.changelog.read_text(encoding="utf-8")
    assert log.count("labeled by hand") == 10 and log.count("(new file)") == 13

    # ---- the integrity check, and the promises at the end
    out = data.ok(
        "check_ground_truth.py",
        "--manifest",
        str(data.manifest),
        "--ground-truth-dir",
        str(data.gt),
    )
    assert "problems: none" in out
    assert "20 document(s) labeled (7 by borrowing), 0 without a label yet" in out
    labels, missing = load_labels(data.rows(), data.gt)
    assert missing == []
    for row in data.rows():
        resolved = labels[row.doc_id]
        assert resolved.label.doc_id == row.doc_id
        assert resolved.owner == (row.gt_from or row.doc_id)
    assert labels[duplicate.doc_id].borrowed and labels[variants[0].doc_id].borrowed
    assert read_json(data.gt / f"{owner.doc_id}.json")["label_mode"] in ("prelabel", "blind")

    # nothing but pages and their sidecars is stored, and the raw files are not under pages/
    stored = {p.name for folder in data.pages.iterdir() for p in folder.iterdir()}
    assert all(name.startswith("p") or name in ("ingest.json", "degrade.json") for name in stored)
    assert not any(data.raw.name in str(p) for p in data.pages.rglob("*"))
    for doc in data.rows():
        assert (data.gt / f"{doc.doc_id}.json").exists() == (doc.gt_from is None)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
