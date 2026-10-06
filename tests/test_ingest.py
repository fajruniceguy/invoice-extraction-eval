import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from docextract.imaging import ELONGATED_AREA_AT_REFERENCE
from docextract.ingest.manifest import COLUMNS, ManifestRow
from docextract.ingest.pipeline import (
    INGEST_VERSION,
    SIDECAR,
    IngestError,
    ingest_document,
)
from docextract.ingest.redact import RedactionError, apply_boxes, validate_boxes
from files import (
    A4,
    make_heic,
    make_image_pdf,
    make_jpeg,
    make_png,
    make_png_with_metadata,
    make_rgba_png,
    make_sized_pdf,
    write_long_named,
)
from ingest_support import Env, doc, gray_png, pixels, run_script, settings

# ------------------------------------------------------------------ settings


class TestIngestSettings:
    def test_defaults(self) -> None:
        s = settings()
        assert (s.dpi, s.max_long_edge, s.jpeg_quality) == (200, 2400, 92)

    def test_environment_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INGEST_DPI", "150")
        monkeypatch.setenv("INGEST_MAX_LONG_EDGE", "0")
        monkeypatch.setenv("INGEST_JPEG_QUALITY", "80")
        s = settings()
        assert (s.dpi, s.max_long_edge, s.jpeg_quality) == (150, 0, 80)

    @pytest.mark.parametrize(
        "bad",
        [
            {"dpi": 10},
            {"dpi": 5000},
            {"max_long_edge": -1},
            {"jpeg_quality": 0},
            {"jpeg_quality": 101},
        ],
    )
    def test_bounds(self, bad: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            settings(**bad)

    def test_it_needs_no_llm_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        settings()  # must not raise


# ------------------------------------------------------------------ rendering and sizes


class TestPdfPages:
    def test_a4_at_the_default_dpi(self, env: Env) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4])
        row = doc("a.pdf")
        result = env.ingest(row)
        assert result.status == "ingested"
        (page,) = result.pages
        assert page.file == "p1.png"
        assert page.size == (1654, 2339)  # 595.276 x 841.89 pt at 200 DPI
        assert page.format == "png"
        assert env.page(row, "p1.png").size == (1654, 2339)

    def test_dpi_controls_the_pixel_size(self, env: Env) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4])
        assert env.ingest(doc("a.pdf"), settings=settings(dpi=100)).pages[0].size == (827, 1169)

    def test_a_pdf_is_rendered_in_colour_on_white(self, env: Env) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4])
        row = doc("a.pdf")
        env.ingest(row)
        page = env.page(row, "p1.png")
        assert page.mode == "RGB"
        assert page.getpixel((5, page.height - 5)) == (255, 255, 255)  # the corner is blank paper
        assert page.getpixel((page.width // 2, 140)) == (0, 0, 0)  # the black bar near the top

    def test_a_higher_dpi_is_capped_on_the_long_edge(self, env: Env) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4])
        (page,) = env.ingest(doc("a.pdf"), settings=settings(dpi=300)).pages
        assert max(page.size) == 2400  # 3508 px would exceed the 2400 ceiling
        assert abs(page.size[0] / page.size[1] - A4[0] / A4[1]) < 0.002

    def test_cap_zero_means_no_cap(self, env: Env) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4])
        (page,) = env.ingest(doc("a.pdf"), settings=settings(dpi=300, max_long_edge=0)).pages
        assert page.size == (2480, 3508)

    def test_an_elongated_page_keeps_its_width(self, env: Env) -> None:
        # a 200 x 900 pt "thermal receipt": 556 x 2500 px at 200 DPI. The long edge is over 2400,
        # but at 1.4 MP it is far inside the area budget, so it must NOT be squeezed.
        make_sized_pdf(env.raw / "r.pdf", [(200, 900)])
        (page,) = env.ingest(doc("r.pdf")).pages
        assert page.size == (556, 2500)

    def test_a_huge_elongated_page_is_capped_by_area(self, env: Env) -> None:
        make_sized_pdf(env.raw / "poster.pdf", [(2000, 6000)])  # 92 MP at 200 DPI
        (page,) = env.ingest(doc("poster.pdf")).pages
        width, height = page.size
        assert width * height <= ELONGATED_AREA_AT_REFERENCE * 1.001
        assert width * height >= ELONGATED_AREA_AT_REFERENCE * 0.98
        assert height > 2400  # the long edge may exceed the cap: the AREA is what is limited
        assert abs(width / height - 1 / 3) < 0.002

    def test_page_range_selects_pages_and_numbers_them_from_one(self, env: Env) -> None:
        make_sized_pdf(env.raw / "bundle.pdf", [A4, A4, (300, 400), A4, A4])
        row = doc("bundle.pdf", page_range="3-4")
        result = env.ingest(row)
        assert [p.file for p in result.pages] == ["p1.png", "p2.png"]
        assert [p.source_page for p in result.pages] == [3, 4]
        assert result.pages[0].size == (833, 1111)  # the 300 x 400 pt page is page 3 of the file
        assert result.pages[1].size == (1654, 2339)

    def test_a_non_contiguous_range(self, env: Env) -> None:
        make_sized_pdf(env.raw / "b.pdf", [A4] * 5)
        result = env.ingest(doc("b.pdf", page_range="1,3-4"))
        assert [p.source_page for p in result.pages] == [1, 3, 4]

    def test_a_page_beyond_the_file_is_an_error(self, env: Env) -> None:
        make_sized_pdf(env.raw / "b.pdf", [A4] * 2)
        with pytest.raises(IngestError, match="page 3 of a 2-page file"):
            env.ingest(doc("b.pdf", page_range="2-3"))

    def test_page_range_on_an_image_is_an_error(self, env: Env) -> None:
        make_jpeg(env.raw / "p.jpg")
        with pytest.raises(IngestError, match="PDFs only"):
            env.ingest(doc("p.jpg", page_range="1"))

    def test_an_image_only_scan_pdf(self, env: Env) -> None:
        make_image_pdf(env.raw / "scan.pdf", pages=2)
        assert len(env.ingest(doc("scan.pdf")).pages) == 2

    def test_a_corrupt_pdf_is_an_error_not_a_crash(self, env: Env) -> None:
        (env.raw / "bad.pdf").write_bytes(b"%PDF-1.7 this is not really a pdf")
        with pytest.raises(IngestError, match="cannot open the PDF"):
            env.ingest(doc("bad.pdf"))


class TestImagePages:
    def test_a_large_photo_is_capped_and_stored_as_jpeg(self, env: Env) -> None:
        make_jpeg(env.raw / "photo.jpg", size=(4000, 3000))
        row = doc("photo.jpg")
        (page,) = env.ingest(row).pages
        assert page.size == (2400, 1800)
        assert page.format == "jpeg"
        assert page.file == "p1.jpg"
        assert env.page(row, "p1.jpg").format == "JPEG"

    def test_a_small_photo_is_not_enlarged(self, env: Env) -> None:
        make_jpeg(env.raw / "photo.jpg", size=(800, 600))
        assert env.ingest(doc("photo.jpg")).pages[0].size == (800, 600)

    def test_exif_orientation_is_applied(self, env: Env) -> None:
        # stored sideways (60 x 40) with orientation 6 = "rotate 90 degrees clockwise to view"
        make_jpeg(env.raw / "p.jpg", size=(60, 40), orientation=6, quadrants=True)
        row = doc("p.jpg")
        (page,) = env.ingest(row).pages
        assert page.size == (40, 60)  # upright: width and height swapped
        image = env.page(row, "p1.jpg")
        red_x, red_y = 36, 4  # the red top-left quarter moved to the top-right
        white_x, white_y = 4, 4
        r, g, b = image.getpixel((red_x, red_y))
        assert r > 200 and g < 60 and b < 60
        assert min(image.getpixel((white_x, white_y))) > 200

    @pytest.mark.parametrize("orientation", [1, 3, 6, 8])
    def test_the_stored_page_never_carries_an_orientation_tag(
        self, env: Env, orientation: int
    ) -> None:
        make_jpeg(env.raw / "p.jpg", orientation=orientation)
        row = doc("p.jpg")
        env.ingest(row)
        assert 0x0112 not in env.page(row, "p1.jpg").getexif()

    def test_all_jpeg_metadata_is_stripped(self, env: Env) -> None:
        make_jpeg(
            env.raw / "p.jpg",
            make="Apple",
            model="iPhone 12",
            gps=True,
            comment="secret-comment",
            orientation=1,
        )
        row = doc("p.jpg")
        env.ingest(row)
        raw = (env.out(row) / "p1.jpg").read_bytes()
        stored = Image.open(io.BytesIO(raw))
        assert len(stored.getexif()) == 0
        assert not stored.getexif().get_ifd(0x8825)
        assert "comment" not in stored.info
        assert "icc_profile" not in stored.info
        for needle in (b"Apple", b"iPhone", b"secret-comment", b"Exif"):
            assert needle not in raw

    def test_png_text_chunks_dpi_and_profiles_are_stripped(self, env: Env) -> None:
        make_png_with_metadata(env.raw / "s.png")
        row = doc("s.png")
        env.ingest(row)
        raw = (env.out(row) / "p1.png").read_bytes()
        stored = Image.open(io.BytesIO(raw))
        assert stored.info.get("dpi") is None
        assert "Comment" not in stored.info
        assert b"secret-comment" not in raw
        assert b"someone" not in raw
        assert b"tEXt" not in raw
        assert b"pHYs" not in raw

    def test_a_png_source_stays_png_and_lossless(self, env: Env) -> None:
        make_png(env.raw / "s.png", size=(50, 40), color=(10, 200, 30))
        row = doc("s.png")
        (page,) = env.ingest(row).pages
        assert page.format == "png"
        assert env.page(row, "p1.png").getpixel((7, 7)) == (10, 200, 30)

    def test_transparency_is_flattened_onto_white(self, env: Env) -> None:
        make_rgba_png(env.raw / "t.png")
        row = doc("t.png")
        env.ingest(row)
        assert env.page(row, "p1.png").getpixel((5, 5)) == (255, 255, 255)

    def test_webp_sources_are_stored_as_png(self, env: Env) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (30, 30), "white").save(buffer, format="WEBP", lossless=True)
        (env.raw / "w.webp").write_bytes(buffer.getvalue())
        assert env.ingest(doc("w.webp")).pages[0].format == "png"

    def test_a_real_heic_photo_becomes_a_jpeg(self, env: Env) -> None:
        pytest.importorskip("pillow_heif")
        make_heic(env.raw / "p.heic", size=(64, 48))
        row = doc("p.heic")
        (page,) = env.ingest(row).pages
        assert page.format == "jpeg"
        assert page.size == (64, 48)
        assert env.page(row, "p1.jpg").format == "JPEG"

    def test_jpeg_quality_is_honoured(self, env: Env) -> None:
        make_jpeg(env.raw / "p.jpg", size=(400, 300), color=(90, 140, 200))
        # noise makes the size depend on quality
        noisy = Image.effect_noise((400, 300), 60).convert("RGB")
        noisy.save(env.raw / "n.jpg", "JPEG", quality=95)
        high = env.ingest(doc("n.jpg"), settings=settings(jpeg_quality=95)).pages[0].bytes
        low = (
            env.ingest(doc("n.jpg"), settings=settings(jpeg_quality=40), force=True).pages[0].bytes
        )
        assert low < high

    def test_several_photos_become_pages_in_order(self, env: Env) -> None:
        for index, color in enumerate(["red", "green", "blue"]):
            make_jpeg(env.raw / f"{index}.jpg", size=(40 + index, 30), color=color)
        row = doc("0.jpg", "1.jpg", "2.jpg")
        result = env.ingest(row)
        assert [p.file for p in result.pages] == ["p1.jpg", "p2.jpg", "p3.jpg"]
        assert [p.source_file for p in result.pages] == [0, 1, 2]
        assert [p.size[0] for p in result.pages] == [40, 41, 42]

    def test_a_document_may_mix_pdf_and_photo_pages_each_stored_by_its_own_source(
        self, env: Env
    ) -> None:
        make_sized_pdf(env.raw / "a.pdf", [(200, 300)])
        make_jpeg(env.raw / "b.jpg")
        result = env.ingest(doc("a.pdf", "b.jpg"))
        assert [(p.file, p.format) for p in result.pages] == [("p1.png", "png"), ("p2.jpg", "jpeg")]

    def test_a_corrupt_image_is_an_error(self, env: Env) -> None:
        (env.raw / "bad.jpg").write_bytes(b"not an image")
        with pytest.raises(IngestError, match="cannot open the image"):
            env.ingest(doc("bad.jpg"))

    def test_a_missing_raw_file_is_an_error(self, env: Env) -> None:
        with pytest.raises(IngestError, match="raw file not found: gone.jpg"):
            env.ingest(doc("gone.jpg"))

    def test_a_raw_file_with_a_very_long_name(self, env: Env) -> None:
        name = "pdf-p-class-" + "truncated-text-" * 12 + "contoh.png"
        write_long_named(env.raw, name, make_png(env.raw / "tmp.png", size=(33, 22)).read_bytes())
        row = doc(name)
        assert env.ingest(row).pages[0].size == (33, 22)

    def test_a_variant_is_refused(self, env: Env) -> None:
        variant = ManifestRow.model_validate(
            {"doc_id": "v", "capture": "synthetic_degraded", "variant": "light", "gt_from": "d"}
        )
        with pytest.raises(IngestError, match="generated from its source"):
            env.ingest(variant)


# ------------------------------------------------------------------ determinism and the sidecar


class TestOutputsAndSidecar:
    def test_the_same_inputs_give_byte_identical_files(self, env: Env, tmp_path: Path) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4, (300, 400)])
        make_jpeg(env.raw / "p.jpg", size=(500, 400), color=(10, 90, 200))
        for row in (doc("a.pdf"), doc("p.jpg", doc_id="doc_002")):
            first = env.ingest(row)
            other = ingest_document(
                row,
                raw_dir=env.raw,
                out_dir=tmp_path / "second" / row.doc_id,
                settings=settings(),
            )
            for a, b in zip(first.pages, other.pages, strict=True):
                assert a.sha256 == b.sha256
                assert (env.out(row) / a.file).read_bytes() == (
                    tmp_path / "second" / row.doc_id / b.file
                ).read_bytes()

    def test_the_sidecar_records_sizes_hashes_and_the_page_mapping(self, env: Env) -> None:
        make_sized_pdf(env.raw / "b.pdf", [A4] * 3)
        row = doc("b.pdf", page_range="2-3")
        result = env.ingest(row)
        data = env.sidecar(row)
        assert data["ingest_version"] == INGEST_VERSION
        assert data["inputs_hash"] == result.inputs_hash
        assert data["settings"] == {"dpi": 200, "max_long_edge": 2400, "jpeg_quality": 92}
        assert data["page_range"] == "2-3"
        assert [p["source"] for p in data["pages"]] == [
            {"file": 0, "page": 2},
            {"file": 0, "page": 3},
        ]
        assert data["pages"][0]["size"] == [1654, 2339]
        assert len(data["pages"][0]["sha256"]) == 64
        assert data["pages"][0]["bytes"] == (env.out(row) / "p1.png").stat().st_size
        assert len(data["raw_sha256"]) == 1

    def test_the_sidecar_never_contains_a_raw_file_name(self, env: Env) -> None:
        make_png(env.raw / "Budi-Santoso-KTP.png")
        row = doc("Budi-Santoso-KTP.png")
        env.ingest(row)
        text = (env.out(row) / SIDECAR).read_text(encoding="utf-8")
        assert "Budi" not in text
        assert "KTP" not in text

    def test_no_temp_files_are_left_behind(self, env: Env) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4])
        row = doc("a.pdf")
        env.ingest(row)
        assert sorted(p.name for p in env.out(row).iterdir()) == [SIDECAR, "p1.png"]


# ------------------------------------------------------------------ idempotency


class TestIdempotency:
    def pdf_row(self, env: Env, pages: int = 3) -> ManifestRow:
        make_sized_pdf(env.raw / "a.pdf", [A4] * pages)
        return doc("a.pdf")

    def mtimes(self, env: Env, row: ManifestRow) -> dict[str, int]:
        return {p.name: p.stat().st_mtime_ns for p in env.out(row).iterdir()}

    def test_the_second_run_does_nothing(self, env: Env) -> None:
        row = self.pdf_row(env)
        first = env.ingest(row)
        before = self.mtimes(env, row)
        second = env.ingest(row)
        assert (first.status, second.status) == ("ingested", "unchanged")
        assert second.pages == first.pages
        assert self.mtimes(env, row) == before  # not one file was rewritten

    def test_a_changed_raw_file_is_re_ingested(self, env: Env) -> None:
        row = self.pdf_row(env, pages=1)
        env.ingest(row)
        make_sized_pdf(env.raw / "a.pdf", [(400, 500)])  # same name, different content
        assert env.ingest(row).status == "ingested"

    @pytest.mark.parametrize(
        "changed", [{"dpi": 150}, {"max_long_edge": 1000}, {"jpeg_quality": 70}]
    )
    def test_changed_settings_are_re_ingested(self, env: Env, changed: dict[str, int]) -> None:
        row = self.pdf_row(env, pages=1)
        env.ingest(row)
        assert env.ingest(row, settings=settings(**changed)).status == "ingested"

    def test_a_changed_page_range_is_re_ingested(self, env: Env) -> None:
        make_sized_pdf(env.raw / "b.pdf", [A4] * 3)
        env.ingest(doc("b.pdf", page_range="1-2"))
        assert env.ingest(doc("b.pdf", page_range="1-3")).status == "ingested"

    def test_shrinking_the_selection_removes_the_stale_pages(self, env: Env) -> None:
        make_sized_pdf(env.raw / "b.pdf", [A4] * 3)
        row3 = doc("b.pdf")
        env.ingest(row3)
        assert (env.out(row3) / "p3.png").exists()
        env.ingest(doc("b.pdf", page_range="1"))
        assert sorted(p.name for p in env.out(row3).iterdir()) == [SIDECAR, "p1.png"]

    def test_unrelated_files_in_the_folder_are_left_alone(self, env: Env) -> None:
        row = self.pdf_row(env, pages=1)
        env.ingest(row)
        (env.out(row) / "notes.txt").write_text("mine")
        env.ingest(row, force=True)
        assert (env.out(row) / "notes.txt").read_text() == "mine"

    def test_a_deleted_page_is_restored(self, env: Env) -> None:
        row = self.pdf_row(env, pages=2)
        env.ingest(row)
        (env.out(row) / "p2.png").unlink()
        assert env.ingest(row).status == "ingested"
        assert (env.out(row) / "p2.png").exists()

    def test_a_tampered_page_is_replaced(self, env: Env) -> None:
        row = self.pdf_row(env, pages=1)
        first = env.ingest(row)
        (env.out(row) / "p1.png").write_bytes(b"corrupted")
        assert env.ingest(row).status == "ingested"
        assert env.ingest(row, dry_run=True).status == "unchanged"
        assert env.sidecar(row)["pages"][0]["sha256"] == first.pages[0].sha256

    @pytest.mark.parametrize("breakage", ["missing", "garbage", "not-an-object", "other-version"])
    def test_a_damaged_sidecar_means_redo(self, env: Env, breakage: str) -> None:
        row = self.pdf_row(env, pages=1)
        env.ingest(row)
        path = env.out(row) / SIDECAR
        if breakage == "missing":
            path.unlink()
        elif breakage == "garbage":
            path.write_text("{not json")
        elif breakage == "not-an-object":
            path.write_text("[1, 2]")
        else:
            data = json.loads(path.read_text())
            data["ingest_version"] = INGEST_VERSION + 1
            path.write_text(json.dumps(data))
        assert env.ingest(row).status == "ingested"

    def test_force_re_ingests_even_when_nothing_changed(self, env: Env) -> None:
        row = self.pdf_row(env, pages=1)
        env.ingest(row)
        assert env.ingest(row, force=True).status == "ingested"

    def test_dry_run_writes_nothing(self, env: Env) -> None:
        row = self.pdf_row(env, pages=1)
        result = env.ingest(row, dry_run=True)
        assert result.status == "would_ingest"
        assert result.pages == ()
        assert not env.out(row).exists()

    def test_dry_run_on_an_up_to_date_document_says_unchanged(self, env: Env) -> None:
        row = self.pdf_row(env, pages=1)
        env.ingest(row)
        before = self.mtimes(env, row)
        assert env.ingest(row, dry_run=True).status == "unchanged"
        assert self.mtimes(env, row) == before

    def test_two_documents_do_not_interfere(self, env: Env) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4])
        make_jpeg(env.raw / "b.jpg")
        first, second = doc("a.pdf"), doc("b.jpg", doc_id="doc_002")
        env.ingest(first)
        env.ingest(second)
        assert env.ingest(first).status == "unchanged"
        assert env.ingest(second).status == "unchanged"


# ------------------------------------------------------------------ redaction


class TestRedactionDecision:
    def test_a_blank_decision_is_refused(self, env: Env) -> None:
        gray_png(env)
        with pytest.raises(IngestError, match="needs_redaction is blank"):
            env.ingest(doc("g.png", needs_redaction=""))

    def test_yes_without_boxes_is_refused_and_nothing_is_written(self, env: Env) -> None:
        gray_png(env)
        row = doc("g.png", needs_redaction="y")
        with pytest.raises(IngestError, match="refusing to store an unredacted page"):
            env.ingest(row)
        with pytest.raises(IngestError, match="refusing to store an unredacted page"):
            env.ingest(row, boxes={})
        with pytest.raises(IngestError, match="refusing to store an unredacted page"):
            env.ingest(row, boxes={1: []})
        assert not env.out(row).exists()

    def test_no_with_boxes_is_refused(self, env: Env) -> None:
        gray_png(env)
        with pytest.raises(IngestError, match="needs_redaction=n"):
            env.ingest(doc("g.png", needs_redaction="n"), boxes={1: [[0, 0, 5, 5]]})

    def test_a_box_beyond_the_page_is_refused(self, env: Env) -> None:
        gray_png(env)
        with pytest.raises(IngestError, match="does not fit the 80x60 page"):
            env.ingest(doc("g.png", needs_redaction="y"), boxes={1: [[70, 0, 20, 5]]})

    def test_boxes_for_a_page_the_document_does_not_have_are_refused(self, env: Env) -> None:
        gray_png(env)
        with pytest.raises(IngestError, match="boxes for page 2, but the document has 1 page"):
            env.ingest(doc("g.png", needs_redaction="y"), boxes={2: [[0, 0, 5, 5]]})


class TestRedactionIsBurnedIn:
    def test_the_box_is_solid_black_and_nothing_else_changes(self, env: Env) -> None:
        gray_png(env)
        row = doc("g.png", needs_redaction="y")
        env.ingest(row, boxes={1: [[10, 5, 20, 8]]})
        page = env.page(row, "p1.png")
        for x in range(10, 30):
            for y in range(5, 13):
                assert page.getpixel((x, y)) == (0, 0, 0), (x, y)
        black = sum(1 for p in pixels(page) if p == (0, 0, 0))
        assert black == 20 * 8  # exactly the box, no spill
        # the pixels just outside every edge are untouched
        for outside in [(9, 5), (30, 5), (10, 4), (10, 13), (30, 12), (29, 13), (0, 0), (79, 59)]:
            assert page.getpixel(outside) == (128, 128, 128), outside

    def test_several_boxes_and_only_the_boxed_page(self, env: Env) -> None:
        make_jpeg(env.raw / "a.jpg", size=(80, 60), color=(128, 128, 128))
        make_jpeg(env.raw / "b.jpg", size=(80, 60), color=(128, 128, 128))
        row = doc("a.jpg", "b.jpg", needs_redaction="y")
        env.ingest(row, boxes={2: [[0, 0, 10, 10], [60, 40, 20, 20]]})
        first, second = env.page(row, "p1.jpg"), env.page(row, "p2.jpg")
        assert min(first.getpixel((5, 5))) > 100  # page 1 has no box
        assert max(second.getpixel((5, 5))) < 10
        assert max(second.getpixel((70, 50))) < 10
        assert min(second.getpixel((40, 30))) > 100

    def test_what_was_under_the_box_is_gone_from_the_stored_file(self, env: Env) -> None:
        # incompressible noise: any trace of the original under the box would be obvious
        Image.effect_noise((120, 90), 80).convert("RGB").save(env.raw / "n.png", "PNG")
        row = doc("n.png", needs_redaction="y")
        env.ingest(row, boxes={1: [[20, 20, 50, 30]]})
        page = env.page(row, "p1.png")
        region = page.crop((20, 20, 70, 50))
        assert set(pixels(region)) == {(0, 0, 0)}

    def test_it_happens_before_encoding_for_jpeg_pages_too(self, env: Env) -> None:
        make_jpeg(env.raw / "p.jpg", size=(100, 80), color=(250, 250, 250))
        row = doc("p.jpg", needs_redaction="y")
        env.ingest(row, boxes={1: [[10, 10, 40, 30]]})
        page = env.page(row, "p1.jpg")
        assert max(page.getpixel((30, 25))) < 8  # solid black, not a faded artefact
        assert min(page.getpixel((80, 70))) > 240

    def test_pdf_pages_can_be_redacted(self, env: Env) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4])
        row = doc("a.pdf", needs_redaction="y")
        env.ingest(row, boxes={1: [[100, 100, 500, 200]]})
        page = env.page(row, "p1.png")
        assert page.getpixel((300, 200)) == (0, 0, 0)
        assert page.getpixel((50, 2000)) == (255, 255, 255)

    def test_redacting_a_resized_photo_uses_the_stored_pixel_space(self, env: Env) -> None:
        make_jpeg(env.raw / "big.jpg", size=(4000, 3000), color=(200, 200, 200))
        row = doc("big.jpg", needs_redaction="y")
        # (2300, 1700, 100, 100) fits the stored 2400 x 1800 page but not the 4000 x 3000 original
        result = env.ingest(row, boxes={1: [[2300, 1700, 100, 100]]})
        assert result.pages[0].size == (2400, 1800)
        assert max(env.page(row, "p1.jpg").getpixel((2350, 1750))) < 10

    def test_the_boxes_are_recorded_in_the_sidecar(self, env: Env) -> None:
        gray_png(env)
        row = doc("g.png", needs_redaction="y")
        env.ingest(row, boxes={1: [[10, 5, 20, 8]]})
        assert env.sidecar(row)["redactions"] == {"1": [[10, 5, 20, 8]]}

    def test_changing_a_box_re_ingests_and_float_spelling_does_not(self, env: Env) -> None:
        gray_png(env)
        row = doc("g.png", needs_redaction="y")
        assert env.ingest(row, boxes={1: [[10, 5, 20, 8]]}).status == "ingested"
        assert env.ingest(row, boxes={1: [[10, 5, 20, 8]]}).status == "unchanged"
        assert env.ingest(row, boxes={1: [[10.0, 5.0, 20.0, 8.0]]}).status == "unchanged"
        assert env.ingest(row, boxes={1: [[10, 5, 21, 8]]}).status == "ingested"


class TestRedactUnit:
    @pytest.mark.parametrize(
        "box",
        [
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            "abcd",
            [1, 2, "3", 4],
            [True, 1, 2, 2],
            [1.5, 0, 2, 2],
            [None, 0, 2, 2],
        ],
    )
    def test_malformed_boxes(self, box: Any) -> None:
        with pytest.raises(RedactionError):
            validate_boxes([box], (100, 100), where="d")

    @pytest.mark.parametrize(
        "box",
        [
            [0, 0, 0, 5],
            [0, 0, 5, 0],
            [0, 0, -1, 5],
            [-1, 0, 5, 5],
            [0, -1, 5, 5],
            [96, 0, 5, 5],
            [0, 96, 5, 5],
        ],
    )
    def test_boxes_that_do_not_fit(self, box: list[int]) -> None:
        with pytest.raises(RedactionError):
            validate_boxes([box], (100, 100), where="d")

    def test_a_box_exactly_filling_the_page_is_fine(self) -> None:
        assert validate_boxes([[0, 0, 100, 100]], (100, 100), where="d") == [(0, 0, 100, 100)]

    def test_whole_number_floats_are_accepted_and_converted(self) -> None:
        assert validate_boxes([[1.0, 2.0, 3.0, 4.0]], (100, 100), where="d") == [(1, 2, 3, 4)]

    def test_the_error_names_the_document_page_and_box(self) -> None:
        with pytest.raises(RedactionError, match=r"doc_007 page 2 box 2: \[90, 0, 20, 5\]"):
            validate_boxes([[0, 0, 5, 5], [90, 0, 20, 5]], (100, 100), where="doc_007 page 2")

    def test_apply_boxes_does_not_modify_its_input(self) -> None:
        image = Image.new("RGB", (20, 20), (200, 200, 200))
        before = image.tobytes()
        result = apply_boxes(image, [(2, 2, 5, 5)])
        assert image.tobytes() == before
        assert result is not image
        assert result.getpixel((3, 3)) == (0, 0, 0)

    def test_apply_boxes_always_returns_rgb(self) -> None:
        assert apply_boxes(Image.new("L", (10, 10), 200), [(0, 0, 2, 2)]).mode == "RGB"

    def test_the_covered_area_is_exactly_w_by_h(self) -> None:
        result = apply_boxes(Image.new("RGB", (50, 50), "white"), [(10, 20, 7, 3)])
        assert sum(1 for p in pixels(result) if p == (0, 0, 0)) == 21

    def test_no_boxes_is_a_plain_copy(self) -> None:
        image = Image.new("RGB", (10, 10), (1, 2, 3))
        assert apply_boxes(image, []).tobytes() == image.tobytes()


# ------------------------------------------------------------------ the command line


@pytest.mark.slow
class TestCommandLine:
    def run(self, *args: str) -> tuple[int, str, str]:
        return run_script("ingest.py", *args)

    def setup_dataset(self, tmp_path: Path, rows: list[str]) -> tuple[Path, Path, Path]:
        raw = tmp_path / "raw"
        raw.mkdir()
        make_sized_pdf(raw / "a.pdf", [A4])
        make_jpeg(raw / "p.jpg", size=(300, 200))
        header = ",".join(COLUMNS)
        manifest = tmp_path / "manifest.csv"
        manifest.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        return raw, manifest, tmp_path / "pages"

    def args(self, raw: Path, manifest: Path, pages: Path, *extra: str) -> list[str]:
        return [
            "--raw-dir",
            str(raw),
            "--manifest",
            str(manifest),
            "--pages-dir",
            str(pages),
            *extra,
        ]

    def test_it_ingests_then_skips_and_check_writes_nothing(self, tmp_path: Path) -> None:
        raw, manifest, pages = self.setup_dataset(
            tmp_path,
            ["doc_001,a.pdf,,,,,,,,,,,,n,,", "doc_002,p.jpg,,,,,,,,,,,,n,,"],
        )
        code, out, _ = self.run(*self.args(raw, manifest, pages, "--check"))
        assert code == 0
        assert "would_ingest" in out
        assert not pages.exists()
        code, out, _ = self.run(*self.args(raw, manifest, pages))
        assert code == 0
        assert "2 ingested, 0 unchanged" in out
        assert (pages / "doc_001" / "p1.png").is_file()
        assert (pages / "doc_002" / "p1.jpg").is_file()
        code, out, _ = self.run(*self.args(raw, manifest, pages))
        assert "0 ingested, 2 unchanged" in out

    def test_a_blank_redaction_decision_fails_that_document_only(self, tmp_path: Path) -> None:
        raw, manifest, pages = self.setup_dataset(
            tmp_path, ["doc_001,a.pdf,,,,,,,,,,,,,,", "doc_002,p.jpg,,,,,,,,,,,,n,,"]
        )
        code, out, _ = self.run(*self.args(raw, manifest, pages))
        assert code == 1
        assert "FAILED" in out
        assert "needs_redaction is blank" in out
        assert not (pages / "doc_001").exists()  # nothing unredacted was stored
        assert (pages / "doc_002" / "p1.jpg").is_file()  # and the other document went through

    def test_a_document_marked_y_is_refused_because_no_boxes_exist_yet(
        self, tmp_path: Path
    ) -> None:
        raw, manifest, pages = self.setup_dataset(tmp_path, ["doc_001,a.pdf,,,,,,,,,,,,y,,"])
        code, out, _ = self.run(*self.args(raw, manifest, pages))
        assert code == 1
        assert "refusing to store an unredacted page" in out

    def test_doc_filter_and_unknown_doc(self, tmp_path: Path) -> None:
        raw, manifest, pages = self.setup_dataset(
            tmp_path, ["doc_001,a.pdf,,,,,,,,,,,,n,,", "doc_002,p.jpg,,,,,,,,,,,,n,,"]
        )
        code, out, _ = self.run(*self.args(raw, manifest, pages, "--doc", "doc_002"))
        assert code == 0
        assert not (pages / "doc_001").exists()
        assert (pages / "doc_002").exists()
        code, _, err = self.run(*self.args(raw, manifest, pages, "--doc", "nope"))
        assert code == 2
        assert "Not in the manifest: nope" in err

    def test_an_unreadable_manifest_exits_2(self, tmp_path: Path) -> None:
        raw, manifest, pages = self.setup_dataset(tmp_path, ["doc_001,a.pdf,,,,,,,,,,,,n,,"])
        manifest.write_text("garbage\n", encoding="utf-8")
        code, _, err = self.run(*self.args(raw, manifest, pages))
        assert code == 2
        assert "cannot be read" in err

    def test_dpi_and_cap_flags(self, tmp_path: Path) -> None:
        raw, manifest, pages = self.setup_dataset(tmp_path, ["doc_001,a.pdf,,,,,,,,,,,,n,,"])
        code, out, _ = self.run(*self.args(raw, manifest, pages, "--dpi", "100"))
        assert code == 0
        assert "827x1169" in out
        shutil.rmtree(pages)
        code, out, _ = self.run(
            *self.args(raw, manifest, pages, "--dpi", "300", "--max-long-edge", "1000")
        )
        assert code == 0
        assert max(Image.open(pages / "doc_001" / "p1.png").size) == 1000
