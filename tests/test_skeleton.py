import hashlib
import os
from pathlib import Path

import pytest

from docextract.fsutil import long_path
from docextract.ingest.manifest import ManifestRow, read_manifest, write_manifest
from docextract.ingest.probe import guess_language, probe_file, sniff_format
from docextract.ingest.skeleton import (
    HAND_FILLED,
    build_skeleton,
    format_report,
    hints_for,
    scan_raw,
)
from files import make_image_pdf, make_jpeg, make_png, make_text_pdf

INDONESIAN = (
    "Faktur untuk pelanggan dengan nomor barang yang dibayar tidak dari tanggal alamat "
    "nama jumlah harga pembayaran kepada terima kasih dan pajak sebesar itu atau ini"
)
ENGLISH = (
    "Thank you for your booking. The confirmation number and the payment for this invoice "
    "are with you. Your name and address are on the receipt, and the amount and the tax are due."
)


# ------------------------------------------------------------------ sniffing and probing


class TestSniffFormat:
    @pytest.mark.parametrize(
        ("head", "expected"),
        [
            (b"%PDF-1.7\n", "pdf"),
            (b"\x89PNG\r\n\x1a\n....", "png"),
            (b"\xff\xd8\xff\xe0\x00\x10JFIF", "jpeg"),
            (b"\xff\xd8\xff\xe1xxxxExif", "jpeg"),
            (b"\x00\x00\x00\x18ftypheic....", "heic"),
            (b"\x00\x00\x00\x18ftypmif1....", "heic"),
            (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "webp"),
            (b"II*\x00....", "tiff"),
            (b"MM\x00*....", "tiff"),
            (b"BMxxxxxxxx", "bmp"),
            (b"", None),
            (b"plain text, not an image", None),
            (b"\x00\x00\x00\x18ftypisom....", None),  # an mp4, not an image
        ],
    )
    def test_format_comes_from_the_bytes(self, head: bytes, expected: str | None) -> None:
        assert sniff_format(head) == expected


class TestGuessLanguage:
    def test_indonesian(self) -> None:
        assert guess_language(INDONESIAN) == "id"

    def test_english(self) -> None:
        assert guess_language(ENGLISH) == "en"

    def test_mixed_text_is_not_guessed(self) -> None:
        assert guess_language(INDONESIAN + " " + ENGLISH) is None

    def test_a_language_that_is_only_slightly_ahead_is_not_guessed(self) -> None:
        # 6 Indonesian stopwords against 4 English ones: ahead, but not decisively
        assert guess_language("yang dan untuk dengan tidak dari the and for with") is None
        # and the same the other way round: 6 English against 4 Indonesian
        assert guess_language("the and for with not from yang dan untuk dengan") is None

    def test_twice_as_many_hits_is_decisive(self) -> None:
        assert guess_language("yang dan untuk dengan tidak dari pada ini the and") == "id"
        assert guess_language("the and for with not from this that yang dan") == "en"

    def test_too_little_text_is_not_guessed(self) -> None:
        assert guess_language("Faktur dan pajak") is None
        assert guess_language("") is None

    def test_numbers_and_names_alone_say_nothing(self) -> None:
        assert guess_language("PT Sumber Makmur 12.345.678 Jl. Merdeka 17 Jakarta") is None


class TestProbeFile:
    def test_a_text_pdf_has_a_text_layer_and_a_language(self, tmp_path: Path) -> None:
        probe = probe_file(make_text_pdf(tmp_path / "a.pdf", INDONESIAN, pages=2))
        assert probe.content_format == "pdf"
        assert probe.page_count == 2
        assert probe.has_text_layer
        assert probe.language == "id"
        assert probe.error is None

    def test_an_image_only_pdf_has_no_text_layer(self, tmp_path: Path) -> None:
        probe = probe_file(make_image_pdf(tmp_path / "scan.pdf", pages=3))
        assert probe.page_count == 3
        assert not probe.has_text_layer
        assert probe.language is None  # no text, nothing to score
        assert probe.text_chars == 0

    def test_a_tiny_text_layer_does_not_count(self, tmp_path: Path) -> None:
        probe = probe_file(make_text_pdf(tmp_path / "tiny.pdf", "Hi"))
        assert not probe.has_text_layer
        assert probe.language is None

    def test_a_phone_photo_has_camera_exif_gps_and_orientation(self, tmp_path: Path) -> None:
        path = make_jpeg(
            tmp_path / "p.jpg", make="Apple", model="iPhone 12", orientation=6, gps=True
        )
        probe = probe_file(path)
        assert probe.content_format == "jpeg"
        assert probe.has_camera_exif
        assert probe.has_gps
        assert probe.exif_orientation == 6
        assert probe.image_size == (60, 40)  # as stored, before any EXIF rotation
        assert probe.page_count == 1

    def test_a_stripped_jpeg_has_no_camera_exif(self, tmp_path: Path) -> None:
        probe = probe_file(make_jpeg(tmp_path / "wa.jpeg"))
        assert not probe.has_camera_exif
        assert not probe.has_gps
        assert probe.exif_orientation is None

    def test_a_png(self, tmp_path: Path) -> None:
        probe = probe_file(make_png(tmp_path / "s.png"))
        assert probe.content_format == "png"
        assert probe.image_size == (30, 20)
        assert not probe.has_camera_exif

    def test_sha256_and_size_match_the_bytes(self, tmp_path: Path) -> None:
        path = make_png(tmp_path / "s.png")
        probe = probe_file(path)
        assert probe.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert probe.size_bytes == path.stat().st_size

    def test_the_name_defaults_to_the_file_name_and_can_be_overridden(self, tmp_path: Path) -> None:
        path = make_png(tmp_path / "s.png")
        assert probe_file(path).name == "s.png"
        assert probe_file(path, name="sub/s.png").name == "sub/s.png"

    def test_a_jfif_extension_on_a_jpeg_is_not_a_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.jfif"
        path.write_bytes(make_jpeg(tmp_path / "tmp.jpg").read_bytes())
        probe = probe_file(path)
        assert probe.extension == "jfif"
        assert probe.content_format == "jpeg"
        assert not probe.extension_mismatch
        assert probe.error is None

    def test_a_png_named_jpg_is_a_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "liar.jpg"
        path.write_bytes(make_png(tmp_path / "tmp.png").read_bytes())
        probe = probe_file(path)
        assert probe.extension_mismatch
        assert probe.error is None  # it is still readable
        assert any("bytes are png" in hint for hint in hints_for("d", probe))

    @pytest.mark.parametrize("extension", ["jpg", "jpeg", "png", "pdf"])
    def test_a_corrupt_file_reports_an_error_instead_of_raising(
        self, tmp_path: Path, extension: str
    ) -> None:
        path = tmp_path / f"broken.{extension}"
        path.write_bytes(b"this is not a real file at all")
        probe = probe_file(path)
        assert probe.error
        assert probe.page_count is None

    def test_a_truncated_pdf_reports_an_error(self, tmp_path: Path) -> None:
        good = make_text_pdf(tmp_path / "ok.pdf", INDONESIAN).read_bytes()
        bad = tmp_path / "cut.pdf"
        bad.write_bytes(good[: len(good) // 3])
        assert probe_file(bad).error

    def test_an_empty_file_reports_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        assert probe_file(path).error

    def test_a_text_file_named_pdf_says_so(self, tmp_path: Path) -> None:
        path = tmp_path / "fake.pdf"
        path.write_text("hello")
        assert "not a PDF" in (probe_file(path).error or "")


# ------------------------------------------------------------------ scanning


class TestVeryLongPaths:
    """A web page saved as PDF can have a 200-character name. On Windows the combined path then
    exceeds 260 characters, and a plain Path.is_file() quietly returns False: the document used
    to vanish from the scan with no error. It must be found, probed and listed like any other."""

    def make_long(self, tmp_path: Path) -> tuple[Path, str]:
        raw = tmp_path / "raw"
        raw.mkdir()
        name = "pdf-p-class-" + "truncated-text-" * 12 + "contoh-dan-jenis-faktur.png"
        source = make_png(tmp_path / "short.png")
        long_path(raw / name).write_bytes(source.read_bytes())
        if os.name == "nt":  # the premise: this really is over the classic limit
            assert len(str(raw / name)) > 260
        return raw, name

    def test_the_scan_finds_it(self, tmp_path: Path) -> None:
        raw, name = self.make_long(tmp_path)
        make_png(raw / "ordinary.png")
        found, ignored = scan_raw(raw)
        assert sorted(found) == sorted([name, "ordinary.png"])
        assert ignored == []

    def test_it_is_probed_and_gets_a_row_like_any_other_file(self, tmp_path: Path) -> None:
        raw, name = self.make_long(tmp_path)
        result = build_skeleton(raw, [])
        assert [row.raw_files for row in result.rows] == [(name,)]
        assert result.unreadable == []
        probe = result.probes["doc_001"]
        assert probe.content_format == "png"
        assert probe.image_size == (30, 20)
        assert probe.error is None

    def test_a_second_run_does_not_add_it_again(self, tmp_path: Path) -> None:
        raw, _ = self.make_long(tmp_path)
        first = build_skeleton(raw, [])
        assert build_skeleton(raw, first.rows).new_ids == []

    def test_a_row_for_a_long_named_file_is_not_reported_missing(self, tmp_path: Path) -> None:
        raw, name = self.make_long(tmp_path)
        existing = [ManifestRow(doc_id="doc_001", raw_files=(name,))]
        assert build_skeleton(raw, existing).missing == []


class TestLongPathHelper:
    def test_the_result_is_absolute_and_idempotent(self, tmp_path: Path) -> None:
        once = long_path(tmp_path / "a.pdf")
        assert once.is_absolute()
        assert long_path(once) == once

    @pytest.mark.skipif(os.name != "nt", reason="the extended-length prefix is Windows only")
    def test_windows_paths_get_the_extended_prefix(self, tmp_path: Path) -> None:
        assert str(long_path(tmp_path / "a.pdf")).startswith("\\\\?\\")

    @pytest.mark.skipif(os.name == "nt", reason="paths are only rewritten on Windows")
    def test_other_platforms_are_untouched(self, tmp_path: Path) -> None:
        assert long_path(tmp_path / "a.pdf") == tmp_path / "a.pdf"

    def test_a_file_can_be_written_and_read_through_it(self, tmp_path: Path) -> None:
        target = long_path(tmp_path / "x.bin")
        target.write_bytes(b"abc")
        assert target.read_bytes() == b"abc"
        assert (tmp_path / "x.bin").read_bytes() == b"abc"


class TestScanRaw:
    def test_a_missing_folder_is_just_empty(self, tmp_path: Path) -> None:
        assert scan_raw(tmp_path / "nope") == ([], [])

    def test_supported_files_are_found_recursively_in_natural_order(self, tmp_path: Path) -> None:
        for name in ("10.png", "2.png", "1.png", "sub/3.png"):
            make_png(tmp_path / name)
        found, ignored = scan_raw(tmp_path)
        assert found == ["1.png", "2.png", "10.png", "sub/3.png"]
        assert ignored == []

    def test_housekeeping_files_and_the_constructed_folder_are_skipped_quietly(
        self, tmp_path: Path
    ) -> None:
        make_png(tmp_path / "real.png")
        make_png(tmp_path / "constructed" / "inv1.png")  # rows for these come from their script
        for name in (".gitkeep", "Thumbs.db", "desktop.ini", ".DS_Store", "~$lock.pdf"):
            (tmp_path / name).write_bytes(b"x")
        found, ignored = scan_raw(tmp_path)
        assert found == ["real.png"]
        assert ignored == []

    def test_unsupported_extensions_are_reported_not_silently_dropped(self, tmp_path: Path) -> None:
        make_png(tmp_path / "ok.png")
        (tmp_path / "notes.txt").write_text("hi")
        (tmp_path / "noext").write_text("hi")
        _, ignored = scan_raw(tmp_path)
        assert [name for name, _ in ignored] == ["noext", "notes.txt"]
        assert "unsupported extension" in ignored[1][1]

    def test_a_name_with_a_semicolon_is_reported_because_the_manifest_cannot_hold_it(
        self, tmp_path: Path
    ) -> None:
        make_png(tmp_path / "a;b.png")
        found, ignored = scan_raw(tmp_path)
        assert found == []
        assert "contains ';'" in ignored[0][1]

    def test_the_extension_check_is_case_insensitive(self, tmp_path: Path) -> None:
        make_png(tmp_path / "SCAN.PNG")
        assert scan_raw(tmp_path)[0] == ["SCAN.PNG"]


# ------------------------------------------------------------------ the skeleton


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    folder = tmp_path / "raw"
    make_text_pdf(folder / "faktur.pdf", INDONESIAN, pages=2)
    make_image_pdf(folder / "scan.pdf", pages=2)
    make_jpeg(folder / "photo.jpg", make="Apple", model="iPhone", gps=True)
    make_jpeg(folder / "WhatsApp Image 2026-10-06 at 12.36.39.jpeg")
    make_png(folder / "screenshot.png")
    return folder


class TestBuildSkeleton:
    def test_one_row_per_file_with_sequential_neutral_ids(self, raw: Path) -> None:
        result = build_skeleton(raw, [])
        assert result.new_ids == [f"doc_{n:03d}" for n in range(1, 6)]
        assert [row.raw_files for row in result.rows] == [
            ("faktur.pdf",),
            ("photo.jpg",),
            ("scan.pdf",),
            ("screenshot.png",),
            ("WhatsApp Image 2026-10-06 at 12.36.39.jpeg",),
        ]

    def test_only_strong_evidence_is_filled_in(self, raw: Path) -> None:
        rows = {row.raw_files[0]: row for row in build_skeleton(raw, []).rows}
        assert rows["faktur.pdf"].capture == "born_digital"  # a real text layer
        assert rows["faktur.pdf"].language == "id"
        assert rows["photo.jpg"].capture == "phone_photo"  # camera Make/Model in EXIF
        assert rows["scan.pdf"].capture is None  # image-only PDF: scan or screenshot?
        assert rows["screenshot.png"].capture is None
        assert rows["WhatsApp Image 2026-10-06 at 12.36.39.jpeg"].capture is None  # EXIF stripped
        assert rows["scan.pdf"].language is None
        assert rows["photo.jpg"].language is None  # no OCR: images get no language guess

    def test_what_only_a_human_can_decide_is_always_blank(self, raw: Path) -> None:
        for row in build_skeleton(raw, []).rows:
            for column in HAND_FILLED:
                assert getattr(row, column) is None, column

    def test_original_format_is_the_extension_and_content_group_defaults_to_the_doc_id(
        self, raw: Path
    ) -> None:
        for row in build_skeleton(raw, []).rows:
            assert row.original_format == Path(row.raw_files[0]).suffix.lstrip(".").lower()
            assert row.content_group == row.doc_id

    def test_the_skeleton_survives_a_csv_round_trip(self, raw: Path, tmp_path: Path) -> None:
        result = build_skeleton(raw, [])
        write_manifest(tmp_path / "manifest.csv", result.rows)
        assert read_manifest(tmp_path / "manifest.csv") == result.rows

    def test_rerunning_changes_nothing(self, raw: Path) -> None:
        first = build_skeleton(raw, [])
        second = build_skeleton(raw, first.rows)
        assert second.new_ids == []
        assert second.rows == first.rows

    def test_hand_edits_are_never_overwritten(self, raw: Path) -> None:
        first = build_skeleton(raw, [])
        edited = [
            row.model_copy(update={"doc_type": "faktur", "needs_redaction": True, "language": "en"})
            for row in first.rows
        ]
        again = build_skeleton(raw, edited)
        assert again.rows == edited

    def test_new_files_are_appended_with_the_next_ids(self, raw: Path) -> None:
        first = build_skeleton(raw, [])
        make_png(raw / "zzz-new.png")
        second = build_skeleton(raw, first.rows)
        assert second.new_ids == ["doc_006"]
        assert second.rows[:5] == first.rows
        assert second.rows[5].raw_files == ("zzz-new.png",)

    def test_numbering_continues_after_the_highest_even_with_gaps_and_renames(
        self, raw: Path
    ) -> None:
        existing = [
            ManifestRow(doc_id="doc_007", raw_files=("faktur.pdf",)),
            ManifestRow(doc_id="my_receipt", raw_files=("photo.jpg",)),  # a hand-chosen id
        ]
        result = build_skeleton(raw, existing)
        assert result.new_ids == ["doc_008", "doc_009", "doc_010"]

    def test_id_width_grows_instead_of_colliding(self, raw: Path) -> None:
        existing = [ManifestRow(doc_id="doc_999", raw_files=("faktur.pdf",))]
        assert build_skeleton(raw, existing).new_ids[0] == "doc_1000"

    def test_a_file_referenced_by_two_page_range_rows_is_not_added_again(self, raw: Path) -> None:
        existing = [
            ManifestRow(doc_id="doc_001", raw_files=("scan.pdf",), page_range="1"),
            ManifestRow(doc_id="doc_002", raw_files=("scan.pdf",), page_range="2"),
        ]
        result = build_skeleton(raw, existing)
        assert all(
            row.raw_files != ("scan.pdf",) or row.doc_id in ("doc_001", "doc_002")
            for row in result.rows
        )
        assert sum(1 for row in result.rows if "scan.pdf" in row.raw_files) == 2

    def test_a_row_whose_file_vanished_is_reported_but_kept(self, raw: Path) -> None:
        first = build_skeleton(raw, [])
        (raw / "photo.jpg").unlink()
        second = build_skeleton(raw, first.rows)
        assert second.missing == [("doc_002", "photo.jpg")]
        assert len(second.rows) == len(first.rows)

    def test_a_multi_file_row_reports_only_the_missing_part(self, raw: Path) -> None:
        existing = [ManifestRow(doc_id="doc_001", raw_files=("photo.jpg", "gone.jpg"))]
        assert build_skeleton(raw, existing).missing == [("doc_001", "gone.jpg")]

    def test_an_unreadable_file_still_gets_a_row_and_is_flagged(self, raw: Path) -> None:
        (raw / "broken.jpg").write_bytes(b"garbage")
        result = build_skeleton(raw, [])
        assert [name for name, _ in result.unreadable] == ["broken.jpg"]
        assert any(row.raw_files == ("broken.jpg",) for row in result.rows)

    def test_byte_identical_files_are_flagged_as_probable_duplicates(self, raw: Path) -> None:
        (raw / "copy.png").write_bytes((raw / "screenshot.png").read_bytes())
        result = build_skeleton(raw, [])
        assert len(result.duplicates) == 1
        assert "byte-identical" in format_report(result, raw, raw / "m.csv", dry_run=False)

    def test_probes_are_kept_for_new_files_only(self, raw: Path) -> None:
        first = build_skeleton(raw, [])
        assert set(first.probes) == set(first.new_ids)
        assert build_skeleton(raw, first.rows).probes == {}


class TestReport:
    def test_it_lists_every_new_file_with_format_pages_and_what_was_inferred(
        self, raw: Path
    ) -> None:
        result = build_skeleton(raw, [])
        report = format_report(result, raw, raw / "manifest.csv", dry_run=False)
        assert "added 5 new" in report
        assert "doc_001" in report
        assert "capture=born_digital, language=id" in report
        assert "capture=phone_photo" in report
        assert "(nothing inferred)" in report

    def test_it_counts_what_could_not_be_inferred(self, raw: Path) -> None:
        report = format_report(build_skeleton(raw, []), raw, raw / "m.csv", dry_run=False)
        assert "Could not infer" in report
        for column in HAND_FILLED:
            assert f"{column:<16} 5 of 5" in report
        assert "capture          3 of 5" in report  # three of the five stay blank
        assert "language         4 of 5" in report

    def test_hints_flag_bundles_scans_missing_exif_and_gps(self, raw: Path) -> None:
        report = format_report(build_skeleton(raw, []), raw, raw / "m.csv", dry_run=False)
        assert "2 pages: one document, or a bundle" in report
        assert "image-only PDF" in report
        assert "no camera EXIF" in report
        assert "contains GPS metadata" in report
        assert "looks like a WhatsApp export" in report

    def test_missing_ignored_and_unreadable_are_called_out(self, raw: Path) -> None:
        existing = [ManifestRow(doc_id="doc_001", raw_files=("gone.pdf",))]
        (raw / "notes.txt").write_text("x")
        (raw / "broken.png").write_bytes(b"junk")
        report = format_report(build_skeleton(raw, existing), raw, raw / "m.csv", dry_run=False)
        assert "MISSING: doc_001 refers to 'gone.pdf'" in report
        assert "IGNORED: notes.txt" in report
        assert "COULD NOT READ" in report

    def test_dry_run_says_so(self, raw: Path) -> None:
        assert "dry run: not written" in format_report(
            build_skeleton(raw, []), raw, raw / "m.csv", dry_run=True
        )

    def test_a_rerun_reports_nothing_new_and_no_inference_table(self, raw: Path) -> None:
        first = build_skeleton(raw, [])
        report = format_report(build_skeleton(raw, first.rows), raw, raw / "m.csv", dry_run=False)
        assert "added 0 new" in report
        assert "Could not infer" not in report

    def test_long_file_names_are_shortened_in_the_table(self, raw: Path) -> None:
        make_png(raw / ("x" * 80 + ".png"))
        assert "..." in format_report(build_skeleton(raw, []), raw, raw / "m.csv", dry_run=False)


# ------------------------------------------------------------------ the command line


@pytest.mark.slow
class TestCommandLine:
    def run(self, *args: str) -> tuple[int, str, str]:
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        done = subprocess.run(
            [sys.executable, str(root / "scripts" / "manifest_init.py"), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        return done.returncode, done.stdout, done.stderr

    def test_it_writes_the_manifest_and_a_rerun_adds_nothing(
        self, raw: Path, tmp_path: Path
    ) -> None:
        manifest = tmp_path / "data" / "manifest.csv"
        code, out, _ = self.run("--raw-dir", str(raw), "--manifest", str(manifest))
        assert code == 0
        assert "added 5 new" in out
        first = manifest.read_text(encoding="utf-8")
        code, out, _ = self.run("--raw-dir", str(raw), "--manifest", str(manifest))
        assert code == 0
        assert "added 0 new" in out
        assert manifest.read_text(encoding="utf-8") == first

    def test_dry_run_writes_nothing(self, raw: Path, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.csv"
        code, out, _ = self.run("--raw-dir", str(raw), "--manifest", str(manifest), "--dry-run")
        assert code == 0
        assert "dry run" in out
        assert not manifest.exists()

    def test_an_unreadable_manifest_aborts_without_touching_it(
        self, raw: Path, tmp_path: Path
    ) -> None:
        manifest = tmp_path / "manifest.csv"
        manifest.write_text("doc_id,raw_files\nbad id,a.pdf\n", encoding="utf-8")
        before = manifest.read_bytes()
        code, _, err = self.run("--raw-dir", str(raw), "--manifest", str(manifest))
        assert code == 2
        assert "cannot be read" in err
        assert manifest.read_bytes() == before
