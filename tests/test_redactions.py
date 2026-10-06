from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from docextract.ingest.pipeline import SIDECAR, IngestError, check_pinned_sizes
from docextract.ingest.preview import PreviewError, check_preview_dir, preview_document
from docextract.ingest.redact import RedactionError, parse_box
from docextract.ingest.redactions import (
    PageRedaction,
    boxes_of,
    check_against_manifest,
    parse_redactions,
    pinned_sizes_of,
    read_redactions,
)
from files import A4, make_jpeg, make_sized_pdf
from ingest_support import (
    Env,
    doc,
    gray_png,
    run_script,
    settings,
    write_manifest_file,
)

# ------------------------------------------------------------------ parsing the file


class TestParse:
    def test_pages_with_lists_of_boxes(self) -> None:
        parsed = parse_redactions(
            "doc_007:\n"
            "  1:\n    - [120, 340, 400, 60]\n    - [1, 2, 3, 4]\n"
            "  2:\n    - [10, 10, 5, 5]\n"
        )
        assert parsed == {
            "doc_007": {
                1: PageRedaction(((120, 340, 400, 60), (1, 2, 3, 4))),
                2: PageRedaction(((10, 10, 5, 5),)),
            }
        }

    def test_a_page_with_a_pinned_size(self) -> None:
        parsed = parse_redactions(
            "doc_007:\n  2:\n    size: [1654, 2339]\n    boxes:\n      - [10, 10, 50, 50]\n"
        )
        assert parsed["doc_007"][2] == PageRedaction(((10, 10, 50, 50),), (1654, 2339))

    def test_a_mapping_page_without_size_is_unpinned(self) -> None:
        parsed = parse_redactions("d:\n  1:\n    boxes: [[1, 1, 2, 2]]\n")
        assert parsed["d"][1].size is None
        assert (
            parse_redactions("d:\n  1:\n    size:\n    boxes: [[1, 1, 2, 2]]\n")["d"][1].size
            is None
        )

    def test_page_numbers_may_be_quoted(self) -> None:
        assert set(parse_redactions('d:\n  "3": [[0, 0, 1, 1]]\n  4: [[0, 0, 1, 1]]\n')["d"]) == {
            3,
            4,
        }

    def test_whole_number_floats_become_integers(self) -> None:
        parsed = parse_redactions("d:\n  1: [[10.0, 20.0, 30.0, 40.0]]\n")
        assert parsed["d"][1].boxes == ((10, 20, 30, 40),)
        assert all(type(v) is int for v in parsed["d"][1].boxes[0])

    @pytest.mark.parametrize("text", ["", "\n", "# only a comment\n", "---\n", "{}\n"])
    def test_an_empty_file_means_no_redactions(self, text: str) -> None:
        assert parse_redactions(text) == {}

    def test_several_documents(self) -> None:
        parsed = parse_redactions("a:\n  1: [[0, 0, 1, 1]]\nb:\n  2: [[0, 0, 2, 2]]\n")
        assert list(parsed) == ["a", "b"]

    def test_yaml_aliases_and_merge_keys_are_not_mistaken_for_duplicates(self) -> None:
        text = "a: &pages\n  1: [[0, 0, 1, 1]]\nb:\n  <<: *pages\n  2: [[0, 0, 2, 2]]\n"
        parsed = parse_redactions(text)
        assert set(parsed["b"]) == {1, 2}

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("- a\n- b\n", "top level must map"),
            ("just text\n", "top level must map"),
            ("7:\n  1: [[0, 0, 1, 1]]\n", "not a doc id"),
            ("d:\n", "expected page numbers"),
            ("d: {}\n", "expected page numbers"),
            ("d: [[0, 0, 1, 1]]\n", "expected page numbers"),
            ("d: 5\n", "expected page numbers"),
            ("d:\n  0: [[0, 0, 1, 1]]\n", "start at 1"),
            ("d:\n  -1: [[0, 0, 1, 1]]\n", "start at 1"),
            ("d:\n  abc: [[0, 0, 1, 1]]\n", "not a page number"),
            ("d:\n  1.5: [[0, 0, 1, 1]]\n", "not a page number"),
            ("d:\n  true: [[0, 0, 1, 1]]\n", "not a page number"),
            ('d:\n  "": [[0, 0, 1, 1]]\n', "not a page number"),
            ("d:\n  1: []\n", "non-empty list"),
            ("d:\n  1:\n", "non-empty list"),
            ("d:\n  1: 5\n", "non-empty list"),
            ("d:\n  1: text\n", "non-empty list"),
            ("d:\n  1:\n    boxes: []\n", "non-empty list"),
            ("d:\n  1:\n    box: [[0, 0, 1, 1]]\n", "unknown key"),
            ("d:\n  1:\n    boxes: [[0, 0, 1, 1]]\n    note: x\n", "unknown key"),
            ("d:\n  1:\n    size: [10, 10]\n", "needs a 'boxes' list"),
            ("d:\n  1:\n    size: [10]\n    boxes: [[0, 0, 1, 1]]\n", "size must be"),
            ("d:\n  1:\n    size: [0, 10]\n    boxes: [[0, 0, 1, 1]]\n", "size must be positive"),
            ("d:\n  1:\n    size: [10.5, 10]\n    boxes: [[0, 0, 1, 1]]\n", "whole number"),
            ("d:\n  1:\n    size: wide\n    boxes: [[0, 0, 1, 1]]\n", "size must be"),
            ("d:\n  1: [[1, 2, 3]]\n", "expected \\[x, y, w, h\\]"),
            ("d:\n  1: [[1, 2, 3, 4, 5]]\n", "expected \\[x, y, w, h\\]"),
            ("d:\n  1: [5]\n", "expected \\[x, y, w, h\\]"),
            ("d:\n  1: [abcd]\n", "expected \\[x, y, w, h\\]"),
            ("d:\n  1: [[1, 2, 3, '4']]\n", "not a number"),
            ("d:\n  1: [[true, 2, 3, 4]]\n", "not a number"),
            ("d:\n  1: [[1, 2, 3, null]]\n", "not a number"),
            ("d:\n  1: [[1, 2, 3.5, 4]]\n", "whole number"),
            ("d:\n  1: [[1, 2, 0, 4]]\n", "must be positive"),
            ("d:\n  1: [[1, 2, 3, -4]]\n", "must be positive"),
            ("d:\n  1: [[-1, 2, 3, 4]]\n", "cannot be negative"),
        ],
    )
    def test_malformed_files_are_refused(self, text: str, message: str) -> None:
        with pytest.raises(RedactionError, match=message):
            parse_redactions(text)

    def test_a_repeated_document_is_refused_not_silently_overwritten(self) -> None:
        text = "d:\n  1: [[0, 0, 1, 1]]\nd:\n  1: [[5, 5, 1, 1]]\n"
        with pytest.raises(
            RedactionError, match=r"redactions.yaml: line 3: the key 'd' appears twice"
        ):
            parse_redactions(text)

    def test_a_repeated_page_is_refused_not_silently_overwritten(self) -> None:
        # PyYAML alone would keep only the last `1:` and the first page's boxes would vanish
        text = "d:\n  1: [[0, 0, 1, 1]]\n  2: [[0, 0, 1, 1]]\n  1: [[5, 5, 1, 1]]\n"
        with pytest.raises(RedactionError, match=r"line 4: the key 1 appears twice"):
            parse_redactions(text)

    def test_a_repeated_key_inside_a_page_mapping_is_refused(self) -> None:
        text = "d:\n  1:\n    boxes: [[0, 0, 1, 1]]\n    boxes: [[5, 5, 1, 1]]\n"
        with pytest.raises(RedactionError, match="appears twice"):
            parse_redactions(text)

    def test_page_1_and_page_quoted_1_are_the_same_page(self) -> None:
        with pytest.raises(RedactionError, match="page 1 is listed twice"):
            parse_redactions('d:\n  1: [[0, 0, 1, 1]]\n  "1": [[5, 5, 1, 1]]\n')

    def test_yaml_syntax_errors_say_where(self) -> None:
        with pytest.raises(RedactionError, match=r"redactions.yaml: line \d+: "):
            parse_redactions("d:\n  1: [[0, 0, 1, 1]\n")

    def test_the_error_names_the_document_page_and_box(self) -> None:
        text = "doc_007:\n  1: [[0, 0, 1, 1]]\n  2:\n    - [0, 0, 1, 1]\n    - [0, 0, 0, 1]\n"
        with pytest.raises(RedactionError, match=r"doc_007 page 2 box 2: .*must be positive"):
            parse_redactions(text)

    def test_the_source_name_is_in_the_message(self) -> None:
        with pytest.raises(RedactionError, match=r"^mine.yaml: "):
            parse_redactions("- x\n", source="mine.yaml")

    def test_python_objects_cannot_be_constructed(self) -> None:
        with pytest.raises(RedactionError):
            parse_redactions("d: !!python/object/apply:os.getcwd []\n")

    def test_parse_box(self) -> None:
        assert parse_box([1, 2, 3, 4], "b") == (1, 2, 3, 4)
        assert parse_box((1.0, 2, 3, 4), "b") == (1, 2, 3, 4)
        with pytest.raises(RedactionError, match="^b: "):
            parse_box(None, "b")


class TestReadFile:
    def test_a_missing_default_file_means_no_redactions(self, tmp_path: Path) -> None:
        assert read_redactions(tmp_path / "nope.yaml") == {}

    def test_a_missing_explicit_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(RedactionError, match="nope.yaml: no such file"):
            read_redactions(tmp_path / "nope.yaml", must_exist=True)

    def test_it_reads_the_file_and_names_it_in_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "r.yaml"
        path.write_text("d:\n  1: [[0, 0, 1, 1]]\n", encoding="utf-8")
        assert read_redactions(path)["d"][1].boxes == ((0, 0, 1, 1),)
        path.write_text("- x\n", encoding="utf-8")
        with pytest.raises(RedactionError, match=r"^r.yaml: "):
            read_redactions(path)

    def test_a_byte_order_mark_from_a_windows_editor_is_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "r.yaml"
        path.write_bytes(b"\xef\xbb\xbfd:\n  1: [[0, 0, 1, 1]]\n")
        assert "d" in read_redactions(path)

    def test_a_file_that_is_not_text_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "r.yaml"
        path.write_bytes(b"d:\n  1: [[0, 0, 1, \xff\xfe]]\n")
        with pytest.raises(RedactionError, match="cannot be read"):
            read_redactions(path)

    def test_the_committed_file_is_valid_and_documents_the_format(self) -> None:
        committed = Path(__file__).resolve().parents[1] / "data" / "redactions.yaml"
        assert read_redactions(committed, must_exist=True) == {}
        assert "x, y, w, h" in committed.read_text(encoding="utf-8")


class TestAgainstTheManifest:
    def test_unknown_documents_and_variants_are_reported(self) -> None:
        rows = [
            doc("a.png", doc_id="doc_001"),
            doc(doc_id="doc_001-deg-light", variant="light", gt_from="doc_001"),
        ]
        redactions = parse_redactions(
            "doc_001: {1: [[0, 0, 1, 1]]}\n"
            "doc_999: {1: [[0, 0, 1, 1]]}\n"
            "doc_001-deg-light: {1: [[0, 0, 1, 1]]}\n"
        )
        problems = check_against_manifest(redactions, rows)
        assert len(problems) == 2
        assert problems[0].startswith("doc_999: not in the manifest")
        assert problems[1].startswith("doc_001-deg-light: is a degraded variant")
        assert "put the boxes on doc_001" in problems[1]

    def test_a_consistent_file_has_no_problems(self) -> None:
        rows = [doc("a.png", doc_id="doc_001")]
        assert (
            check_against_manifest(parse_redactions("doc_001: {1: [[0, 0, 1, 1]]}\n"), rows) == []
        )

    def test_plain_conversions(self) -> None:
        entry = parse_redactions(
            "d:\n  1: [[1, 2, 3, 4]]\n  2:\n    size: [9, 8]\n    boxes: [[0, 0, 1, 1]]\n"
        )["d"]
        assert boxes_of(entry) == {1: [[1, 2, 3, 4]], 2: [[0, 0, 1, 1]]}
        assert pinned_sizes_of(entry) == {2: (9, 8)}
        assert boxes_of({}) == {} and pinned_sizes_of({}) == {}


# ------------------------------------------------------------------ pinned sizes


class TestPinnedSize:
    def test_a_matching_pin_is_accepted(self, env: Env) -> None:
        gray_png(env)
        row = doc("g.png", needs_redaction="y")
        result = env.ingest(row, boxes={1: [[10, 5, 20, 8]]}, pinned_sizes={1: (80, 60)})
        assert result.status == "ingested"

    def test_a_mismatching_pin_is_refused_and_nothing_is_written(self, env: Env) -> None:
        gray_png(env)
        row = doc("g.png", needs_redaction="y")
        with pytest.raises(
            IngestError, match=r"page 1 is 80x60 but redactions.yaml says .* 100x50"
        ):
            env.ingest(row, boxes={1: [[10, 5, 20, 8]]}, pinned_sizes={1: (100, 50)})
        assert not env.out(row).exists()

    def test_width_and_height_are_not_interchangeable(self, env: Env) -> None:
        gray_png(env, size=(80, 60))
        with pytest.raises(IngestError, match="60x80"):
            env.ingest(
                doc("g.png", needs_redaction="y"),
                boxes={1: [[0, 0, 5, 5]]},
                pinned_sizes={1: (60, 80)},
            )

    def test_an_unchanged_document_is_still_checked_against_a_new_pin(self, env: Env) -> None:
        gray_png(env)
        row = doc("g.png", needs_redaction="y")
        boxes = {1: [[10, 5, 20, 8]]}
        assert env.ingest(row, boxes=boxes).status == "ingested"
        assert env.ingest(row, boxes=boxes, pinned_sizes={1: (80, 60)}).status == "unchanged"
        with pytest.raises(IngestError, match="measured on 81x60"):
            env.ingest(row, boxes=boxes, pinned_sizes={1: (81, 60)})

    def test_the_pin_helper_ignores_pages_without_a_pin(self) -> None:
        check_pinned_sizes("d", {1: (10, 10), 2: (20, 20)}, {2: (20, 20)})
        with pytest.raises(IngestError, match="page 2 is 20x20"):
            check_pinned_sizes("d", {1: (10, 10), 2: (20, 20)}, {2: (21, 20)})


# ------------------------------------------------------------------ the resize guard

BOXES = {1: [[100, 100, 300, 100]]}


class TestResizeGuard:
    def pdf_doc(self, env: Env) -> Any:
        make_sized_pdf(env.raw / "a.pdf", [A4])
        return doc("a.pdf", needs_redaction="y")

    def stored(self, env: Env, row: Any) -> bytes:
        return (env.out(row) / "p1.png").read_bytes()

    def test_untouched_boxes_on_a_page_that_changed_size_are_refused(self, env: Env) -> None:
        row = self.pdf_doc(env)
        assert env.ingest(row, boxes=BOXES, settings=settings(dpi=100)).pages[0].size == (827, 1169)
        before = self.stored(env, row)
        with pytest.raises(IngestError, match=r"was 827x1169 .* now 1240x1754 \(DPI or size cap"):
            env.ingest(row, boxes=BOXES, settings=settings(dpi=150))
        assert self.stored(env, row) == before  # the old page is left exactly as it was

    def test_a_changed_size_cap_is_caught_the_same_way(self, env: Env) -> None:
        make_jpeg(env.raw / "big.jpg", size=(4000, 3000), color=(200, 200, 200))
        row = doc("big.jpg", needs_redaction="y")
        boxes = {1: [[100, 100, 50, 50]]}
        env.ingest(row, boxes=boxes)  # stored at 2400x1800
        with pytest.raises(IngestError, match=r"was 2400x1800 .* now 1200x900"):
            env.ingest(row, boxes=boxes, settings=settings(max_long_edge=1200))

    def test_accept_resize_lets_it_through(self, env: Env) -> None:
        row = self.pdf_doc(env)
        env.ingest(row, boxes=BOXES, settings=settings(dpi=100))
        result = env.ingest(row, boxes=BOXES, settings=settings(dpi=150), accept_resize=True)
        assert result.status == "ingested"
        assert result.pages[0].size == (1240, 1754)

    def test_edited_boxes_on_the_new_size_are_fine(self, env: Env) -> None:
        row = self.pdf_doc(env)
        env.ingest(row, boxes=BOXES, settings=settings(dpi=100))
        moved = {1: [[150, 150, 450, 150]]}
        assert env.ingest(row, boxes=moved, settings=settings(dpi=150)).status == "ingested"

    def test_a_pinned_size_that_matches_the_new_page_is_fine(self, env: Env) -> None:
        row = self.pdf_doc(env)
        env.ingest(row, boxes=BOXES, settings=settings(dpi=100))
        result = env.ingest(
            row, boxes=BOXES, pinned_sizes={1: (1240, 1754)}, settings=settings(dpi=150)
        )
        assert result.status == "ingested"

    def test_a_pinned_size_that_is_stale_is_refused_even_with_accept_resize(self, env: Env) -> None:
        row = self.pdf_doc(env)
        env.ingest(row, boxes=BOXES, settings=settings(dpi=100))
        with pytest.raises(IngestError, match="measured on 827x1169"):
            env.ingest(
                row,
                boxes=BOXES,
                pinned_sizes={1: (827, 1169)},
                settings=settings(dpi=150),
                accept_resize=True,
            )

    def test_the_first_ingestion_has_nothing_to_compare_with(self, env: Env) -> None:
        row = self.pdf_doc(env)
        assert env.ingest(row, boxes=BOXES, settings=settings(dpi=150)).status == "ingested"

    def test_a_forced_rerun_at_the_same_size_is_fine(self, env: Env) -> None:
        row = self.pdf_doc(env)
        env.ingest(row, boxes=BOXES, settings=settings(dpi=100))
        assert (
            env.ingest(row, boxes=BOXES, settings=settings(dpi=100), force=True).status
            == "ingested"
        )

    def test_a_page_without_boxes_may_change_size_freely(self, env: Env) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4, A4])
        row = doc("a.pdf", needs_redaction="y")
        boxes = {1: [[100, 100, 300, 100]]}  # page 2 has none
        env.ingest(row, boxes=boxes, settings=settings(dpi=100))
        moved = {1: [[150, 150, 450, 150]]}
        assert env.ingest(row, boxes=moved, settings=settings(dpi=150)).status == "ingested"

    def test_a_damaged_sidecar_does_not_crash_the_guard(self, env: Env) -> None:
        row = self.pdf_doc(env)
        env.ingest(row, boxes=BOXES, settings=settings(dpi=100))
        (env.out(row) / SIDECAR).write_text(
            '{"ingest_version": 1, "pages": [{}]}', encoding="utf-8"
        )
        assert env.ingest(row, boxes=BOXES, settings=settings(dpi=150)).status == "ingested"


# ------------------------------------------------------------------ previews


class TestPreview:
    def preview(self, env: Env, row: Any, tmp_path: Path, **kwargs: Any) -> Any:
        return preview_document(
            row,
            raw_dir=env.raw,
            preview_dir=tmp_path / "private" / "preview",
            pages_dir=env.pages,
            settings=kwargs.pop("settings", settings()),
            **kwargs,
        )

    def test_it_is_the_stored_page_without_the_black_box(self, env: Env, tmp_path: Path) -> None:
        gray_png(env)
        row = doc("g.png", needs_redaction="y")
        env.ingest(row, boxes={1: [[10, 5, 20, 8]]})
        stored = env.page(row, "p1.png")
        records = self.preview(env, row, tmp_path)
        shown = Image.open(tmp_path / "private" / "preview" / "doc_001" / "p1.png")
        shown.load()
        assert records[0].size == shown.size == stored.size == (80, 60)
        assert shown.getpixel((15, 8)) == (128, 128, 128)  # unredacted
        assert stored.getpixel((15, 8)) == (0, 0, 0)  # redacted
        differing = {
            (x, y)
            for x in range(80)
            for y in range(60)
            if shown.getpixel((x, y)) != stored.getpixel((x, y))
        }
        assert differing == {(x, y) for x in range(10, 30) for y in range(5, 13)}

    def test_it_shares_the_pixel_space_after_rotation_rendering_and_the_cap(
        self, env: Env, tmp_path: Path
    ) -> None:
        make_jpeg(env.raw / "p.jpg", size=(4000, 3000), orientation=6, color=(200, 200, 200))
        make_sized_pdf(env.raw / "a.pdf", [A4])
        for name in ("p.jpg", "a.pdf"):
            row = doc(name, needs_redaction="n")
            stored = env.ingest(row).pages
            shown = self.preview(env, row, tmp_path)
            assert [p.size for p in shown] == [p.size for p in stored]
            assert [p.file for p in shown] == [p.file for p in stored]
        assert shown[0].size == (1654, 2339)

    def test_it_works_before_any_box_exists(self, env: Env, tmp_path: Path) -> None:
        gray_png(env)
        for decision in ("y", ""):  # undecided, and "y" with no boxes: both are refused by ingest
            row = doc("g.png", needs_redaction=decision)
            with pytest.raises(IngestError):
                env.ingest(row)
            assert self.preview(env, row, tmp_path)[0].file == "p1.png"

    def test_it_never_writes_to_the_pages_directory_and_has_no_sidecar(
        self, env: Env, tmp_path: Path
    ) -> None:
        gray_png(env)
        self.preview(env, doc("g.png", needs_redaction=""), tmp_path)
        assert not env.pages.exists()
        folder = tmp_path / "private" / "preview" / "doc_001"
        assert sorted(p.name for p in folder.iterdir()) == ["p1.png"]

    def test_pages_that_no_longer_exist_are_removed(self, env: Env, tmp_path: Path) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4, A4])
        folder = tmp_path / "private" / "preview" / "doc_001"
        self.preview(env, doc("a.pdf", needs_redaction=""), tmp_path)
        assert (folder / "p2.png").exists()
        self.preview(env, doc("a.pdf", needs_redaction="", page_range="1"), tmp_path)
        assert not (folder / "p2.png").exists()

    def test_a_variant_has_nothing_to_preview(self, env: Env, tmp_path: Path) -> None:
        row = doc(doc_id="doc_001-deg-light", variant="light", gt_from="doc_001")
        with pytest.raises(IngestError, match="no raw pages"):
            self.preview(env, row, tmp_path)

    def test_a_missing_raw_file_is_reported(self, env: Env, tmp_path: Path) -> None:
        with pytest.raises(IngestError, match="raw file not found: gone.png"):
            self.preview(env, doc("gone.png"), tmp_path)

    def test_a_pages_directory_is_not_a_preview_directory(self, env: Env, tmp_path: Path) -> None:
        gray_png(env)
        row = doc("g.png", needs_redaction="y")
        env.ingest(row, boxes={1: [[0, 0, 5, 5]]})
        # even through a different spelling of the path, a folder holding ingest.json is refused
        elsewhere = tmp_path / "copy"
        (elsewhere / "doc_001").mkdir(parents=True)
        (elsewhere / "doc_001" / SIDECAR).write_text("{}", encoding="utf-8")
        with pytest.raises(PreviewError, match="stored-pages directory"):
            preview_document(
                row,
                raw_dir=env.raw,
                preview_dir=elsewhere,
                pages_dir=env.pages,
                settings=settings(),
            )
        assert not (elsewhere / "doc_001" / "p1.png").exists()


class TestPreviewDirectory:
    @pytest.mark.parametrize(
        "relative",
        [
            "pages",  # the pages directory itself
            "pages/doc_001",  # inside it
            "pages/sub/deeper",
            ".",  # a parent of it
            "pages/../pages",  # the same place spelled differently
            "other/../pages",  # ".." is what makes this the pages directory
            "other/../pages/doc_001",
        ],
    )
    def test_directories_that_overlap_the_pages_directory_are_refused(
        self, tmp_path: Path, relative: str
    ) -> None:
        with pytest.raises(PreviewError, match="overlaps the stored-pages directory"):
            check_preview_dir(tmp_path / relative, tmp_path / "pages")

    @pytest.mark.parametrize(
        "relative",
        [
            "private/preview",
            "pages-preview",
            "preview",
            "other",
            "pages/../preview",  # looks like it is inside pages, but is not
        ],
    )
    def test_other_directories_are_fine(self, tmp_path: Path, relative: str) -> None:
        check_preview_dir(tmp_path / relative, tmp_path / "pages")

    def test_the_default_layout_is_allowed(self) -> None:
        check_preview_dir(Path("data/private/preview"), Path("data/pages"))

    def test_a_directory_named_like_a_prefix_of_pages_is_not_overlap(self, tmp_path: Path) -> None:
        check_preview_dir(tmp_path / "pagesX", tmp_path / "pages")


# ------------------------------------------------------------------ the command line


def dataset(tmp_path: Path, *rows: Any) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_sized_pdf(raw / "a.pdf", [A4])
    make_jpeg(raw / "p.jpg", size=(300, 200), color=(200, 200, 200))
    return raw, write_manifest_file(tmp_path, list(rows)), tmp_path / "pages"


def common(raw: Path, manifest: Path, pages: Path) -> list[str]:
    return ["--raw-dir", str(raw), "--manifest", str(manifest), "--pages-dir", str(pages)]


@pytest.mark.slow
class TestCommandLine:
    def test_boxes_from_the_yaml_are_burned_into_the_stored_page(self, tmp_path: Path) -> None:
        raw, manifest, pages = dataset(
            tmp_path,
            doc("a.pdf", doc_id="doc_001", needs_redaction="y"),
            doc("p.jpg", doc_id="doc_002", needs_redaction="n"),
        )
        yaml_file = tmp_path / "r.yaml"
        yaml_file.write_text("doc_001:\n  1:\n    - [100, 100, 300, 100]\n", encoding="utf-8")
        args = [*common(raw, manifest, pages), "--redactions", str(yaml_file)]
        code, out, _ = run_script("ingest.py", *args)
        assert code == 0, out
        assert "[1 redaction box(es)]" in out
        page = Image.open(pages / "doc_001" / "p1.png")
        assert page.convert("RGB").getpixel((200, 150)) == (0, 0, 0)
        assert page.convert("RGB").getpixel((50, 50)) == (255, 255, 255)
        code, out, _ = run_script("ingest.py", *args)
        assert "0 ingested, 2 unchanged" in out
        yaml_file.write_text("doc_001:\n  1:\n    - [100, 100, 301, 100]\n", encoding="utf-8")
        code, out, _ = run_script("ingest.py", *args)
        assert "1 ingested, 1 unchanged" in out  # only the edited document is redone

    def test_y_without_boxes_in_the_file_is_refused(self, tmp_path: Path) -> None:
        raw, manifest, pages = dataset(tmp_path, doc("a.pdf", needs_redaction="y"))
        empty = tmp_path / "r.yaml"
        empty.write_text("# nothing yet\n", encoding="utf-8")
        code, out, _ = run_script(
            "ingest.py", *common(raw, manifest, pages), "--redactions", str(empty)
        )
        assert code == 1
        assert "refusing to store an unredacted page" in out
        assert not (pages / "doc_001").exists()

    def test_boxes_for_a_document_marked_n_fail_that_document(self, tmp_path: Path) -> None:
        raw, manifest, pages = dataset(tmp_path, doc("a.pdf", needs_redaction="n"))
        yaml_file = tmp_path / "r.yaml"
        yaml_file.write_text("doc_001: {1: [[1, 1, 5, 5]]}\n", encoding="utf-8")
        code, out, _ = run_script(
            "ingest.py", *common(raw, manifest, pages), "--redactions", str(yaml_file)
        )
        assert code == 1
        assert "needs_redaction=n" in out

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("doc_999: {1: [[0, 0, 1, 1]]}\n", "doc_999: not in the manifest"),
            ("doc_001: {1: [[0, 0, 1, 1]]}\ndoc_001: {1: [[2, 2, 1, 1]]}\n", "appears twice"),
            ("- nonsense\n", "top level must map"),
        ],
    )
    def test_a_bad_file_stops_everything_before_any_page_is_written(
        self, tmp_path: Path, text: str, message: str
    ) -> None:
        raw, manifest, pages = dataset(
            tmp_path,
            doc("a.pdf", doc_id="doc_001", needs_redaction="y"),
            doc("p.jpg", doc_id="doc_002", needs_redaction="n"),
        )
        yaml_file = tmp_path / "r.yaml"
        yaml_file.write_text(text, encoding="utf-8")
        code, _, err = run_script(
            "ingest.py", *common(raw, manifest, pages), "--redactions", str(yaml_file)
        )
        assert code == 2
        assert message in err
        assert not pages.exists()  # not even the document that had nothing wrong with it

    def test_a_check_of_the_whole_file_happens_even_when_one_doc_is_selected(
        self, tmp_path: Path
    ) -> None:
        raw, manifest, pages = dataset(tmp_path, doc("p.jpg", doc_id="doc_002"))
        yaml_file = tmp_path / "r.yaml"
        yaml_file.write_text("doc_999: {1: [[0, 0, 1, 1]]}\n", encoding="utf-8")
        code, _, err = run_script(
            "ingest.py",
            *common(raw, manifest, pages),
            "--redactions",
            str(yaml_file),
            "--doc",
            "doc_002",
        )
        assert code == 2
        assert "doc_999" in err

    def test_an_explicit_redactions_file_must_exist(self, tmp_path: Path) -> None:
        raw, manifest, pages = dataset(tmp_path, doc("p.jpg"))
        code, _, err = run_script(
            "ingest.py", *common(raw, manifest, pages), "--redactions", str(tmp_path / "nope.yaml")
        )
        assert code == 2
        assert "no such file" in err

    def test_a_pinned_size_that_is_wrong_fails_the_document(self, tmp_path: Path) -> None:
        raw, manifest, pages = dataset(tmp_path, doc("p.jpg", needs_redaction="y"))
        yaml_file = tmp_path / "r.yaml"
        yaml_file.write_text(
            "doc_001:\n  1:\n    size: [640, 480]\n    boxes: [[1, 1, 5, 5]]\n", encoding="utf-8"
        )
        code, out, _ = run_script(
            "ingest.py", *common(raw, manifest, pages), "--redactions", str(yaml_file)
        )
        assert code == 1
        assert (
            "page 1 is 300x200 but redactions.yaml says its boxes were measured on 640x480" in out
        )

    def test_a_changed_dpi_under_untouched_boxes_is_refused_until_accepted(
        self, tmp_path: Path
    ) -> None:
        raw, manifest, pages = dataset(tmp_path, doc("a.pdf", needs_redaction="y"))
        yaml_file = tmp_path / "r.yaml"
        yaml_file.write_text("doc_001: {1: [[100, 100, 300, 100]]}\n", encoding="utf-8")
        args = [*common(raw, manifest, pages), "--redactions", str(yaml_file)]
        assert run_script("ingest.py", *args, "--dpi", "100")[0] == 0
        code, out, _ = run_script("ingest.py", *args, "--dpi", "150")
        assert code == 1
        assert "DPI or size cap changed" in out
        code, out, _ = run_script("ingest.py", *args, "--dpi", "150", "--accept-resize")
        assert code == 0
        assert "1240x1754" in out

    def test_preview_writes_unredacted_pages_even_with_a_broken_redactions_file(
        self, tmp_path: Path
    ) -> None:
        raw, manifest, pages = dataset(
            tmp_path,
            doc("a.pdf", doc_id="doc_001", needs_redaction="y"),
            doc("p.jpg", doc_id="doc_002", needs_redaction=""),
        )
        preview = tmp_path / "private" / "preview"
        broken = tmp_path / "r.yaml"
        broken.write_text("d:\n  1: [[0, 0, 1, 1]]\nd:\n", encoding="utf-8")
        code, out, _ = run_script(
            "ingest.py",
            *common(raw, manifest, pages),
            "--preview",
            "--preview-dir",
            str(preview),
            "--redactions",
            str(broken),
        )
        assert code == 0, out
        assert "2 previewed" in out
        assert "UNREDACTED" in out
        assert not pages.exists()
        assert (preview / "doc_001" / "p1.png").is_file()
        assert (preview / "doc_002" / "p1.jpg").is_file()
        assert not (preview / "doc_001" / SIDECAR).exists()
        shown = Image.open(preview / "doc_002" / "p1.jpg").convert("RGB")
        assert shown.size == (300, 200)
        assert min(shown.getpixel((150, 100))) > 150  # a plain grey photo: nothing blacked out
        code, out, _ = run_script("ingest.py", *common(raw, manifest, pages), "--doc", "doc_002")
        assert code == 1  # still undecided: ingest refuses, preview did not change that

    @pytest.mark.parametrize("relative", ["pages", "pages/doc_001", "."])
    def test_preview_refuses_a_directory_that_overlaps_the_pages(
        self, tmp_path: Path, relative: str
    ) -> None:
        raw, manifest, pages = dataset(tmp_path, doc("p.jpg"))
        code, out, err = run_script(
            "ingest.py",
            *common(raw, manifest, pages),
            "--preview",
            "--preview-dir",
            str(tmp_path / relative),
        )
        assert code == 2
        assert "overlaps the stored-pages directory" in err
        assert not pages.exists()
        assert "previewed" not in out

    def test_the_overlap_is_refused_before_anything_is_announced_or_selected(
        self, tmp_path: Path
    ) -> None:
        # --doc names only the variant, so no document would be previewed: only the up-front
        # check can refuse
        raw, manifest, pages = dataset(
            tmp_path,
            doc("p.jpg", doc_id="doc_001"),
            doc(doc_id="doc_001-deg-light", variant="light", gt_from="doc_001"),
        )
        code, out, err = run_script(
            "ingest.py",
            *common(raw, manifest, pages),
            "--preview",
            "--preview-dir",
            str(pages),
            "--doc",
            "doc_001-deg-light",
        )
        assert code == 2
        assert "overlaps the stored-pages directory" in err
        assert "writing to" not in out

    def test_preview_cannot_be_combined_with_check(self, tmp_path: Path) -> None:
        raw, manifest, pages = dataset(tmp_path, doc("p.jpg"))
        code, _, err = run_script(
            "ingest.py", *common(raw, manifest, pages), "--preview", "--check"
        )
        assert code == 2
        assert "cannot be combined" in err

    def test_preview_of_one_document(self, tmp_path: Path) -> None:
        raw, manifest, pages = dataset(
            tmp_path, doc("a.pdf", doc_id="doc_001"), doc("p.jpg", doc_id="doc_002")
        )
        preview = tmp_path / "pv"
        code, out, _ = run_script(
            "ingest.py",
            *common(raw, manifest, pages),
            "--preview",
            "--preview-dir",
            str(preview),
            "--doc",
            "doc_002",
        )
        assert code == 0
        assert (preview / "doc_002").is_dir()
        assert not (preview / "doc_001").exists()
