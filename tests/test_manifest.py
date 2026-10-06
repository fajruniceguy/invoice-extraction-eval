from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from docextract.ingest.manifest import (
    COLUMNS,
    REQUIRED_COLUMNS,
    ManifestError,
    ManifestRow,
    check_manifest,
    content_group_split_conflicts,
    missing_fields,
    parse_manifest,
    parse_page_range,
    read_manifest,
    write_manifest,
)

HEADER = ",".join(COLUMNS)


def row(**overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {"doc_id": "doc_001", "raw_files": "a.pdf"}
    values.update(overrides)
    return ManifestRow.model_validate(values)


def csv_text(*lines: str, header: str = HEADER) -> str:
    return "\n".join([header, *lines]) + "\n"


class TestRow:
    def test_a_skeleton_row_needs_only_an_id_and_a_file(self) -> None:
        r = row()
        assert r.raw_files == ("a.pdf",)
        assert r.doc_type is None
        assert r.needs_redaction is None
        assert r.split is None

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("a.pdf", ("a.pdf",)),
            ("a.jpg;b.jpg", ("a.jpg", "b.jpg")),
            (" a.jpg ; b.jpg ;; ", ("a.jpg", "b.jpg")),
            ("sub/dir/a.jpg", ("sub/dir/a.jpg",)),
            (("a.jpg", "b.jpg"), ("a.jpg", "b.jpg")),
        ],
    )
    def test_raw_files_are_split_on_semicolons(self, given: Any, expected: tuple[str, ...]) -> None:
        assert row(raw_files=given).raw_files == expected

    @pytest.mark.parametrize("doc_id", ["doc_001", "a", "Z9", "faktur-2024.03", "x_y-z.1"])
    def test_good_doc_ids(self, doc_id: str) -> None:
        assert row(doc_id=doc_id).doc_id == doc_id

    @pytest.mark.parametrize(
        "doc_id", ["", "-a", "_a", ".a", "a b", "a/b", "a\\b", "a;b", "é", "a:b"]
    )
    def test_a_doc_id_must_be_safe_as_a_directory_name(self, doc_id: str) -> None:
        with pytest.raises(ValidationError):
            row(doc_id=doc_id)

    @pytest.mark.parametrize(
        "name",
        [
            "/etc/passwd",
            "C:\\x.pdf",
            "C:/x.pdf",
            "..\\x.pdf",
            "../x.pdf",
            "a/../b.pdf",
            "a//b.pdf",
            "a\\b.pdf",
        ],
    )
    def test_raw_files_must_be_relative_and_stay_inside_the_raw_folder(self, name: str) -> None:
        with pytest.raises(ValidationError):
            row(raw_files=name)

    @pytest.mark.parametrize(
        ("given", "normalized"),
        [("3", "3"), ("2-4", "2-4"), ("1,3-4", "1,3-4"), (" 1 - 2 ", "1-2"), ("1, 3", "1,3")],
    )
    def test_page_range_grammar(self, given: str, normalized: str) -> None:
        assert row(page_range=given).page_range == normalized

    @pytest.mark.parametrize("bad", ["0", "3-2", "a", "1-", "-2", "1,,2", "1;2", "0-3", "2.5"])
    def test_bad_page_ranges(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            row(page_range=bad)

    def test_page_range_slices_exactly_one_file(self) -> None:
        with pytest.raises(ValidationError, match="exactly one raw file"):
            row(raw_files="a.jpg;b.jpg", page_range="1")
        assert row(raw_files="bundle.pdf", page_range="2-3").page_range == "2-3"

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("y", True),
            ("YES", True),
            ("true", True),
            ("1", True),
            ("n", False),
            ("No", False),
            ("false", False),
            ("0", False),
            ("", None),
            ("  ", None),
        ],
    )
    def test_needs_redaction_is_yes_or_no(self, given: str, expected: bool | None) -> None:
        assert row(needs_redaction=given).needs_redaction is expected

    def test_needs_redaction_rejects_anything_else(self) -> None:
        with pytest.raises(ValidationError, match="must be y or n"):
            row(needs_redaction="maybe")

    @pytest.mark.parametrize(
        ("column", "bad"),
        [
            ("doc_type", "memo"),
            ("capture", "scan"),
            ("split", "train"),
            ("label_mode", "manual"),
            ("variant", "extreme"),
        ],
    )
    def test_enums_reject_unknown_values(self, column: str, bad: str) -> None:
        with pytest.raises(ValidationError):
            row(**{column: bad})

    def test_blank_cells_are_none(self) -> None:
        r = row(doc_type="", template_group=" ", language="", split="", label_mode="")
        assert r.doc_type is None
        assert r.template_group is None
        assert r.language is None
        assert r.split is None
        assert r.label_mode is None

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            row(color="red")

    def test_whitespace_is_stripped(self) -> None:
        assert row(doc_id="  doc_001  ", language=" id ").language == "id"

    def test_a_non_derived_doc_needs_raw_files(self) -> None:
        with pytest.raises(ValidationError, match="raw_files is required"):
            ManifestRow.model_validate({"doc_id": "doc_001"})

    def test_gt_from_cannot_point_at_itself(self) -> None:
        with pytest.raises(ValidationError, match="itself"):
            row(gt_from="doc_001")

    def test_a_content_group_duplicate_may_borrow_labels_and_keeps_its_files(self) -> None:
        r = row(doc_id="doc_002", gt_from="doc_001")
        assert r.gt_from == "doc_001"
        assert r.raw_files == ("a.pdf",)
        assert not r.is_derived

    def test_a_valid_degraded_variant(self) -> None:
        r = ManifestRow.model_validate(
            {
                "doc_id": "doc_001-deg-light",
                "capture": "synthetic_degraded",
                "variant": "light",
                "gt_from": "doc_001",
                "content_group": "doc_001",
            }
        )
        assert r.is_derived
        assert r.raw_files == ()

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"variant": "light"}, "both variant and gt_from"),  # no gt_from
            ({"variant": "light", "gt_from": "x", "raw_files": "a.jpg"}, "no raw_files"),
            ({"capture": "synthetic_degraded", "gt_from": "x"}, "both variant and gt_from"),
            (
                {"variant": "light", "gt_from": "x", "capture": "born_digital"},
                "capture=synthetic_degraded",
            ),
        ],
    )
    def test_variant_rules(self, overrides: dict[str, Any], message: str) -> None:
        values: dict[str, Any] = {"doc_id": "v1", **overrides}
        with pytest.raises(ValidationError, match=message):
            ManifestRow.model_validate(values)

    def test_to_cells_uses_blanks_and_y_n(self) -> None:
        cells = row(raw_files="a.jpg;b.jpg", needs_redaction="y", doc_type="receipt").to_cells()
        assert cells["raw_files"] == "a.jpg;b.jpg"
        assert cells["needs_redaction"] == "y"
        assert cells["doc_type"] == "receipt"
        assert cells["split"] == ""
        assert list(cells) == list(COLUMNS)
        assert row(needs_redaction="n").to_cells()["needs_redaction"] == "n"
        assert row().to_cells()["needs_redaction"] == ""


class TestParsePageRange:
    @pytest.mark.parametrize(
        ("spec", "pages", "expected"),
        [
            (None, 3, [1, 2, 3]),
            ("2", 3, [2]),
            ("2-3", 5, [2, 3]),
            ("1,3-4", 5, [1, 3, 4]),
            ("1-5", 5, [1, 2, 3, 4, 5]),
        ],
    )
    def test_selected_pages(self, spec: str | None, pages: int, expected: list[int]) -> None:
        assert parse_page_range(spec, pages) == expected

    def test_a_page_beyond_the_file_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="page 4 of a 3-page file"):
            parse_page_range("2-4", 3)

    @pytest.mark.parametrize("spec", ["1,1", "3,1", "2-3,3"])
    def test_repeated_or_reordered_pages_are_an_error(self, spec: str) -> None:
        with pytest.raises(ValueError, match="repeats or reorders"):
            parse_page_range(spec, 9)


class TestCsv:
    def full_rows(self) -> list[ManifestRow]:
        return [
            row(
                doc_id="doc_001",
                raw_files="x/a.jpg;x/b.jpg",
                doc_type="receipt",
                template_group="indomaret",
                content_group="g1",
                capture="phone_photo",
                original_format="jpg;jpg",
                language="id",
                source_url="https://example.test/a",
                license_note="own photo",
                split="dev",
                label_mode="prelabel",
                needs_redaction="y",
            ),
            row(doc_id="doc_002", raw_files="bundle.pdf", page_range="1-2", needs_redaction="n"),
            ManifestRow.model_validate(
                {
                    "doc_id": "doc_001-deg-heavy",
                    "capture": "synthetic_degraded",
                    "variant": "heavy",
                    "gt_from": "doc_001",
                    "content_group": "g1",
                    "split": "dev",
                }
            ),
        ]

    def test_write_then_read_round_trips_every_column(self, tmp_path: Path) -> None:
        path = tmp_path / "data" / "manifest.csv"
        rows = self.full_rows()
        write_manifest(path, rows)
        assert read_manifest(path) == rows

    def test_the_file_has_the_documented_header_and_unix_newlines(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        write_manifest(path, self.full_rows())
        raw = path.read_bytes()
        assert raw.startswith(HEADER.encode() + b"\n")
        assert b"\r" not in raw
        assert not raw.startswith(b"\xef\xbb\xbf")

    def test_writing_leaves_no_temp_files_and_creates_parents(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "manifest.csv"
        write_manifest(path, [row()])
        assert [p.name for p in path.parent.iterdir()] == ["manifest.csv"]

    def test_the_header_lists_the_14_documented_columns_first(self) -> None:
        assert len(REQUIRED_COLUMNS) == 14
        assert COLUMNS[:14] == REQUIRED_COLUMNS
        assert COLUMNS[14:] == ("gt_from", "variant")

    def test_optional_columns_may_be_absent_from_an_older_file(self) -> None:
        header = ",".join(REQUIRED_COLUMNS)
        text = csv_text("doc_001,a.pdf,,,,,,,,,,,,", header=header)
        assert parse_manifest(text)[0].doc_id == "doc_001"

    def test_a_byte_order_mark_from_excel_is_tolerated(self) -> None:
        assert (
            parse_manifest("\ufeff" + csv_text("doc_001,a.pdf,,,,,,,,,,,,,,"))[0].doc_id
            == "doc_001"
        )

    def test_windows_line_endings_are_tolerated(self) -> None:
        text = csv_text("doc_001,a.pdf,,,,,,,,,,,,,,").replace("\n", "\r\n")
        assert parse_manifest(text)[0].raw_files == ("a.pdf",)

    def test_a_semicolon_delimited_save_from_excel_is_detected(self) -> None:
        header = ";".join(COLUMNS)
        # Excel quotes a cell containing the delimiter, and the file-list cell contains ';'
        line = 'doc_001;"x.jpg;y.jpg";;receipt;;;;;;;;;;y;;'
        r = parse_manifest(csv_text(line, header=header))[0]
        assert r.raw_files == ("x.jpg", "y.jpg")
        assert r.doc_type == "receipt"
        assert r.needs_redaction is True

    def test_blank_lines_are_ignored(self) -> None:
        text = csv_text(
            "doc_001,a.pdf,,,,,,,,,,,,,,", ",,,,,,,,,,,,,,,", "doc_002,b.pdf,,,,,,,,,,,,,,"
        )
        assert [r.doc_id for r in parse_manifest(text)] == ["doc_001", "doc_002"]

    def test_quoted_cells_with_commas_survive(self) -> None:
        text = csv_text('doc_001,a.pdf,,,,,,,,,"Own photo, taken in Jakarta",,,,,')
        assert parse_manifest(text)[0].license_note == "Own photo, taken in Jakarta"

    @pytest.mark.parametrize("text", ["", "\n", "   \n"])
    def test_an_empty_file_is_an_error(self, text: str) -> None:
        with pytest.raises(ManifestError, match="empty"):
            parse_manifest(text)

    def test_the_header_must_start_with_doc_id(self) -> None:
        with pytest.raises(ManifestError, match="header must start with 'doc_id'"):
            parse_manifest("raw_files,doc_id\na.pdf,doc_001\n")

    def test_unknown_columns_are_named(self) -> None:
        with pytest.raises(ManifestError, match="unknown column.*colour"):
            parse_manifest(csv_text("doc_001,a.pdf,,,,,,,,,,,,,,,red", header=HEADER + ",colour"))

    def test_missing_required_columns_are_named(self) -> None:
        with pytest.raises(ManifestError, match="missing column.*split"):
            parse_manifest("doc_id,raw_files\ndoc_001,a.pdf\n")

    def test_every_row_problem_is_reported_with_its_row_number_and_doc_id(self) -> None:
        text = csv_text(
            "doc_001,a.pdf,,,,,,,,,,,,,,",
            "doc_002,b.pdf,,memo,,,,,,,,,,,,",  # bad doc_type
            "doc_003,c.pdf,5-1,,,,,,,,,,,,,",  # bad page_range
            "bad id,d.pdf,,,,,,,,,,,,,,",  # bad doc_id
        )
        with pytest.raises(ManifestError) as exc:
            parse_manifest(text)
        problems = "\n".join(exc.value.problems)
        assert "row 3 (doc_002) doc_type" in problems
        assert "row 4 (doc_003) page_range" in problems
        assert "row 5 (bad id) doc_id" in problems
        assert len(exc.value.problems) == 3

    def test_duplicate_doc_ids_are_an_error(self) -> None:
        text = csv_text("doc_001,a.pdf,,,,,,,,,,,,,,", "doc_001,b.pdf,,,,,,,,,,,,,,")
        with pytest.raises(ManifestError, match="'doc_001' appears 2 times"):
            parse_manifest(text)

    def test_more_cells_than_columns_is_an_error(self) -> None:
        with pytest.raises(ManifestError, match="more cells than columns"):
            parse_manifest(csv_text("doc_001,a.pdf" + "," * 20))


class TestCrossRowChecks:
    def test_gt_from_must_exist(self) -> None:
        problems = check_manifest(
            [row(doc_id="a"), row(doc_id="b", raw_files="b.pdf", gt_from="zzz")]
        )
        assert any("'zzz' is not in the manifest" in p for p in problems)

    def test_gt_from_must_point_at_the_owner_of_the_labels_not_at_another_borrower(self) -> None:
        rows = [
            row(doc_id="a"),
            row(doc_id="b", raw_files="b.pdf", gt_from="a"),
            row(doc_id="c", raw_files="c.pdf", gt_from="b"),
        ]
        problems = check_manifest(rows)
        assert len(problems) == 1
        assert "c: gt_from 'b' itself takes its GT from 'a'" in problems[0]

    def test_a_valid_borrow_has_no_problems(self) -> None:
        assert (
            check_manifest([row(doc_id="a"), row(doc_id="b", raw_files="b.pdf", gt_from="a")]) == []
        )

    def test_a_content_group_must_not_straddle_dev_and_test(self) -> None:
        rows = [
            row(doc_id="a", content_group="g", split="dev"),
            row(doc_id="b", raw_files="b.pdf", content_group="g", split="test"),
        ]
        (problem,) = content_group_split_conflicts(rows)
        assert "'g' straddles splits ['dev', 'test']" in problem

    def test_groups_without_a_split_yet_or_inside_one_split_are_fine(self) -> None:
        rows = [
            row(doc_id="a", content_group="g", split="dev"),
            row(doc_id="b", raw_files="b.pdf", content_group="g", split="dev"),
            row(doc_id="c", raw_files="c.pdf", content_group="h"),
            row(doc_id="d", raw_files="d.pdf", content_group="h", split="test"),
        ]
        assert content_group_split_conflicts(rows) == []

    def test_a_straddling_group_makes_the_whole_manifest_unreadable(self) -> None:
        text = csv_text(
            "doc_001,a.pdf,,,,g,,,,,,dev,,,,",
            "doc_002,b.pdf,,,,g,,,,,,test,,,,",
        )
        with pytest.raises(ManifestError, match="straddles"):
            parse_manifest(text)

    def test_missing_fields_per_stage(self) -> None:
        bare = row()
        assert missing_fields(bare, "ingest") == ["needs_redaction"]
        assert missing_fields(bare, "split") == ["doc_type", "content_group"]
        assert missing_fields(bare, "label") == [
            "doc_type",
            "content_group",
            "capture",
            "split",
            "label_mode",
        ]
        done = row(
            doc_type="receipt",
            content_group="g",
            capture="phone_photo",
            split="dev",
            label_mode="blind",
            needs_redaction="n",
        )
        for stage in ("ingest", "split", "label"):
            assert missing_fields(done, stage) == []  # type: ignore[arg-type]

    def test_docs_that_are_never_hand_labeled_need_no_label_mode(self) -> None:
        borrower = row(
            doc_id="b",
            raw_files="b.pdf",
            gt_from="a",
            doc_type="receipt",
            content_group="g",
            capture="phone_photo",
            split="dev",
        )
        assert "label_mode" not in missing_fields(borrower, "label")
        variant = ManifestRow.model_validate(
            {
                "doc_id": "v",
                "capture": "synthetic_degraded",
                "variant": "light",
                "gt_from": "a",
                "doc_type": "receipt",
                "content_group": "g",
                "split": "dev",
            }
        )
        assert "label_mode" not in missing_fields(variant, "label")
