from pathlib import Path

import pytest
from PIL import Image

from docextract.api.uploads import (
    FileTooLarge,
    TooManyPages,
    UnreadableUpload,
    UploadError,
    pages_from_uploads,
)
from files import A4, make_sized_pdf
from ingest_support import Env, doc, gray_png, settings

CONFIG = settings(dpi=40)


def jpeg(path: Path) -> Path:
    Image.new("RGB", (120, 90), (200, 30, 30)).save(path, "JPEG", quality=80)
    return path


def stored(env: Env, row_files: tuple[str, ...], **row: object) -> list[bytes]:
    """What scripts/ingest.py stores for these raw files, page by page."""
    manifest_row = doc(*row_files, **row)
    result = env.ingest(manifest_row, settings=CONFIG)
    assert result.status == "ingested"
    return [(env.out(manifest_row) / record.file).read_bytes() for record in result.pages]


def uploaded(tmp_path: Path, *files: bytes, **kwargs: object) -> list[bytes]:
    paths = pages_from_uploads(
        list(files),
        tmp_path / "work",
        max_pages=20,
        settings=CONFIG,
        **kwargs,  # type: ignore[arg-type]
    )
    return [path.read_bytes() for path in paths]


class TestParityWithIngestion:
    def test_a_pdf(self, env: Env, tmp_path: Path) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4, (420.0, 595.0)])
        assert uploaded(tmp_path, (env.raw / "a.pdf").read_bytes()) == stored(env, ("a.pdf",))

    def test_a_png(self, env: Env, tmp_path: Path) -> None:
        gray_png(env, "g.png", (90, 70))
        assert uploaded(tmp_path, (env.raw / "g.png").read_bytes()) == stored(env, ("g.png",))

    def test_a_jpeg_keeps_its_format(self, env: Env, tmp_path: Path) -> None:
        jpeg(env.raw / "j.jpg")
        pages = uploaded(tmp_path, (env.raw / "j.jpg").read_bytes())
        assert pages == stored(env, ("j.jpg",))
        assert pages[0][:2] == b"\xff\xd8"  # still a JPEG

    def test_a_document_of_several_files_in_upload_order(self, env: Env, tmp_path: Path) -> None:
        gray_png(env, "a.png", (80, 60))
        gray_png(env, "b.png", (50, 90))
        both = uploaded(
            tmp_path, (env.raw / "a.png").read_bytes(), (env.raw / "b.png").read_bytes()
        )
        assert both == stored(env, ("a.png", "b.png"))
        assert len(both) == 2 and both[0] != both[1]

    def test_a_page_range(self, env: Env, tmp_path: Path) -> None:
        make_sized_pdf(env.raw / "five.pdf", [A4] * 5)
        pages = uploaded(tmp_path, (env.raw / "five.pdf").read_bytes(), page_range="2-3")
        assert len(pages) == 2
        assert pages == stored(env, ("five.pdf",), page_range="2-3")

    def test_the_default_settings_are_the_ingestion_defaults(
        self, env: Env, tmp_path: Path
    ) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4])
        default = pages_from_uploads(
            [(env.raw / "a.pdf").read_bytes()], tmp_path / "w", max_pages=5
        )
        explicit = pages_from_uploads(
            [(env.raw / "a.pdf").read_bytes()],
            tmp_path / "x",
            max_pages=5,
            settings=settings(),
        )
        assert default[0].read_bytes() == explicit[0].read_bytes()


class TestRefusals:
    def test_garbage_is_unreadable(self, tmp_path: Path) -> None:
        with pytest.raises(UnreadableUpload, match="cannot open the image"):
            pages_from_uploads([b"this is not a document"], tmp_path / "w", max_pages=5)

    def test_a_pdf_header_without_a_pdf_is_unreadable(self, tmp_path: Path) -> None:
        with pytest.raises(UnreadableUpload, match="cannot open the PDF"):
            pages_from_uploads([b"%PDF-1.7 and then nothing"], tmp_path / "w", max_pages=5)

    def test_no_files_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(UnreadableUpload, match="no files"):
            pages_from_uploads([], tmp_path / "w", max_pages=5)

    def test_a_page_range_needs_exactly_one_file(self, env: Env, tmp_path: Path) -> None:
        gray_png(env, "a.png")
        data = (env.raw / "a.png").read_bytes()
        with pytest.raises(UnreadableUpload, match="exactly one file"):
            pages_from_uploads([data, data], tmp_path / "w", max_pages=5, page_range="1")

    def test_a_page_range_on_an_image_is_refused(self, env: Env, tmp_path: Path) -> None:
        gray_png(env, "a.png")
        with pytest.raises(UnreadableUpload, match="PDFs only"):
            pages_from_uploads(
                [(env.raw / "a.png").read_bytes()], tmp_path / "w", max_pages=5, page_range="1"
            )

    def test_a_page_range_past_the_end_is_refused(self, env: Env, tmp_path: Path) -> None:
        make_sized_pdf(env.raw / "two.pdf", [A4, A4])
        with pytest.raises(UnreadableUpload, match="file 1"):
            pages_from_uploads(
                [(env.raw / "two.pdf").read_bytes()], tmp_path / "w", max_pages=5, page_range="3"
            )

    def test_an_oversize_file_is_refused_before_anything_is_written(self, tmp_path: Path) -> None:
        with pytest.raises(FileTooLarge, match="file 2 is 2000 bytes; the limit is 1000"):
            pages_from_uploads(
                [b"x" * 10, b"y" * 2000], tmp_path / "w", max_pages=5, max_bytes=1000
            )
        assert not (tmp_path / "w").exists()

    def test_too_many_pages_is_refused_without_rendering(self, env: Env, tmp_path: Path) -> None:
        make_sized_pdf(env.raw / "long.pdf", [A4] * 21)
        with pytest.raises(TooManyPages, match="21 pages, but at most 20"):
            pages_from_uploads([(env.raw / "long.pdf").read_bytes()], tmp_path / "w", max_pages=20)
        assert not (tmp_path / "w").exists()  # not even the files were written

    def test_the_limit_counts_the_selected_pages_and_all_files(
        self, env: Env, tmp_path: Path
    ) -> None:
        make_sized_pdf(env.raw / "five.pdf", [A4] * 5)
        pdf = (env.raw / "five.pdf").read_bytes()
        assert len(pages_from_uploads([pdf], tmp_path / "a", max_pages=2, page_range="1-2")) == 2
        gray_png(env, "a.png")
        image = (env.raw / "a.png").read_bytes()
        with pytest.raises(TooManyPages, match="3 pages"):
            pages_from_uploads([image, image, image], tmp_path / "b", max_pages=2)

    def test_every_refusal_is_an_upload_error_with_a_code(self) -> None:
        assert UnreadableUpload.code == "unreadable_file"
        assert FileTooLarge.code == "file_too_large" and TooManyPages.code == "too_many_pages"
        assert all(
            issubclass(e, UploadError) for e in (UnreadableUpload, FileTooLarge, TooManyPages)
        )


class TestOnDisk:
    def test_nothing_is_written_outside_the_work_directory(self, env: Env, tmp_path: Path) -> None:
        gray_png(env, "a.png")
        before = {p.name for p in tmp_path.iterdir()}
        pages_from_uploads(
            [(env.raw / "a.png").read_bytes()], tmp_path / "work", max_pages=5, settings=CONFIG
        )
        assert {p.name for p in tmp_path.iterdir()} - before == {"work"}

    def test_files_are_named_by_position_never_by_anything_the_client_sent(
        self, env: Env, tmp_path: Path
    ) -> None:
        gray_png(env, "a.png")
        data = (env.raw / "a.png").read_bytes()
        pages_from_uploads([data, data], tmp_path / "work", max_pages=5, settings=CONFIG)
        assert sorted(p.name for p in (tmp_path / "work" / "in").iterdir()) == ["f0.bin", "f1.bin"]
        assert sorted(p.name for p in (tmp_path / "work" / "pages").iterdir()) == [
            "p1.png",
            "p2.png",
        ]
