import json

import pytest

from docextract.ingest.degrade import degrade_document, variant_id
from docextract.ingest.manifest import ManifestRow
from docextract.ingest.stored import StoredPagesError, stored_pages
from files import A4, make_sized_pdf
from ingest_support import Env, doc, gray_png, settings


def ingest_pdf(env: Env, pages: int) -> ManifestRow:
    make_sized_pdf(env.raw / "doc_001.pdf", [A4] * pages)
    row = doc("doc_001.pdf", capture="born_digital")
    env.ingest(row, settings=settings(dpi=40))
    return row


def ingest_png(env: Env) -> ManifestRow:
    gray_png(env, "doc_001.png")
    row = doc("doc_001.png", capture="born_digital")
    env.ingest(row)
    return row


class TestIngestedDocuments:
    def test_the_pages_come_back_in_page_order(self, env: Env) -> None:
        ingest_pdf(env, 3)
        assert [p.name for p in stored_pages("doc_001", env.pages)] == [
            "p1.png",
            "p2.png",
            "p3.png",
        ]
        assert all(p.parent == env.pages / "doc_001" for p in stored_pages("doc_001", env.pages))

    def test_the_raw_files_are_never_needed(self, env: Env) -> None:
        ingest_png(env)
        for raw in env.raw.iterdir():
            raw.unlink()
        assert [p.name for p in stored_pages("doc_001", env.pages)] == ["p1.png"]

    def test_a_document_that_was_not_ingested_is_refused(self, env: Env) -> None:
        with pytest.raises(StoredPagesError, match=r"^doc_001: not ingested yet \(no ingest\.json"):
            stored_pages("doc_001", env.pages)

    def test_a_missing_page_is_refused(self, env: Env) -> None:
        ingest_pdf(env, 2)
        (env.pages / "doc_001" / "p2.png").unlink()
        with pytest.raises(StoredPagesError, match="^doc_001: stored page p2.png is missing$"):
            stored_pages("doc_001", env.pages)

    def test_a_page_that_changed_since_ingestion_is_refused(self, env: Env) -> None:
        ingest_pdf(env, 2)
        page = env.pages / "doc_001" / "p2.png"
        page.write_bytes(page.read_bytes() + b"x")
        with pytest.raises(
            StoredPagesError, match="^doc_001: stored page p2.png no longer matches ingest.json$"
        ):
            stored_pages("doc_001", env.pages)

    def test_a_malformed_sidecar_is_refused_not_a_crash(self, env: Env) -> None:
        ingest_png(env)
        sidecar = env.pages / "doc_001" / "ingest.json"
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        del data["pages"][0]["sha256"]
        sidecar.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(StoredPagesError, match="^doc_001: ingest.json in .* is malformed$"):
            stored_pages("doc_001", env.pages)

    def test_pages_must_be_listed_one_to_n_in_order(self, env: Env) -> None:
        ingest_pdf(env, 2)
        sidecar = env.pages / "doc_001" / "ingest.json"
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["pages"].reverse()  # every file intact, but page 2 would be cited as page 1
        sidecar.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(StoredPagesError, match=r"does not list pages 1\.\.2 in order"):
            stored_pages("doc_001", env.pages)


class TestDegradedVariants:
    def make_variant(self, env: Env, pages: int = 2) -> str:
        source = ingest_pdf(env, pages)
        degrade_document(source, "light", pages_dir=env.pages, backend="opencv")
        return variant_id("doc_001", "light")

    def test_a_variant_s_pages_come_from_degrade_json(self, env: Env) -> None:
        name = self.make_variant(env)
        pages = stored_pages(name, env.pages, derived=True)
        assert [p.name for p in pages] == ["p1.png", "p2.png"]
        assert all(p.parent == env.pages / name for p in pages)

    def test_a_variant_read_as_an_ingested_document_is_refused(self, env: Env) -> None:
        name = self.make_variant(env)
        with pytest.raises(StoredPagesError, match=f"^{name}: not ingested yet"):
            stored_pages(name, env.pages)

    def test_a_variant_that_was_not_generated_is_refused(self, env: Env) -> None:
        ingest_pdf(env, 1)
        with pytest.raises(
            StoredPagesError, match=r"^doc_001-deg-light: not generated yet \(no degrade\.json"
        ):
            stored_pages("doc_001-deg-light", env.pages, derived=True)

    def test_a_changed_variant_page_is_refused(self, env: Env) -> None:
        name = self.make_variant(env)
        page = env.pages / name / "p1.png"
        page.write_bytes(page.read_bytes() + b"x")
        with pytest.raises(StoredPagesError, match="p1.png no longer matches degrade.json$"):
            stored_pages(name, env.pages, derived=True)


def test_prelabel_keeps_its_own_error_type_and_message(env: Env) -> None:
    from evalkit.prelabel import PrelabelRefused
    from evalkit.prelabel import stored_pages as prelabel_pages

    with pytest.raises(PrelabelRefused, match=r"^doc_001: not ingested yet \(no ingest\.json"):
        prelabel_pages("doc_001", env.pages)
