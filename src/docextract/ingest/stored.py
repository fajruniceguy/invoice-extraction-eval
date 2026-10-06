"""A document's STORED pages (data/pages/<doc_id>/), in page order, checked against its sidecar.

Ingested documents record their pages in `ingest.json`, degraded variants in `degrade.json`. Only
those pages are ever handed on (to the extractor, to the pre-labeler): never a raw file, never an
unredacted preview. A page that is missing, or whose bytes no longer match the hash recorded for it,
is refused, because what would reach a model is then not what ingestion (and its redaction) wrote.
"""

from pathlib import Path

from docextract.ingest.degrade import SIDECAR as DEGRADE_SIDECAR
from docextract.ingest.degrade import read_sidecar_named
from docextract.ingest.pipeline import SIDECAR as INGEST_SIDECAR
from docextract.ingest.pipeline import read_sidecar, sha256_hex


class StoredPagesError(Exception):
    """The document's stored pages cannot be used (and why)."""


def stored_pages(doc_id: str, pages_dir: Path, *, derived: bool = False) -> list[Path]:
    """The stored pages of `doc_id`, in page order. `derived` = a degraded variant, whose pages are
    recorded in degrade.json instead of ingest.json."""
    folder = pages_dir / doc_id
    if derived:
        sidecar, name, missing = read_sidecar_named(folder), DEGRADE_SIDECAR, "not generated yet"
    else:
        sidecar, name, missing = read_sidecar(folder), INGEST_SIDECAR, "not ingested yet"
    if not sidecar or not sidecar.get("pages"):
        raise StoredPagesError(f"{doc_id}: {missing} (no {name} in {folder})")
    try:
        entries = [
            (int(page["n"]), str(page["file"]), str(page["sha256"])) for page in sidecar["pages"]
        ]
    except (KeyError, TypeError, ValueError):
        raise StoredPagesError(f"{doc_id}: {name} in {folder} is malformed") from None
    if [number for number, _, _ in entries] != list(range(1, len(entries) + 1)):
        # page numbers are what the model cites as evidence: they must be 1..N in order
        raise StoredPagesError(f"{doc_id}: {name} does not list pages 1..{len(entries)} in order")
    paths: list[Path] = []
    for _, file, recorded in entries:
        path = folder / file
        try:
            actual = sha256_hex(path.read_bytes())
        except OSError:
            raise StoredPagesError(f"{doc_id}: stored page {path.name} is missing") from None
        if actual != recorded:
            raise StoredPagesError(f"{doc_id}: stored page {path.name} no longer matches {name}")
        paths.append(path)
    return paths
