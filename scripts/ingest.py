"""Turn manifest documents into normalized pages: data/pages/<doc_id>/p<N>.png|jpg.

Usage: python -m uv run python scripts/ingest.py [--doc DOC_ID ...] [--check] [--force]
           [--manifest FILE] [--raw-dir DIR] [--pages-dir DIR] [--redactions FILE]
           [--dpi N] [--max-long-edge N] [--jpeg-quality N] [--accept-resize]
       python -m uv run python scripts/ingest.py --preview [--doc DOC_ID ...] [--preview-dir DIR]

Every page is made upright from its EXIF orientation, stripped of ALL metadata, size-capped
(INGEST_MAX_LONG_EDGE, default 2400; elongated receipts are capped by area), redacted with the
boxes in data/redactions.yaml, and stored as PNG (PDFs, PNG sources) or JPEG q92 (photos).
Idempotent: unchanged inputs are skipped.

A document with a blank needs_redaction, or needs_redaction=y without boxes, is refused: an
unredacted page is never stored in data/pages. Boxes are pixel coordinates on the stored page; if
the DPI or size cap changes under boxes that were not touched, the run refuses (re-measure them,
or --accept-resize when they are still right).

--preview writes the UNREDACTED normalized pages to data/private/preview/ (same pixel space as
the stored pages) so the box coordinates can be measured before any box exists. It never touches
redactions.yaml or the manifest's redaction decision, and refuses a directory that overlaps
--pages-dir. Do not commit or share what it writes.

--check reports what would be ingested without writing anything.
Exit codes: 0 all fine, 1 at least one document failed, 2 the manifest or redactions file could
not be read (or is inconsistent with the manifest), or the preview directory is not allowed.
"""

import argparse
import sys
from pathlib import Path

from docextract.config import IngestSettings
from docextract.ingest.manifest import ManifestError, ManifestRow, read_manifest
from docextract.ingest.pipeline import IngestError, ingest_document
from docextract.ingest.preview import PreviewError, check_preview_dir, preview_document
from docextract.ingest.redact import RedactionError
from docextract.ingest.redactions import (
    Redactions,
    boxes_of,
    check_against_manifest,
    pinned_sizes_of,
    read_redactions,
)
from docextract.paths import DatasetPaths


def load_redactions(path: Path, explicit: bool, rows: list[ManifestRow]) -> Redactions | None:
    """The redactions, or None after printing why the file cannot be used."""
    try:
        redactions = read_redactions(path, must_exist=explicit)
    except RedactionError as exc:
        print(f"The redactions file cannot be used: {exc}", file=sys.stderr)
        return None
    problems = check_against_manifest(redactions, rows)
    for problem in problems:
        print(f"{path.name}: {problem}", file=sys.stderr)
    return None if problems else redactions


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(description="Normalize raw documents into stored pages.")
    parser.add_argument("--doc", action="append", default=[], help="only this doc_id (repeatable)")
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--raw-dir", type=Path, default=paths.raw_dir)
    parser.add_argument("--pages-dir", type=Path, default=paths.pages_dir)
    parser.add_argument("--redactions", type=Path, help=f"default: {paths.redactions}")
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    parser.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    parser.add_argument(
        "--accept-resize",
        action="store_true",
        help="keep untouched boxes although their page changed size (DPI or cap changed)",
    )
    parser.add_argument(
        "--preview", action="store_true", help="write UNREDACTED pages to --preview-dir"
    )
    parser.add_argument("--preview-dir", type=Path, default=paths.preview_dir)
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--max-long-edge", type=int)
    parser.add_argument("--jpeg-quality", type=int)
    args = parser.parse_args()
    if args.preview and args.check:
        parser.error("--preview and --check cannot be combined")

    try:
        rows = read_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        print(f"The manifest cannot be read: {exc}", file=sys.stderr)
        return 2

    overrides = {
        key: value
        for key, value in (
            ("dpi", args.dpi),
            ("max_long_edge", args.max_long_edge),
            ("jpeg_quality", args.jpeg_quality),
        )
        if value is not None
    }
    settings = IngestSettings(**overrides)
    unknown = [doc for doc in args.doc if doc not in {row.doc_id for row in rows}]
    if unknown:
        print(f"Not in the manifest: {', '.join(unknown)}", file=sys.stderr)
        return 2

    redactions: Redactions = {}
    if args.preview:
        try:  # before anything is announced or attempted, even if no document is selected
            check_preview_dir(args.preview_dir, args.pages_dir)
        except PreviewError as exc:
            print(exc, file=sys.stderr)
            return 2
    else:  # a preview is how the file gets written: it must not need it valid
        loaded = load_redactions(
            args.redactions or paths.redactions, args.redactions is not None, rows
        )
        if loaded is None:
            return 2
        redactions = loaded

    failures = 0
    counts = {"ingested": 0, "unchanged": 0, "would_ingest": 0, "previewed": 0}
    mode = "PREVIEW, UNREDACTED" if args.preview else ("check only" if args.check else "")
    print(
        f"ingest: dpi={settings.dpi} max_long_edge={settings.max_long_edge} "
        f"jpeg_quality={settings.jpeg_quality}{f'  ({mode})' if mode else ''}"
    )
    if args.preview:
        print(f"  writing to {args.preview_dir}: never commit or share these files")
    for row in rows:
        if args.doc and row.doc_id not in args.doc:
            continue
        if row.is_derived:
            continue  # variants are generated from their source's pages by degrade_docs.py
        try:
            if args.preview:
                records = preview_document(
                    row,
                    raw_dir=args.raw_dir,
                    preview_dir=args.preview_dir,
                    pages_dir=args.pages_dir,
                    settings=settings,
                )
                status, pages, note = "previewed", records, ""
            else:
                entry = redactions.get(row.doc_id, {})
                result = ingest_document(
                    row,
                    raw_dir=args.raw_dir,
                    out_dir=args.pages_dir / row.doc_id,
                    settings=settings,
                    boxes=boxes_of(entry),
                    pinned_sizes=pinned_sizes_of(entry),
                    accept_resize=args.accept_resize,
                    force=args.force,
                    dry_run=args.check,
                )
                status, pages = result.status, result.pages
                count = sum(len(found.boxes) for found in entry.values())
                note = f"  [{count} redaction box(es)]" if count else ""
        except PreviewError as exc:
            print(exc, file=sys.stderr)
            return 2
        except IngestError as exc:
            failures += 1
            print(f"  FAILED    {exc}")
            continue
        counts[status] += 1
        sizes = ", ".join(f"{p.size[0]}x{p.size[1]} {p.format}" for p in pages[:3])
        more = f", +{len(pages) - 3} more" if len(pages) > 3 else ""
        detail = f"{len(pages)} page(s): {sizes}{more}" if pages else ""
        print(f"  {status:<10}{row.doc_id}  {detail}{note}")
    summary = (
        f"{counts['previewed']} previewed"
        if args.preview
        else f"{counts['ingested']} ingested, {counts['unchanged']} unchanged, "
        f"{counts['would_ingest']} would be ingested"
    )
    print(f"done: {summary}, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
