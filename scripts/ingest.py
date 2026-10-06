"""Turn manifest documents into normalized pages: data/pages/<doc_id>/p<N>.png|jpg.

Usage: python -m uv run python scripts/ingest.py [--doc DOC_ID ...] [--check] [--force]
           [--manifest FILE] [--raw-dir DIR] [--pages-dir DIR]
           [--dpi N] [--max-long-edge N] [--jpeg-quality N]

Every page is made upright from its EXIF orientation, stripped of ALL metadata, size-capped
(INGEST_MAX_LONG_EDGE, default 2400; elongated receipts are capped by area), redacted, and stored
as PNG (PDFs, PNG sources) or JPEG q92 (photos). Idempotent: unchanged inputs are skipped.

A document with a blank needs_redaction, or needs_redaction=y without boxes, is refused: an
unredacted page is never stored in data/pages.

--check reports what would be ingested without writing anything.
Exit codes: 0 all fine, 1 at least one document failed, 2 the manifest could not be read.
"""

import argparse
import sys
from pathlib import Path

from docextract.config import IngestSettings
from docextract.ingest.manifest import ManifestError, read_manifest
from docextract.ingest.pipeline import IngestError, ingest_document
from docextract.paths import DatasetPaths


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(description="Normalize raw documents into stored pages.")
    parser.add_argument("--doc", action="append", default=[], help="only this doc_id (repeatable)")
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--raw-dir", type=Path, default=paths.raw_dir)
    parser.add_argument("--pages-dir", type=Path, default=paths.pages_dir)
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    parser.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--max-long-edge", type=int)
    parser.add_argument("--jpeg-quality", type=int)
    args = parser.parse_args()

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

    failures = 0
    counts = {"ingested": 0, "unchanged": 0, "would_ingest": 0}
    print(
        f"ingest: dpi={settings.dpi} max_long_edge={settings.max_long_edge} "
        f"jpeg_quality={settings.jpeg_quality}{'  (check only)' if args.check else ''}"
    )
    for row in rows:
        if args.doc and row.doc_id not in args.doc:
            continue
        if row.is_derived:
            continue  # variants are generated from their source's pages by degrade_docs.py
        try:
            result = ingest_document(
                row,
                raw_dir=args.raw_dir,
                out_dir=args.pages_dir / row.doc_id,
                settings=settings,
                boxes=None,
                force=args.force,
                dry_run=args.check,
            )
        except IngestError as exc:
            failures += 1
            print(f"  FAILED    {exc}")
            continue
        counts[result.status] += 1
        sizes = ", ".join(f"{p.size[0]}x{p.size[1]} {p.format}" for p in result.pages[:3])
        more = f", +{len(result.pages) - 3} more" if len(result.pages) > 3 else ""
        detail = f"{len(result.pages)} page(s): {sizes}{more}" if result.pages else ""
        print(f"  {result.status:<10}{row.doc_id}  {detail}")
    print(
        f"done: {counts['ingested']} ingested, {counts['unchanged']} unchanged, "
        f"{counts['would_ingest']} would be ingested, {failures} failed"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
