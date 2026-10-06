"""Make light / medium / heavy degraded variants of born-digital documents.

Usage: python -m uv run python scripts/degrade_docs.py (--doc DOC_ID ... | --all)
           [--levels light,medium,heavy] [--seed TEXT] [--backend auto|augraphy|opencv]
           [--manifest FILE] [--pages-dir DIR] [--jpeg-quality N] [--force] [--check]

A variant `<doc_id>-deg-<level>` is written next to its source in data/pages, from the source's
STORED (already redacted) pages: same page count and size, pixel transforms only, deterministic
for a given seed. Its manifest row (capture=synthetic_degraded, variant, gt_from the source,
same content_group, split and doc_type) is added right after the source's row, so it inherits
the source's ground truth and always lands on the same side of the dev/test split.

Only born_digital documents qualify; they must have been ingested first (scripts/ingest.py).
Augraphy is used when it imports (the backend is recorded per variant), else OpenCV.
Idempotent: a variant whose inputs have not changed is left alone.

--check reports what would be made without writing pages or the manifest.
Exit codes: 0 done, 1 a variant could not be made, 2 bad arguments, unreadable manifest or an
unusable backend.
"""

import argparse
import sys
from pathlib import Path

from docextract.ingest.degrade import (
    DEFAULT_SEED,
    LEVELS,
    DegradeError,
    add_variant_rows,
    degrade_document,
    resolve_backend,
)
from docextract.ingest.manifest import (
    ManifestError,
    ManifestRow,
    Variant,
    read_manifest,
    write_manifest,
)
from docextract.paths import DatasetPaths


def parse_levels(text: str) -> list[Variant]:
    chosen = [part.strip() for part in text.split(",")]
    unknown = [name for name in chosen if name not in LEVELS]
    if unknown:
        raise ValueError(f"unknown level {unknown[0]!r} (choose from {', '.join(LEVELS)})")
    return [level for level in LEVELS if level in chosen]  # canonical order, no repeats


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(
        description="Make degraded variants of born-digital documents."
    )
    parser.add_argument("--doc", action="append", default=[], help="a source doc_id (repeatable)")
    parser.add_argument(
        "--all", action="store_true", help="every born_digital, non-derived document"
    )
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--backend", choices=["auto", "augraphy", "opencv"], default="auto")
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--pages-dir", type=Path, default=paths.pages_dir)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--force", action="store_true", help="redo variants even if unchanged")
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    if bool(args.doc) == args.all:
        print("Choose the documents: --doc DOC_ID (repeatable) or --all", file=sys.stderr)
        return 2
    try:
        levels = parse_levels(args.levels)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        rows = read_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        print(f"The manifest cannot be read: {exc}", file=sys.stderr)
        return 2
    by_id = {row.doc_id: row for row in rows}
    unknown = [doc for doc in args.doc if doc not in by_id]
    if unknown:
        print(f"Not in the manifest: {', '.join(unknown)}", file=sys.stderr)
        return 2
    try:
        backend, note = resolve_backend(args.backend)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    sources: list[ManifestRow] = (
        [r for r in rows if not r.is_derived and r.capture == "born_digital"]
        if args.all
        else [by_id[doc] for doc in args.doc]
    )
    print(
        f"degrade: {note}, seed {args.seed!r}, levels {', '.join(levels)}"
        f"{'  (check only)' if args.check else ''}"
    )
    failures = 0
    made: list[tuple[ManifestRow, Variant]] = []
    for source in sources:
        for level in levels:
            try:
                result = degrade_document(
                    source,
                    level,
                    pages_dir=args.pages_dir,
                    seed=args.seed,
                    backend=backend,
                    jpeg_quality=args.jpeg_quality,
                    force=args.force,
                    dry_run=args.check,
                )
            except DegradeError as exc:
                failures += 1
                print(f"  FAILED    {exc}")
                continue
            made.append((source, level))
            print(f"  {result.status:<12}{result.doc_id}  {len(result.pages)} page(s)")
    if made and not args.check:
        try:
            write_manifest(args.manifest, add_variant_rows(rows, made))
        except DegradeError as exc:
            print(f"  FAILED    {exc}")
            return 1
        print(f"updated {args.manifest}")
    print(f"done: {len(made)} variant(s), {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
