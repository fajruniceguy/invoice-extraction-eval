"""Construct three multi-page rupiah invoices in three layouts, with ground truth.

Usage: python -m uv run python scripts/construct_invoices.py [--check] [--force] [--seed TEXT]
           [--raw-dir DIR] [--ground-truth-dir DIR] [--manifest FILE] [--changelog FILE]

Writes, for cons_001..cons_003 (layouts classic, modern, compact):
  data/private/raw/constructed/<doc_id>.pdf   30-60 line items over 2-3 pages; subtotal, discount,
                                              PPN and total on the last page only; the modern
                                              layout repeats a "(lanjutan)" header
  ground_truth/<doc_id>.json                  the label AS PRINTED (a field the layout does not
                                              print is null), logged in ground_truth/CHANGELOG.md
  data/manifest.csv                           a row per document, capture=constructed
Then ingest them like any other document (scripts/ingest.py) and split them (assign_split.py).

Deterministic: the same seed gives byte-identical PDFs and labels. A label that already exists is
left alone when identical; if it differs (another seed) the run is refused, because ground truth
is never edited silently: --force replaces it and logs the change.

--check reports what would happen and writes nothing.
Exit codes: 0 done, 1 a ground-truth or manifest conflict, 2 unreadable manifest.
"""

import argparse
import sys
from pathlib import Path

from docextract.ingest.manifest import ManifestError
from docextract.paths import DatasetPaths
from evalkit.construct import DEFAULT_SEED
from evalkit.construct_io import ConstructError, construct


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(description="Construct multi-page invoices with ground truth.")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--raw-dir", type=Path, default=paths.raw_dir)
    parser.add_argument("--ground-truth-dir", type=Path, default=paths.ground_truth_dir)
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--changelog", type=Path, default=paths.changelog)
    parser.add_argument(
        "--force", action="store_true", help="replace differing ground truth (logged)"
    )
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    try:
        actions = construct(
            raw_dir=args.raw_dir,
            ground_truth_dir=args.ground_truth_dir,
            manifest=args.manifest,
            changelog=args.changelog,
            seed=args.seed,
            force=args.force,
            check=args.check,
        )
    except ManifestError as exc:
        print(f"The manifest cannot be read: {exc}", file=sys.stderr)
        return 2
    except ConstructError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"construct: seed {args.seed!r}{'  (check only)' if args.check else ''}")
    for action in actions:
        print(
            f"  {action.doc_id}  {action.items} items, {action.pages} pages | "
            f"pdf {action.pdf} | ground truth {action.ground_truth} | manifest {action.manifest}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
