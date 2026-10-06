"""Scan data/private/raw/ and write (or extend) data/manifest.csv.

Usage: python -m uv run python scripts/manifest_init.py [--dry-run] [--raw-dir DIR]
       [--manifest FILE]

One row per file, with everything that can be inferred filled in (format, page count in the
report, capture and language only on strong evidence); you fill in the rest by hand. Safe to
re-run: existing rows are never touched, only files not yet referenced are appended.

Exit codes: 0 done, 2 the existing manifest could not be read (nothing is written).
"""

import argparse
import sys
from pathlib import Path

from docextract.ingest.manifest import ManifestError, read_manifest, write_manifest
from docextract.ingest.skeleton import build_skeleton, format_report
from docextract.paths import DatasetPaths


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(description="Generate or extend the manifest skeleton.")
    parser.add_argument("--raw-dir", type=Path, default=paths.raw_dir)
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--dry-run", action="store_true", help="print the report, write nothing")
    args = parser.parse_args()

    existing = []
    if args.manifest.is_file():
        try:
            existing = read_manifest(args.manifest)
        except ManifestError as exc:
            print("The existing manifest cannot be read; nothing was written:", file=sys.stderr)
            for problem in exc.problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2

    result = build_skeleton(args.raw_dir, existing)
    if not args.dry_run and result.new_ids:
        write_manifest(args.manifest, result.rows)
    print(format_report(result, args.raw_dir, args.manifest, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
