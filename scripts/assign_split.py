"""Assign dev/test splits and pick the blind documents, writing them into data/manifest.csv.

Usage: python -m uv run python scripts/assign_split.py [--dry-run] [--force]
           [--manifest FILE] [--ground-truth-dir DIR]
           [--seed TEXT] [--dev-fraction 0.35] [--blind-count 10]

The unit is the content_group, stratified by (real | constructed, doc_type): about 35% of the
groups go to dev, the rest to test, and no group is ever split across the two. A doc_type with a
single group goes to test. Degraded variants and gt_from duplicates follow their group.
Assignments already in the manifest are NEVER moved: re-running only fills in new rows (so the
test set stays put). --force starts from scratch, and refuses to touch a document whose ground
truth already exists in --ground-truth-dir.

Then 8-10 real test documents (--blind-count, default 10) are marked label_mode=blind, spread
round-robin over doc types, whole content groups at a time. Every other hand-labeled document
with a blank label_mode becomes `prelabel`. Constructed and degraded documents get no label_mode.

--dry-run prints the plan and writes nothing.
Exit codes: 0 done, 1 the manifest cannot be split yet (the reasons are listed), 2 the manifest
cannot be read.
"""

import argparse
import sys
from pathlib import Path

from docextract.ingest.manifest import ManifestError, read_manifest, write_manifest
from docextract.ingest.splits import (
    DEFAULT_BLIND_COUNT,
    DEFAULT_DEV_FRACTION,
    DEFAULT_SEED,
    SplitError,
    assign_splits,
    format_plan,
)
from docextract.paths import DatasetPaths


def documents_with_ground_truth(directory: Path) -> set[str]:
    """doc_ids that already have a final label (drafts do not count: nobody has verified them)."""
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob("*.json")}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(description="Assign dev/test splits and blind documents.")
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--ground-truth-dir", type=Path, default=paths.ground_truth_dir)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--dev-fraction", default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--blind-count", type=int, default=DEFAULT_BLIND_COUNT)
    parser.add_argument("--force", action="store_true", help="re-assign everything from scratch")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args()

    try:
        rows = read_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        print(f"The manifest cannot be read: {exc}", file=sys.stderr)
        return 2
    try:
        plan = assign_splits(
            rows,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
            blind_count=args.blind_count,
            force=args.force,
            locked=documents_with_ground_truth(args.ground_truth_dir),
        )
    except SplitError as exc:
        print("The manifest cannot be split yet:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    except ValueError as exc:  # a dev fraction outside (0, 1)
        print(exc, file=sys.stderr)
        return 1

    print(format_plan(plan, seed=args.seed, dev_fraction=args.dev_fraction))
    if args.force:
        print("--force: assignments were recomputed from scratch; review the changes below.")
    for change in plan.changes if (args.force or args.dry_run) else []:
        print(f"  {change.doc_id:<14}{change.field:<11}{change.old or '-'} -> {change.new}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    write_manifest(args.manifest, plan.rows)
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
