"""Check the ground-truth files against the manifest and print how much is labeled.

Usage: python -m uv run python scripts/check_ground_truth.py [--manifest FILE]
           [--ground-truth-dir DIR] [--drafts-dir DIR]

Checks every ground_truth/<doc_id>.json: it loads (unknown keys, bad numbers, naive timestamps
are errors), its doc_id is its file name, the document is in the manifest, a document that takes
its labels from another (gt_from) has no file of its own, the label_mode in the file is the one
in the manifest, and a BLIND document has no model draft in ground_truth/drafts/.

Exit codes: 0 consistent, 1 problems found, 2 the manifest cannot be read.
"""

import argparse
import sys
from pathlib import Path

from docextract.ingest.manifest import ManifestError, read_manifest
from docextract.paths import DatasetPaths
from evalkit.gt_store import format_report


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(description="Check the ground truth against the manifest.")
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--ground-truth-dir", type=Path, default=paths.ground_truth_dir)
    parser.add_argument("--drafts-dir", type=Path)
    args = parser.parse_args()
    drafts = args.drafts_dir or args.ground_truth_dir / "drafts"
    try:
        rows = read_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        print(f"The manifest cannot be read: {exc}", file=sys.stderr)
        return 2
    report, problems = format_report(rows, args.ground_truth_dir, drafts)
    print(report)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
