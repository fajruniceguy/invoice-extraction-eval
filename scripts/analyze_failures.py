"""How a run fails, by kind: what the model said before a field was withheld, which fields
are wrong, which documents the missed and spurious rows come from. Offline, deterministic.

  python -m uv run python scripts/analyze_failures.py outputs/runs/<run_id> [--out F] [--notes F]

Prints the markdown report, or writes it to --out. --notes appends a hand-written "Reading"
section (the tables are data; the interpretation lives in a file you edit). Exit 0 done, 2 the
run or the labels cannot be read.
"""

import argparse
import sys
from pathlib import Path

from docextract.paths import DatasetPaths
from evalkit.failures import FailureError, collect, to_markdown
from evalkit.gt_store import GroundTruthError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse how an extraction run fails.")
    parser.add_argument("run_dir", type=Path, help="outputs/runs/<run_id>")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    parser.add_argument("--notes", type=Path, default=None, help="a hand-written Reading section")
    parser.add_argument("--manifest", type=Path, default=None, help="the dataset manifest")
    parser.add_argument("--gt-dir", type=Path, default=None, help="the ground-truth directory")
    args = parser.parse_args(argv)

    paths = DatasetPaths.default()
    if not args.run_dir.is_dir():
        print(f"{args.run_dir}: not a directory", file=sys.stderr)
        return 2
    notes = None
    if args.notes is not None:
        try:
            notes = args.notes.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"{args.notes}: cannot be read ({exc})", file=sys.stderr)
            return 2
    try:
        result = collect(
            args.run_dir,
            manifest_path=args.manifest or paths.manifest,
            labels_dir=args.gt_dir or paths.ground_truth_dir,
        )
    except (FailureError, GroundTruthError) as exc:
        print(f"{args.run_dir}: {exc}", file=sys.stderr)
        return 2
    report = to_markdown(result, notes=notes)
    if args.out is None:
        print(report, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8", newline="\n")
        print(f"{args.run_dir.name}: {len(result.cases)} fields analysed -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
