"""Take the `hallucinated_value` errors of a run apart, and audit documents that share a label.

  python -m uv run python scripts/analyze_hallucinations.py RUN [--out F] [--review F] [--notes F]

Prints the markdown report, or writes it to --out. --notes appends a hand-written "Reading"
section. --review writes the list for a human (it holds evidence quotes from the redacted pages,
so it is refused inside docs/ or results/: write it under outputs/, which is gitignored).
Offline; changes no label and no score. Exit 0 done, 2 the run, the labels or an argument cannot
be used.
"""

import argparse
import sys
from pathlib import Path

from docextract.paths import DatasetPaths, find_repo_root
from evalkit.gt_store import GroundTruthError
from evalkit.hallucinations import (
    HallucinationError,
    collect,
    review_markdown,
    to_markdown,
)

TRACKED_FOLDERS = ("docs", "results", "ground_truth", "data", "src", "scripts", "tests")


def inside_tracked_folder(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    return any(resolved.is_relative_to((root / name).resolve()) for name in TRACKED_FOLDERS)


def main(argv: list[str] | None = None) -> int:
    root = find_repo_root()
    parser = argparse.ArgumentParser(description="Analyse the hallucinated values of a run.")
    parser.add_argument("run_dir", type=Path, help="outputs/runs/<run_id> or results/<run_id>")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    parser.add_argument("--review", type=Path, default=None, help="write the review list here")
    parser.add_argument("--notes", type=Path, default=None, help="a hand-written Reading section")
    parser.add_argument("--manifest", type=Path, default=None, help="the dataset manifest")
    parser.add_argument("--gt-dir", type=Path, default=None, help="the ground-truth directory")
    args = parser.parse_args(argv)

    paths = DatasetPaths.default()
    if not args.run_dir.is_dir():
        print(f"{args.run_dir}: not a directory", file=sys.stderr)
        return 2
    if args.review is not None and inside_tracked_folder(args.review, root):
        print(
            f"--review {args.review}: it holds evidence quotes, so it must not go inside "
            f"{', '.join(TRACKED_FOLDERS)}; use outputs/",
            file=sys.stderr,
        )
        return 2
    notes = None
    if args.notes is not None:
        try:
            notes = args.notes.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"{args.notes}: cannot be read ({exc})", file=sys.stderr)
            return 2
    try:
        report = collect(
            args.run_dir,
            manifest_path=args.manifest or paths.manifest,
            labels_dir=args.gt_dir or paths.ground_truth_dir,
        )
    except (HallucinationError, GroundTruthError) as exc:
        print(f"{args.run_dir}: {exc}", file=sys.stderr)
        return 2
    text = to_markdown(report, notes=notes)
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(
            f"{report.run_id}: {len(report.fields)} scalar + {len(report.cells)} cell "
            f"hallucinated value(s) -> {args.out}"
        )
    if args.review is not None:
        args.review.parent.mkdir(parents=True, exist_ok=True)
        args.review.write_text(review_markdown(report), encoding="utf-8", newline="\n")
        print(f"review list ({len(report.items)} case(s)) -> {args.review}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
