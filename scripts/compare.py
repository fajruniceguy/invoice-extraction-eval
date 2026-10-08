"""Compare two extraction runs offline: did the candidate get worse than the baseline?

  python -m uv run python scripts/compare.py outputs/runs/<baseline> outputs/runs/<candidate>
      [--alpha 0.05] [--max-drop 0.02] [--resamples 10000] [--seed 1] [--max-flips 30]
      [--manifest FILE] [--pages-dir DIR] [--gt-dir DIR] [--out DIR]

Both runs are scored against the labels as they are now (a stored score/ is never read) and the
fields of the documents scored in BOTH are paired. Prints the verdict first, then what differs
between the runs, the headline numbers, the exact McNemar test, a paired bootstrap over documents
and the fields that flipped; writes comparison.json and comparison.md to --out (default
outputs/compare/<baseline>__vs__<candidate>/). See `evalkit/compare.py`.

A regression needs BOTH the one-sided exact McNemar p-value <= alpha AND an accuracy drop >=
max-drop (a fraction: 0.02 = two points). The bootstrap interval is reported, not gated.

Exit codes: 0 no regression, 1 REGRESSION, 2 the runs cannot be compared (a run folder that is
missing or unreadable, a label that cannot be read, no labeled document scored in both, bad
arguments). Unlike score.py, where 1 means "could not be scored", a gate needs 1 to mean
"regressed".
"""

import argparse
import sys
from pathlib import Path

from docextract.paths import DatasetPaths, find_repo_root
from evalkit.compare import CompareError, Thresholds, compare_runs
from evalkit.gt_store import GroundTruthError
from evalkit.scorecard import ScorecardError


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description="Compare two extraction runs: diff, flips, exact paired test, verdict."
    )
    parser.add_argument("baseline", type=Path, help="outputs/runs/<baseline_run_id>")
    parser.add_argument("candidate", type=Path, help="outputs/runs/<candidate_run_id>")
    parser.add_argument("--alpha", type=float, default=0.05, help="p-value level (default 0.05)")
    parser.add_argument(
        "--max-drop", type=float, default=0.02, help="accuracy drop that matters (default 0.02)"
    )
    parser.add_argument("--resamples", type=int, default=10_000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=1, help="bootstrap seed")
    parser.add_argument("--max-flips", type=int, default=30, help="flips listed per section")
    parser.add_argument("--manifest", type=Path, default=None, help="the dataset manifest")
    parser.add_argument("--pages-dir", type=Path, default=None, help="the stored pages directory")
    parser.add_argument("--gt-dir", type=Path, default=None, help="the ground-truth directory")
    parser.add_argument("--out", type=Path, default=None, help="where the report goes")
    args = parser.parse_args(argv)

    for run_dir in (args.baseline, args.candidate):
        if not run_dir.is_dir():
            print(f"{run_dir}: not a directory", file=sys.stderr)
            return 2
    try:
        thresholds = Thresholds(alpha=args.alpha, max_drop=args.max_drop)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    paths = DatasetPaths.default()
    try:
        comparison = compare_runs(
            args.baseline,
            args.candidate,
            manifest_path=args.manifest or paths.manifest,
            labels_dir=args.gt_dir or paths.ground_truth_dir,
            pages_dir=args.pages_dir or paths.pages_dir,
            thresholds=thresholds,
            seed=args.seed,
            resamples=args.resamples,
        )
    except (CompareError, ScorecardError, GroundTruthError, ValueError) as exc:
        print(f"cannot compare: {exc}", file=sys.stderr)
        return 2

    out_dir = args.out or (
        find_repo_root()
        / "outputs"
        / "compare"
        / f"{args.baseline.name}__vs__{args.candidate.name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(comparison.to_json(), encoding="utf-8", newline="\n")
    (out_dir / "comparison.md").write_text(
        comparison.to_markdown(args.max_flips), encoding="utf-8", newline="\n"
    )
    print(comparison.to_text(args.max_flips), end="")
    print(f"report: {out_dir / 'comparison.md'}")
    return 1 if comparison.decision.verdict == "regression" else 0


if __name__ == "__main__":
    sys.exit(main())
