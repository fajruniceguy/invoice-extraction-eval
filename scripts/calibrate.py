"""Is the confidence meaningful, and where should the abstention threshold sit? Offline.

  python -m uv run python scripts/calibrate.py RUN [RUN ...] [--out F] [--svg-dir D] [--notes F]

Pools the labeled fields of the given runs (ideally repeats of one setting). Writes the markdown
report to --out (or prints it) and, with --svg-dir, `reliability.svg` and `risk_coverage.svg`.
--abstain-below overrides the threshold reported as "in force" (default: the configured
ABSTAIN_BELOW). Exit 0 done, 2 the runs or the labels cannot be read.
"""

import argparse
import sys
from pathlib import Path

from docextract.config import Settings
from docextract.paths import DatasetPaths
from evalkit.calibration import analyse, reliability_svg, risk_coverage_svg, to_markdown
from evalkit.failures import FailureError, collect
from evalkit.gt_store import GroundTruthError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the confidence on labeled runs.")
    parser.add_argument("run_dir", nargs="+", type=Path, help="outputs/runs/<run_id>")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    parser.add_argument(
        "--svg-dir", type=Path, default=None, help="write the two SVG diagrams here"
    )
    parser.add_argument("--notes", type=Path, default=None, help="a hand-written Reading section")
    parser.add_argument("--abstain-below", type=float, default=None, help="the threshold in force")
    parser.add_argument("--manifest", type=Path, default=None, help="the dataset manifest")
    parser.add_argument("--gt-dir", type=Path, default=None, help="the ground-truth directory")
    args = parser.parse_args(argv)

    for run_dir in args.run_dir:
        if not run_dir.is_dir():
            print(f"{run_dir}: not a directory", file=sys.stderr)
            return 2
    in_force = args.abstain_below
    if in_force is None:
        in_force = Settings().abstain_below
    notes = None
    if args.notes is not None:
        try:
            notes = args.notes.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"{args.notes}: cannot be read ({exc})", file=sys.stderr)
            return 2
    paths = DatasetPaths.default()
    try:
        runs = [
            collect(
                run_dir,
                manifest_path=args.manifest or paths.manifest,
                labels_dir=args.gt_dir or paths.ground_truth_dir,
            )
            for run_dir in args.run_dir
        ]
    except (FailureError, GroundTruthError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report = analyse(runs, abstain_below=in_force)
    text = to_markdown(report, run_ids=[run.run_id for run in runs], notes=notes)
    if args.svg_dir is not None:
        args.svg_dir.mkdir(parents=True, exist_ok=True)
        (args.svg_dir / "reliability.svg").write_text(
            reliability_svg(report), encoding="utf-8", newline="\n"
        )
        (args.svg_dir / "risk_coverage.svg").write_text(
            risk_coverage_svg(report), encoding="utf-8", newline="\n"
        )
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"{len(runs)} run(s), {report.data.labeled} field instances -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
