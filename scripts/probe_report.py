"""Report the hallucination probe (S4) from a run of the probe set.

Usage: python -m uv run python scripts/probe_report.py RUN_DIR [--probes-dir DIR] [--out F]
           [--notes F]

RUN_DIR is a run of the probe set, made with
  scripts/extract.py --split test --manifest data/probes/manifest.csv --pages-dir data/probes/pages
      --ground-truth-dir data/probes/labels --run-id probe-s4
Prints the markdown report, or writes it to --out; --notes appends a hand-written "Reading".
Offline; changes nothing. Exit 0 done, 2 the run or the probe set cannot be read.
"""

import argparse
import sys
from pathlib import Path

from docextract.paths import DatasetPaths
from evalkit.probe_report import ProbeReportError, analyse, to_markdown


def main(argv: list[str] | None = None) -> int:
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(description="Report the hallucination probe.")
    parser.add_argument("run_dir", type=Path, help="outputs/runs/<run_id> of the probe set")
    parser.add_argument("--probes-dir", type=Path, default=paths.data_dir / "probes")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    parser.add_argument("--notes", type=Path, default=None, help="a hand-written Reading section")
    args = parser.parse_args(argv)

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
        report = analyse(args.run_dir, args.probes_dir)
    except ProbeReportError as exc:
        print(f"{args.run_dir}: {exc}", file=sys.stderr)
        return 2
    text = to_markdown(report, notes=notes)
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"{report.run_id}: {len(report.cases)} erased field(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
