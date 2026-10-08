"""Run-to-run noise: how much runs of the SAME setting differ from each other. Offline.

  python -m uv run python scripts/noise_report.py RUN RUN [RUN ...] [--out F] [--notes F]

Give it runs made with identical settings (fresh samples: `extract.py --cache refresh`). Dollars
are recomputed from tokens at --price-in/--price-out (USD per million; default: the configured
LLM_PRICE_* in the environment/.env), so runs billed at different prices still compare. Exit 0
done, 2 the runs cannot be read or are not repeats of one another.
"""

import argparse
import sys
from decimal import Decimal
from pathlib import Path

from docextract.config import Settings
from docextract.paths import DatasetPaths
from evalkit.failures import FailureError
from evalkit.gt_store import GroundTruthError
from evalkit.noise import NoiseError, measure, to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the noise between repeated runs.")
    parser.add_argument("run_dir", nargs="+", type=Path, help="outputs/runs/<run_id>, two or more")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    parser.add_argument("--notes", type=Path, default=None, help="a hand-written Reading section")
    parser.add_argument("--price-in", type=Decimal, default=None, help="USD per 1M input tokens")
    parser.add_argument("--price-out", type=Decimal, default=None, help="USD per 1M output tokens")
    parser.add_argument("--manifest", type=Path, default=None, help="the dataset manifest")
    parser.add_argument("--gt-dir", type=Path, default=None, help="the ground-truth directory")
    args = parser.parse_args(argv)

    for run_dir in args.run_dir:
        if not run_dir.is_dir():
            print(f"{run_dir}: not a directory", file=sys.stderr)
            return 2
    price_in, price_out = args.price_in, args.price_out
    if price_in is None or price_out is None:
        settings = Settings()
        price_in = settings.llm_price_input_per_mtok if price_in is None else price_in
        price_out = settings.llm_price_output_per_mtok if price_out is None else price_out
    notes = None
    if args.notes is not None:
        try:
            notes = args.notes.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"{args.notes}: cannot be read ({exc})", file=sys.stderr)
            return 2
    paths = DatasetPaths.default()
    try:
        report = measure(
            args.run_dir,
            manifest_path=args.manifest or paths.manifest,
            labels_dir=args.gt_dir or paths.ground_truth_dir,
        )
    except (NoiseError, FailureError, GroundTruthError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    text = to_markdown(report, price_in=price_in, price_out=price_out, notes=notes)
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"{len(report.runs)} runs, {report.labeled} fields -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
