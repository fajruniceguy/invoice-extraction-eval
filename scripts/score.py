"""Score an extraction run against ground truth, offline.

  python -m uv run python scripts/score.py outputs/runs/<run_id>

Writes <run_dir>/score/manifest.json, scorecard.json, scorecard.md and docs/<doc_id>.json (see
`evalkit/scorecard.py`). A run folder whose documents have no label yet is not an error: they are
listed as unscored, but scoring nothing at all exits 1, and so does a run folder that cannot be
scored (no run.json, a broken record, a broken or missing label) or that is not a directory.
"""

import argparse
import sys
from pathlib import Path

from docextract.paths import DatasetPaths
from evalkit.gt_store import GroundTruthError
from evalkit.perf import PerfStats
from evalkit.scorecard import Scorecard, ScorecardError, money, score_run, seconds


def format_perf(perf: PerfStats) -> str:
    """One line: what the run cost (paid, and at list price) and how long a live call took."""
    cost = (
        f"{money(perf.cost_usd)} paid ({money(perf.list_cost_usd)} at list price, "
        f"{money(perf.list_cost_per_doc)}/doc) over {perf.documents} document(s)"
    )
    if perf.live_calls:
        latency = (
            f"latency p50 {seconds(perf.latency_p50_ms)} / p95 {seconds(perf.latency_p95_ms)} "
            f"over {perf.live_calls} live call(s)"
        )
    else:
        latency = "latency n/a (no live calls: every answer came from the cache)"
    return f"  cost: {cost}; {latency}"


def format_report(card: Scorecard, out_dir: Path) -> str:
    overall = card.overall
    rows = card.rows
    lines = [
        f"run {card.manifest['run_id']}: {len(card.docs)} document(s) scored, "
        f"{len(card.unscored)} without a ground-truth label",
        f"  fields: {overall.labeled} labeled, {overall.correct} correct, "
        f"accuracy {overall.accuracy:.3f} (abstentions are errors), "
        f"selective {overall.selective_accuracy:.3f}, coverage {overall.coverage:.3f}",
        f"  rows: {rows.matched} matched, {rows.missed} missed, {rows.spurious} spurious, "
        f"F1 {rows.row_f1:.3f}, row accuracy {rows.row_accuracy:.3f}, "
        f"cell accuracy {rows.cell_accuracy:.3f}",
    ]
    lines.append(format_perf(card.perf))
    if card.errors:
        counts = ", ".join(f"{name} {count}" for name, count in card.errors.items())
        lines.append(f"  errors: {counts}")
    else:
        lines.append("  errors: none")
    if card.unscored:
        lines.append("  unscored: " + ", ".join(card.unscored))
    lines.append(f"  scorecard: {out_dir / 'scorecard.md'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an extraction run against ground truth.")
    parser.add_argument("run_dir", nargs="+", type=Path, help="outputs/runs/<run_id>")
    parser.add_argument("--manifest", type=Path, default=None, help="the dataset manifest")
    parser.add_argument("--pages-dir", type=Path, default=None, help="the stored pages directory")
    parser.add_argument("--gt-dir", type=Path, default=None, help="the ground-truth directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where the scorecard goes (default <run_dir>/score)",
    )
    args = parser.parse_args(argv)
    if args.out is not None and len(args.run_dir) > 1:
        parser.error("--out takes a single run folder")

    paths = DatasetPaths.default()
    manifest = args.manifest or paths.manifest
    pages_dir = args.pages_dir or paths.pages_dir
    labels_dir = args.gt_dir or paths.ground_truth_dir

    status = 0
    for run_dir in args.run_dir:
        if not run_dir.is_dir():
            print(f"{run_dir}: not a directory", file=sys.stderr)
            status = 2
            continue
        out_dir = args.out or run_dir / "score"
        try:
            card = score_run(
                run_dir,
                manifest_path=manifest,
                labels_dir=labels_dir,
                pages_dir=pages_dir,
            )
        except (ScorecardError, GroundTruthError) as exc:
            print(f"{run_dir}: {exc}", file=sys.stderr)
            status = 1
            continue
        card.write(out_dir)
        print(format_report(card, out_dir))
        if not card.docs:
            print(f"{run_dir}: no document has a ground-truth label yet", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
