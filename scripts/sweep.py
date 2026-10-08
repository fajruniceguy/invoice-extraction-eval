"""Sweep one setting at a time against a baseline run and say which setting to keep.

  python -m uv run python scripts/sweep.py --baseline-run outputs/runs/dev-baseline --sweep-id m8
      [--split dev] [--resolution 1024,1568] [--effort off,low] [--confirm] [--concurrency 3]
      [--tolerance 0.02] [--dry-run] [--resume] [--env-file PATH] [--manifest FILE]
      [--pages-dir DIR] [--ground-truth-dir DIR] [--runs-dir DIR] [--sweeps-dir DIR]

COSTS MONEY (one run of the split per point; --dry-run prints the plan and calls nothing).

The baseline is an EXISTING run (the full-quality setting you already measured). Each level of each
knob is run alone from the settings in the environment (`LLM_IMAGE_MAX_LONG_EDGE` for --resolution,
`LLM_REASONING_EFFORT` for --effort), one run folder per point (outputs/runs/sweep-<id>-<point>).
With --confirm the winning level of each knob is combined and run once more. Then every point is
scored against the same labels and compared with the baseline (paired McNemar, bootstrap), and the
setting to keep is chosen by the rule in `evalkit/sweep.py`: the cheapest variant with no failed
document, accuracy within --tolerance of the baseline and no regression verdict; otherwise the
baseline stays. Output: <sweeps-dir>/<id>/sweep.md and sweep.json.

--resume skips a point whose run.json exists (an interrupted sweep continues); a run folder without
a run.json is a half-finished run and is never overwritten (delete it). The sweep refuses zero
prices (cost is one of its axes). Exit codes: 0 done, 1 a document failed in some run, 2 bad
arguments, configuration or baseline, 3 the run cost cap (MAX_RUN_COST_USD) stopped a run.
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from docextract.config import Settings
from docextract.extract.runner import (
    DocRecord,
    SelectionError,
    exit_code,
    run_extraction,
    select_documents,
)
from docextract.ingest.manifest import ManifestError, ManifestRow, read_manifest
from docextract.paths import DatasetPaths, find_repo_root
from docextract.providers.openai_compat import OpenAICompatClient
from evalkit.compare import CompareError
from evalkit.gt_store import GroundTruthError
from evalkit.scorecard import ScorecardError, money
from evalkit.sweep import (
    BASELINE,
    SweepPoint,
    SweepRow,
    build_sweep,
    choose,
    to_json,
    to_markdown,
)


@dataclass(frozen=True)
class Planned:
    label: str
    knob: str
    setting: dict[str, Any]  # what differs from the baseline, by short name


FIELD_OF = {
    "image_max_long_edge": "llm_image_max_long_edge",
    "reasoning_effort": "llm_reasoning_effort",
}


def levels(text: str, cast: type) -> list[Any]:
    """`1024,1568` -> [1024, 1568]; an empty string is no levels."""
    items = [part.strip() for part in text.split(",") if part.strip()]
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError(f"{text!r}: a level is listed twice")
    try:
        values = [cast(item) for item in items]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: cannot read every level") from None
    if cast is int and any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(f"{text!r}: sizes are whole numbers of pixels, 1 or more")
    return values


def resolution_levels(text: str) -> list[int]:
    return levels(text, int)


def effort_levels(text: str) -> list[str]:
    return levels(text, str)


def plan(resolutions: list[int], efforts: list[str]) -> list[Planned]:
    return [
        *(Planned(f"res-{n}", "resolution", {"image_max_long_edge": n}) for n in resolutions),
        *(Planned(f"effort-{e}", "effort", {"reasoning_effort": e}) for e in efforts),
    ]


def settings_for(base: Settings, setting: dict[str, Any]) -> Settings:
    return base.model_copy(update={FIELD_OF[key]: value for key, value in setting.items()})


def run_dir_of(runs_dir: Path, sweep_id: str, label: str) -> Path:
    return runs_dir / f"sweep-{sweep_id}-{label}"


def print_record(record: DocRecord) -> None:
    print(f"      {record.status:<8} {record.doc_id}", flush=True)


async def run_point(
    point: Planned,
    *,
    base: Settings,
    selected: list[ManifestRow],
    rows: list[ManifestRow],
    args: argparse.Namespace,
    run_dir: Path,
) -> int:
    """Run one point; returns the runner's exit code (0 ok, 1 a failure, 3 the cost cap)."""
    settings = settings_for(base, point.setting)
    async with OpenAICompatClient(settings) as client:
        record = await run_extraction(
            selected,
            rows=rows,
            client=client,
            settings=settings,
            pages_dir=args.pages_dir,
            labels_dir=args.ground_truth_dir,
            run_dir=run_dir,
            selection={"split": args.split, "sweep": args.sweep_id, "point": point.label},
            on_record=print_record,
            concurrency=args.concurrency,
        )
    totals = record.totals
    print(
        f"    {point.label}: {totals.ok} ok, {totals.failed} failed, {totals.refused} refused, "
        f"{totals.not_run} not run; ${totals.cost_usd:.4f} paid",
        flush=True,
    )
    return exit_code(record)


def describe(setting: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(setting.items()))


def winners(rows: list[SweepRow], tolerance: Fraction) -> dict[str, dict[str, Any]]:
    """The winning level of each knob, judged alone against the baseline."""
    result: dict[str, dict[str, Any]] = {}
    for knob in ("resolution", "effort"):
        of_knob = [row for row in rows[1:] if row.knob == knob]
        if not of_knob:
            continue
        choice = choose([rows[0], *of_knob], tolerance)
        if choice.label != BASELINE:
            result[knob] = next(row.setting for row in of_knob if row.label == choice.label)
    return result


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    paths = DatasetPaths.default()
    root = find_repo_root()
    parser = argparse.ArgumentParser(description="Sweep one setting at a time against a baseline.")
    parser.add_argument("--baseline-run", type=Path, required=True, help="outputs/runs/<run_id>")
    parser.add_argument(
        "--sweep-id", required=True, help="names the folders (letters, digits, _ -)"
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--resolution", type=resolution_levels, default=[1024, 1568])
    parser.add_argument("--effort", type=effort_levels, default=["off", "low"])
    parser.add_argument("--confirm", action="store_true", help="also run the winning combination")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, call nothing")
    parser.add_argument("--resume", action="store_true", help="skip points that already have a run")
    parser.add_argument("--env-file", type=Path, default=root / ".env")
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--pages-dir", type=Path, default=paths.pages_dir)
    parser.add_argument("--ground-truth-dir", type=Path, default=paths.ground_truth_dir)
    parser.add_argument("--runs-dir", type=Path, default=root / "outputs" / "runs")
    parser.add_argument("--sweeps-dir", type=Path, default=root / "outputs" / "sweeps")
    args = parser.parse_args()

    if not args.sweep_id.replace("-", "").replace("_", "").isalnum():
        print(f"--sweep-id {args.sweep_id!r}: letters, digits, '_' and '-' only", file=sys.stderr)
        return 2
    if args.concurrency < 1 or not 0 <= args.tolerance <= 1:
        print("--concurrency must be 1 or more and --tolerance between 0 and 1", file=sys.stderr)
        return 2
    if not (args.baseline_run / "run.json").is_file():
        print(
            f"{args.baseline_run}: no run.json (the baseline must be a finished run)",
            file=sys.stderr,
        )
        return 2
    planned = plan(args.resolution, args.effort)
    if not planned:
        print("nothing to sweep: give --resolution and/or --effort levels", file=sys.stderr)
        return 2
    try:
        settings = Settings(_env_file=args.env_file)
    except ValidationError as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2
    if settings.llm_price_input_per_mtok == 0 and settings.llm_price_output_per_mtok == 0:
        print(
            "LLM_PRICE_INPUT_PER_MTOK and LLM_PRICE_OUTPUT_PER_MTOK are both 0: a sweep compares "
            "cost, so set the model's real prices first",
            file=sys.stderr,
        )
        return 2
    try:
        manifest = read_manifest(args.manifest)
        selected = select_documents(manifest, split=args.split)
    except (ManifestError, OSError, SelectionError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    tolerance = Fraction(str(args.tolerance))
    calls = (len(planned) + (1 if args.confirm else 0)) * len(selected)
    print(
        f"sweep {args.sweep_id}: baseline {args.baseline_run.name} on the {args.split} split "
        f"({len(selected)} documents), model {settings.llm_model}, concurrency {args.concurrency}"
    )
    for point in planned:
        print(f"  {point.label:<14} {describe(point.setting)}")
    if args.confirm:
        print("  combined       (the winning level of each knob, decided after the others)")
    print(
        f"  planned requests: {calls} ({len(selected)} documents x {calls // len(selected)} runs)"
    )
    if args.dry_run:
        print("dry run: nothing was called")
        return 0

    status = 0
    done: list[SweepPoint] = []

    def execute(point: Planned) -> int | None:
        """0/1/3 from the run, None when skipped by --resume, 2 for a half-finished folder."""
        folder = run_dir_of(args.runs_dir, args.sweep_id, point.label)
        if (folder / "run.json").is_file():
            if args.resume:
                print(f"  {point.label}: already done, skipped (--resume)")
                done.append(SweepPoint(point.label, folder, point.knob, point.setting))
                return None
            print(
                f"{folder} already exists: use --resume or choose another --sweep-id",
                file=sys.stderr,
            )
            return 2
        if folder.exists():
            print(f"{folder} holds a half-finished run: delete it first", file=sys.stderr)
            return 2
        print(f"  {point.label}: {describe(point.setting)}", flush=True)
        code = asyncio.run(
            run_point(
                point, base=settings, selected=selected, rows=manifest, args=args, run_dir=folder
            )
        )
        done.append(SweepPoint(point.label, folder, point.knob, point.setting))
        return code

    def current_rows() -> list[SweepRow]:
        return build_sweep(
            args.baseline_run,
            done,
            manifest_path=args.manifest,
            labels_dir=args.ground_truth_dir,
            pages_dir=args.pages_dir,
        )

    try:
        for point in planned:
            code = execute(point)
            if code in (2, 3):
                return code
            status = max(status, code or 0)
        if args.confirm:
            wins = winners(current_rows(), tolerance)
            if wins:
                combined = Planned(
                    "combined", "combined", {k: v for w in wins.values() for k, v in w.items()}
                )
                print(f"  winners per knob: { {k: describe(v) for k, v in wins.items()} }")
                code = execute(combined)
                if code in (2, 3):
                    return code
                status = max(status, code or 0)
            else:
                print("  no knob has a winner on its own: nothing to confirm")
        rows = current_rows()
    except (CompareError, ScorecardError, GroundTruthError) as exc:
        print(f"cannot score the sweep: {exc}", file=sys.stderr)
        return 2

    choice = choose(rows, tolerance)
    out = args.sweeps_dir / args.sweep_id
    out.mkdir(parents=True, exist_ok=True)
    markdown = to_markdown(rows, choice, sweep_id=args.sweep_id, tolerance=tolerance)
    (out / "sweep.md").write_text(markdown, encoding="utf-8", newline="\n")
    (out / "sweep.json").write_text(
        to_json(rows, choice, tolerance=tolerance), encoding="utf-8", newline="\n"
    )
    print()
    print(markdown, end="")
    base_cost = rows[0].list_cost_per_doc
    print(f"baseline cost per document {money(base_cost)}; report: {out / 'sweep.md'}")
    if any(row.failed for row in rows):
        status = max(status, 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
