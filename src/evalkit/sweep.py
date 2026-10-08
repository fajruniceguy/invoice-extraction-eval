"""A sweep of one setting at a time against a baseline run: accuracy against cost and latency, and
which setting to keep.

Each variant is scored against the SAME labels as the baseline and compared with it the way
`compare` does (paired fields, exact McNemar, document-level bootstrap), so a row says not only how
much accuracy moved but whether that is distinguishable from noise. The cost and latency come from
the run's own records (`evalkit.perf`).

The rule for the setting to keep is fixed here, before any result:

  among the variants with no failed or refused document, take those whose accuracy is within
  `tolerance` of the baseline (exactly, as fractions) AND whose M6 verdict against the baseline is
  "ok"; of those, the one that is strictly cheaper at list price (ties: faster at p50) than the
  baseline, cheapest first. If none qualifies, the baseline stays.

A non-significant but large drop is NOT acceptable: with a small sample, "no significant loss"
mostly means "cannot tell". The report says how few independent documents back each number.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from evalkit.compare import FEW_DOCUMENTS, Comparison, Thresholds, compare_runs
from evalkit.scorecard import Scorecard, money, score_run, seconds

DEFAULT_TOLERANCE = Fraction(2, 100)  # two points of accuracy
BASELINE = "baseline"


@dataclass(frozen=True)
class SweepPoint:
    """One run of the sweep. `setting` is what differs from the baseline (the knob's value)."""

    label: str
    run_dir: Path
    knob: str = ""  # "resolution", "effort", "combined"; empty for the baseline
    setting: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SweepRow:
    label: str
    knob: str
    setting: dict[str, Any]
    run_id: str
    documents: int  # scored documents in the run
    failed: int  # scored documents that are not ok (failed, refused, not run)
    accuracy: float
    selective_accuracy: float
    coverage: float
    row_f1: float
    cost_per_doc: Decimal | None  # paid
    list_cost_per_doc: Decimal | None  # paid + avoided by the cache
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    input_tokens: int
    output_tokens: int
    # against the baseline (the baseline row itself: delta 0, p 1, verdict "baseline")
    drop: Fraction  # baseline accuracy - this accuracy over the paired fields (exact)
    p_worse: Fraction
    interval_low: float  # bootstrap, delta = this - baseline, in accuracy units
    interval_high: float
    independent: int  # independent documents behind the comparison
    verdict: str  # "baseline" | "ok" | "regression"


@dataclass(frozen=True)
class Choice:
    label: str  # the kept setting's label (BASELINE when nothing beat it)
    why: str
    reasons: dict[str, str]  # per variant: why it was or was not eligible


def _row(
    point: SweepPoint, card: Scorecard, comparison: Comparison | None, baseline: bool = False
) -> SweepRow:
    perf = card.perf
    return SweepRow(
        label=BASELINE if baseline else point.label,
        knob=point.knob,
        setting=dict(point.setting or {}),
        run_id=str(card.manifest["run_id"]),
        documents=len(card.docs),
        failed=sum(score.status != "ok" for score in card.docs),
        accuracy=card.overall.accuracy,
        selective_accuracy=card.overall.selective_accuracy,
        coverage=card.overall.coverage,
        row_f1=card.rows.row_f1,
        cost_per_doc=perf.cost_per_doc,
        list_cost_per_doc=perf.list_cost_per_doc,
        latency_p50_ms=perf.latency_p50_ms,
        latency_p95_ms=perf.latency_p95_ms,
        input_tokens=perf.input_tokens,
        output_tokens=perf.output_tokens,
        drop=Fraction(0) if comparison is None else comparison.decision.drop,
        p_worse=Fraction(1) if comparison is None else comparison.mcnemar.p_worse,
        interval_low=0.0 if comparison is None else comparison.bootstrap.low,
        interval_high=0.0 if comparison is None else comparison.bootstrap.high,
        independent=0 if comparison is None else comparison.bootstrap.clusters,
        verdict=BASELINE if comparison is None else comparison.decision.verdict,
    )


def build_sweep(
    baseline_dir: Path,
    points: Sequence[SweepPoint],
    *,
    manifest_path: Path,
    labels_dir: Path,
    pages_dir: Path,
    thresholds: Thresholds | None = None,
    seed: int = 1,
    resamples: int = 10_000,
) -> list[SweepRow]:
    """The baseline row first, then one row per point, each compared with the baseline."""
    paths = {"manifest_path": manifest_path, "labels_dir": labels_dir, "pages_dir": pages_dir}

    def card(run_dir: Path) -> Scorecard:
        return score_run(run_dir, git_sha="unknown", git_dirty=False, **paths)

    rows = [_row(SweepPoint(BASELINE, baseline_dir), card(baseline_dir), None, baseline=True)]
    for point in points:
        comparison = compare_runs(
            baseline_dir,
            point.run_dir,
            thresholds=thresholds,
            seed=seed,
            resamples=resamples,
            **paths,
        )
        rows.append(_row(point, card(point.run_dir), comparison))
    return rows


# ---------------------------------------------------------------- the rule


def _cheaper(row: SweepRow, base: SweepRow) -> bool:
    """Strictly cheaper at list price than the baseline, or as cheap and strictly faster."""
    if row.list_cost_per_doc is None or base.list_cost_per_doc is None:
        return False
    if row.list_cost_per_doc != base.list_cost_per_doc:
        return row.list_cost_per_doc < base.list_cost_per_doc
    if row.latency_p50_ms is None or base.latency_p50_ms is None:
        return False
    return row.latency_p50_ms < base.latency_p50_ms


def _order(row: SweepRow) -> tuple[Decimal, float, str]:
    inf = float("inf")
    cost = row.list_cost_per_doc if row.list_cost_per_doc is not None else Decimal("Infinity")
    return (cost, row.latency_p50_ms if row.latency_p50_ms is not None else inf, row.label)


def choose(rows: Sequence[SweepRow], tolerance: Fraction = DEFAULT_TOLERANCE) -> Choice:
    """Apply the rule to a sweep (the baseline row first). See the module docstring."""
    base, variants = rows[0], list(rows[1:])
    reasons: dict[str, str] = {}
    eligible: list[SweepRow] = []
    for row in variants:
        if row.failed:
            reasons[row.label] = f"{row.failed} document(s) failed or were refused"
        elif row.drop > tolerance:
            reasons[row.label] = (
                f"accuracy fell {float(row.drop) * 100:.1f} points, more than the "
                f"{float(tolerance) * 100:.1f}-point tolerance"
            )
        elif row.verdict != "ok":
            reasons[row.label] = "the regression gate fired against the baseline"
        elif not _cheaper(row, base):
            reasons[row.label] = "not cheaper than the baseline"
        else:
            reasons[row.label] = "eligible"
            eligible.append(row)
    if not eligible:
        return Choice(BASELINE, "no variant qualified, so the baseline stays", reasons)
    best = min(eligible, key=_order)
    return Choice(best.label, _justification(best, base, tolerance), reasons)


def _pts(accuracy_change: float) -> str:
    """An accuracy change in points with a sign, and never a negative zero ("-0.0")."""
    return f"{accuracy_change * 100 + 0.0:+.1f}"


def _change(before: Decimal | None, after: Decimal | None) -> str:
    if before is None or after is None or before == 0:
        return "n/a"
    return f"{float((after - before) / before) * 100:+.0f}%"


def _latency_change(before: float | None, after: float | None) -> str:
    if before is None or after is None or before == 0:
        return "n/a"
    return f"{(after - before) / before * 100:+.0f}%"


def _justification(row: SweepRow, base: SweepRow, tolerance: Fraction) -> str:
    text = (
        f"{row.label}: accuracy {row.accuracy:.3f} against {base.accuracy:.3f} "
        f"({_pts(-float(row.drop))} points, within the {float(tolerance) * 100:.1f}-point "
        f"tolerance), McNemar p_worse {float(row.p_worse):.3f} (no regression), cost per document "
        f"{money(row.list_cost_per_doc)} against {money(base.list_cost_per_doc)} "
        f"({_change(base.list_cost_per_doc, row.list_cost_per_doc)}), latency p50 "
        f"{seconds(row.latency_p50_ms)} against {seconds(base.latency_p50_ms)} "
        f"({_latency_change(base.latency_p50_ms, row.latency_p50_ms)})."
    )
    if row.independent < FEW_DOCUMENTS:
        text += (
            f" Thin evidence: only {row.independent} independent documents, so a loss under about "
            "5 points could not have been detected; treat the accuracy as unchanged within noise, "
            "not as proven equal."
        )
    return text


# ---------------------------------------------------------------- reports


def _setting_text(setting: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(setting.items())) or "baseline"


def to_json(rows: Sequence[SweepRow], choice: Choice, *, tolerance: Fraction) -> str:
    import json

    payload = {
        "tolerance": float(tolerance),
        "chosen": choice.label,
        "why": choice.why,
        "reasons": choice.reasons,
        "rows": [
            {
                "label": r.label,
                "knob": r.knob,
                "setting": r.setting,
                "run_id": r.run_id,
                "documents": r.documents,
                "failed": r.failed,
                "accuracy": r.accuracy,
                "selective_accuracy": r.selective_accuracy,
                "coverage": r.coverage,
                "row_f1": r.row_f1,
                "cost_per_doc": None if r.cost_per_doc is None else str(r.cost_per_doc),
                "list_cost_per_doc": (
                    None if r.list_cost_per_doc is None else str(r.list_cost_per_doc)
                ),
                "latency_p50_ms": r.latency_p50_ms,
                "latency_p95_ms": r.latency_p95_ms,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "accuracy_delta": -float(r.drop) + 0.0,
                "p_worse": float(r.p_worse),
                "interval": [r.interval_low, r.interval_high],
                "independent_documents": r.independent,
                "verdict": r.verdict,
            }
            for r in rows
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def to_markdown(
    rows: Sequence[SweepRow], choice: Choice, *, sweep_id: str, tolerance: Fraction
) -> str:
    lines = [
        f"# Sweep `{sweep_id}`: one setting at a time against the baseline",
        "",
        f"Chosen: **{choice.label}**. {choice.why}",
        "",
        f"Rule: the cheapest variant with no failed document, accuracy within "
        f"{float(tolerance) * 100:.1f} points of the baseline and no regression verdict; otherwise "
        "the baseline stays.",
        "",
        "| point | setting | docs | failed | accuracy | delta | p (worse) | 95% interval "
        "| selective | coverage | row F1 | cost/doc (list) | p50 | p95 | tokens in/out | verdict |",
        "| " + " | ".join(["---"] * 16) + " |",
    ]
    for r in rows:
        is_base = r.label == BASELINE
        delta = "-" if is_base else f"{_pts(-float(r.drop))} pts"
        p = "-" if is_base else f"{float(r.p_worse):.3f}"
        interval = "-" if is_base else f"[{_pts(r.interval_low)}, {_pts(r.interval_high)}] pts"
        lines.append(
            f"| {r.label} | {_setting_text(r.setting)} | {r.documents} | {r.failed} "
            f"| {r.accuracy:.3f} | {delta} | {p} | {interval} | {r.selective_accuracy:.3f} "
            f"| {r.coverage:.3f} | {r.row_f1:.3f} | {money(r.list_cost_per_doc)} "
            f"| {seconds(r.latency_p50_ms)} | {seconds(r.latency_p95_ms)} "
            f"| {r.input_tokens}/{r.output_tokens} | {r.verdict} |"
        )
    lines += ["", "## Why each variant was or was not kept", ""]
    lines += [f"- {label}: {reason}" for label, reason in choice.reasons.items()]
    thin = min((r.independent for r in rows[1:]), default=0)
    if rows[1:] and thin < FEW_DOCUMENTS:
        lines += [
            "",
            f"Caveat: the comparisons rest on as few as {thin} independent documents. Fields "
            "inside a document are not independent, so the p-values are optimistic, and a "
            "difference under about 5 points cannot be resolved at this size.",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the scorecard's before/after


def before_after(
    baseline: Scorecard, candidate: Scorecard, comparison: Comparison
) -> dict[str, Any]:
    """The numbers of the "Before / after" section of a scorecard (see scorecard.py)."""

    def metric(before: Any, after: Any) -> dict[str, Any]:
        delta = None if before is None or after is None else after - before
        return {"before": before, "after": after, "delta": delta}

    def number(value: Decimal | None) -> float | None:
        return None if value is None else float(value)

    b, a = baseline.perf, candidate.perf
    return {
        "baseline": str(baseline.manifest["run_id"]),
        "candidate": str(candidate.manifest["run_id"]),
        "documents_paired": len(comparison.documents),
        "fields_paired": len(comparison.pairs),
        "metrics": {
            "accuracy": metric(comparison.baseline.accuracy, comparison.candidate.accuracy),
            "selective_accuracy": metric(
                comparison.baseline.selective_accuracy, comparison.candidate.selective_accuracy
            ),
            "coverage": metric(comparison.baseline.coverage, comparison.candidate.coverage),
            "row_f1": metric(comparison.baseline_rows.row_f1, comparison.candidate_rows.row_f1),
            "list_cost_per_doc_usd": metric(
                number(b.list_cost_per_doc), number(a.list_cost_per_doc)
            ),
            "latency_p50_ms": metric(b.latency_p50_ms, a.latency_p50_ms),
            "latency_p95_ms": metric(b.latency_p95_ms, a.latency_p95_ms),
            "input_tokens_per_doc": metric(
                b.input_tokens / max(b.documents, 1), a.input_tokens / max(a.documents, 1)
            ),
            "output_tokens_per_doc": metric(
                b.output_tokens / max(b.documents, 1), a.output_tokens / max(a.documents, 1)
            ),
        },
        "mcnemar": {
            "regressions": comparison.mcnemar.regressions,
            "improvements": comparison.mcnemar.improvements,
            "p_worse": float(comparison.mcnemar.p_worse),
        },
        "bootstrap": {
            "delta": comparison.bootstrap.delta,
            "low": comparison.bootstrap.low,
            "high": comparison.bootstrap.high,
            "documents": comparison.bootstrap.clusters,
        },
        "verdict": comparison.decision.verdict,
        "differs": [
            {"key": d.key, "baseline": d.baseline, "candidate": d.candidate}
            for d in comparison.differs
        ],
    }
