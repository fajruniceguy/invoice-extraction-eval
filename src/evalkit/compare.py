"""Side-by-side comparison of two evaluation runs: did the candidate get worse than the baseline?

`compare_runs` scores both runs against the labels as they are now, pairs every labeled scalar field
that both runs answered for, and reports what flipped between them (correct -> not correct is a
regression, the reverse an improvement), an exact McNemar test on those flips and a paired bootstrap
interval over documents, and a verdict. The statistics are hand-written (no scipy) and exact.

  * McNemar looks only at the discordant pairs: with b regressions and c improvements, and no real
    difference, each flip is equally likely to go either way, so b ~ Binomial(b + c, 1/2).
    `p_worse` = P(X >= b) is the evidence that the candidate is worse.
  * The bootstrap resamples whole documents (grouped by label owner, so a variant resamples with the
    document it borrows its label from), because fields inside one document are not independent and
    McNemar alone is optimistic about that.
"""

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from docextract.extract.runner import DocRecord
from docextract.extract.schema import SCALAR_FIELD_NAMES
from docextract.ingest.manifest import read_manifest
from evalkit.metrics import DocumentScore, FieldMetrics, FieldScore, RowsScore
from evalkit.scorecard import (
    NO_VALUE,
    SLICE_COLUMNS,
    Scorecard,
    field_dict,
    metrics_dict,
    score_run,
)
from evalkit.taxonomy import ERROR_CLASSES, count_errors

BOOTSTRAP_LEVEL = 0.95
# Below this many independent documents the p-value and the interval are only indicative.
FEW_DOCUMENTS = 10


class CompareError(Exception):
    """Two runs cannot be compared (and why)."""


# ---------------------------------------------------------------- McNemar


@dataclass(frozen=True)
class McNemar:
    """The exact test on the discordant pairs. The p-values are exact fractions."""

    regressions: int  # b: the baseline was right, the candidate is not
    improvements: int  # c: the baseline was not right, the candidate is
    p_worse: Fraction  # P(X >= b): evidence that the candidate is worse (one-sided)
    p_better: Fraction  # P(X >= c): evidence that the candidate is better (one-sided)
    p_two_sided: Fraction


def _upper(n: int, k: int) -> Fraction:
    """P(X >= k) for X ~ Binomial(n, 1/2), exactly."""
    return Fraction(sum(comb(n, i) for i in range(k, n + 1)), 2**n)


def mcnemar_exact(regressions: int, improvements: int) -> McNemar:
    """The exact McNemar test from the two discordant counts (no flips at all: every p is 1)."""
    if regressions < 0 or improvements < 0:
        raise ValueError("the counts of flips cannot be negative")
    n = regressions + improvements
    if n == 0:
        one = Fraction(1)
        return McNemar(0, 0, one, one, one)
    smaller = min(regressions, improvements)
    lower_tail = Fraction(sum(comb(n, i) for i in range(smaller + 1)), 2**n)  # P(X <= smaller)
    return McNemar(
        regressions=regressions,
        improvements=improvements,
        p_worse=_upper(n, regressions),
        p_better=_upper(n, improvements),
        p_two_sided=min(Fraction(1), 2 * lower_tail),
    )


# ---------------------------------------------------------------- paired bootstrap


@dataclass(frozen=True)
class Cluster:
    """One independent unit (a label owner): its paired fields, and how many each run got right."""

    pairs: int
    baseline_correct: int
    candidate_correct: int


@dataclass(frozen=True)
class Bootstrap:
    """Percentile interval of delta = candidate - baseline accuracy, resampling clusters."""

    delta: float  # on the observed data
    low: float
    high: float
    level: float
    resamples: int
    seed: int
    clusters: int


def paired_bootstrap(
    clusters: Sequence[Cluster],
    *,
    seed: int,
    resamples: int = 10_000,
    level: float = BOOTSTRAP_LEVEL,
) -> Bootstrap:
    """Resample `clusters` with replacement `resamples` times (seeded: the same inputs always give
    the same interval) and take the central `level` of the accuracy differences."""
    if not clusters:
        raise ValueError("no clusters to resample")
    if resamples < 1 or not 0 < level < 1:
        raise ValueError("resamples must be positive and level between 0 and 1")
    pairs = sum(c.pairs for c in clusters)
    observed = (
        sum(c.candidate_correct for c in clusters) - sum(c.baseline_correct for c in clusters)
    ) / pairs
    rng = random.Random(seed)
    count = len(clusters)
    deltas: list[float] = []
    for _ in range(resamples):
        total = baseline = candidate = 0
        for _ in range(count):
            cluster = clusters[rng.randrange(count)]
            total += cluster.pairs
            baseline += cluster.baseline_correct
            candidate += cluster.candidate_correct
        deltas.append((candidate - baseline) / total)
    deltas.sort()
    cut = int(resamples * (1 - level) / 2 + 1e-9)
    return Bootstrap(
        delta=observed,
        low=deltas[cut],
        high=deltas[resamples - 1 - cut],
        level=level,
        resamples=resamples,
        seed=seed,
        clusters=count,
    )


# ---------------------------------------------------------------- pairing and flips

Flip = Literal["regression", "improvement", "changed_wrong", "unchanged"]


@dataclass(frozen=True)
class FieldPair:
    """One labeled field of one document, as both runs answered it."""

    doc_id: str
    owner: str
    name: str
    baseline: FieldScore
    candidate: FieldScore

    @property
    def flip(self) -> Flip:
        before, after = self.baseline, self.candidate
        if before.correct and not after.correct:
            return "regression"
        if after.correct and not before.correct:
            return "improvement"
        if before.correct:
            return "unchanged"
        same = (before.outcome, before.error_class, before.predicted_status, before.predicted) == (
            after.outcome,
            after.error_class,
            after.predicted_status,
            after.predicted,
        )
        return "unchanged" if same else "changed_wrong"


def pair_fields(
    baseline: Scorecard, candidate: Scorecard
) -> tuple[list[FieldPair], list[str], list[str]]:
    """Pair the labeled fields of the documents scored in BOTH runs, in document and field order.
    Also returns the documents scored only in the baseline and only in the candidate."""
    left = {score.doc_id: score for score in baseline.docs}
    right = {score.doc_id: score for score in candidate.docs}
    pairs: list[FieldPair] = []
    for doc_id in sorted(left.keys() & right.keys()):
        theirs = {field.name: field for field in right[doc_id].fields}
        if {field.name for field in left[doc_id].fields} != theirs.keys():
            raise CompareError(f"{doc_id}: the two runs scored different labeled fields")
        pairs.extend(
            FieldPair(doc_id, left[doc_id].owner, field.name, field, theirs[field.name])
            for field in left[doc_id].fields
        )
    return pairs, sorted(left.keys() - right.keys()), sorted(right.keys() - left.keys())


# ---------------------------------------------------------------- the verdict

Verdict = Literal["regression", "ok"]


@dataclass(frozen=True)
class Thresholds:
    """The gate: a regression needs BOTH statistical evidence (the one-sided exact McNemar p-value
    at or below `alpha`) and a material drop (baseline accuracy minus candidate accuracy at or
    above `max_drop`, a fraction: 0.02 is two points)."""

    alpha: float = 0.05
    max_drop: float = 0.02

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1 (exclusive), got {self.alpha}")
        if not 0 <= self.max_drop <= 1:
            raise ValueError(f"max_drop must be between 0 and 1, got {self.max_drop}")


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    significant: bool  # p_worse <= alpha
    material: bool  # drop >= max_drop
    drop: Fraction  # baseline accuracy - candidate accuracy over the paired fields (exact)


def decide(
    test: McNemar,
    *,
    baseline_correct: int,
    candidate_correct: int,
    pairs: int,
    thresholds: Thresholds,
) -> Decision:
    """Both comparisons are exact (Fractions): the boundary is the stated `<=` / `>=`, never float
    noise. `Fraction(str(x))` reads 0.05 as 1/20, as a person means it."""
    drop = Fraction(baseline_correct - candidate_correct, pairs)
    significant = test.p_worse <= Fraction(str(thresholds.alpha))
    material = drop >= Fraction(str(thresholds.max_drop))
    return Decision("regression" if significant and material else "ok", significant, material, drop)


# ---------------------------------------------------------------- the comparison


@dataclass(frozen=True)
class Difference:
    """A setting that is not the same in the two runs."""

    key: str
    baseline: Any
    candidate: Any


@dataclass(frozen=True)
class FieldDelta:
    """How one field name fared over every document that has it."""

    name: str
    pairs: int
    baseline_correct: int
    candidate_correct: int
    regressions: int
    improvements: int


@dataclass(frozen=True)
class SliceDelta:
    column: str
    value: str
    baseline: FieldMetrics
    candidate: FieldMetrics


@dataclass(frozen=True)
class Comparison:
    """Everything `compare_runs` found. Metrics are over the documents scored in both runs."""

    baseline_id: str
    candidate_id: str
    differs: list[Difference]
    documents: list[str]  # scored in both
    only_in_baseline: list[str]
    only_in_candidate: list[str]
    status_changes: list[tuple[str, str, str]]  # (doc_id, baseline status, candidate status)
    pages_differ: list[str]  # the stored pages sent were not the same
    pairs: list[FieldPair]
    both_correct: int
    both_wrong: int
    mcnemar: McNemar
    bootstrap: Bootstrap
    baseline: FieldMetrics
    candidate: FieldMetrics
    baseline_rows: RowsScore
    candidate_rows: RowsScore
    baseline_errors: dict[str, int]
    candidate_errors: dict[str, int]
    per_field: list[FieldDelta]
    slices: list[SliceDelta]
    notes: list[str]
    thresholds: Thresholds
    decision: Decision

    def flips(self, kind: Flip) -> list[FieldPair]:
        return [pair for pair in self.pairs if pair.flip == kind]

    @property
    def verdict_line(self) -> str:
        decision, test, rule = self.decision, self.mcnemar, self.thresholds
        name = "REGRESSION" if decision.verdict == "regression" else "OK"
        return (
            f"VERDICT: {name} (p_worse {float(test.p_worse):.4f} "
            f"{'<=' if decision.significant else '>'} alpha {rule.alpha:g}; "
            f"accuracy drop {float(decision.drop) * 100:.1f} pts "
            f"{'>=' if decision.material else '<'} {rule.max_drop * 100:.1f})"
        )

    def to_json(self) -> str:
        """Sorted keys and no timestamps: the same inputs always give the same bytes."""
        payload = _Report(self).payload()
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def to_markdown(self, max_flips: int = 30) -> str:
        return _Report(self).markdown(max_flips)

    def to_text(self, max_flips: int = 30) -> str:
        """What the CLI prints: the verdict line first."""
        return _Report(self).text(max_flips)


def differences(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[Difference]:
    """What differs between two run manifests: model, endpoint, prompt and schema hash, settings."""
    keys = ["model", "endpoint", "prompt_sha256", "schema_sha256"]
    old, new = baseline["settings"], candidate["settings"]
    keys.extend(f"settings.{name}" for name in sorted(old.keys() | new.keys()))

    def value(manifest: dict[str, Any], key: str) -> Any:
        if key.startswith("settings."):
            return manifest["settings"].get(key.removeprefix("settings."))
        return manifest[key]

    return [
        Difference(key, value(baseline, key), value(candidate, key))
        for key in keys
        if value(baseline, key) != value(candidate, key)
    ]


def _page_hashes(run_dir: Path, doc_id: str) -> list[str]:
    path = run_dir / "docs" / f"{doc_id}.json"
    try:
        record = DocRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise CompareError(f"{path}: cannot be read as a document record ({exc})") from exc
    return [page.sha256 for page in record.pages]


def _errors(docs: Sequence[DocumentScore]) -> dict[str, int]:
    return count_errors(
        [
            *(field.error_class for score in docs for field in score.fields),
            *(error for score in docs for error in score.rows.errors),
        ]
    )


def error_mix(baseline: dict[str, int], candidate: dict[str, int]) -> list[tuple[str, int, int]]:
    """(error class, baseline count, candidate count) for every class either run has."""
    return [
        (name, baseline.get(name, 0), candidate.get(name, 0))
        for name in ERROR_CLASSES
        if baseline.get(name, 0) or candidate.get(name, 0)
    ]


def _field_deltas(pairs: Sequence[FieldPair]) -> list[FieldDelta]:
    grouped: dict[str, list[FieldPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.name].append(pair)
    return [
        FieldDelta(
            name=name,
            pairs=len(grouped[name]),
            baseline_correct=sum(p.baseline.correct for p in grouped[name]),
            candidate_correct=sum(p.candidate.correct for p in grouped[name]),
            regressions=sum(p.flip == "regression" for p in grouped[name]),
            improvements=sum(p.flip == "improvement" for p in grouped[name]),
        )
        for name in SCALAR_FIELD_NAMES
        if name in grouped
    ]


def _slice_deltas(pairs: Sequence[FieldPair], manifest_path: Path) -> list[SliceDelta]:
    try:
        by_id = {row.doc_id: row for row in read_manifest(manifest_path)}
    except (OSError, ValueError) as exc:
        raise CompareError(f"{manifest_path}: cannot be read as a manifest ({exc})") from exc
    grouped: dict[tuple[str, str], list[FieldPair]] = defaultdict(list)
    for pair in pairs:
        for column in SLICE_COLUMNS:
            grouped[(column, getattr(by_id[pair.doc_id], column) or NO_VALUE)].append(pair)
    return [
        SliceDelta(
            column,
            value,
            FieldMetrics.from_outcomes(p.baseline.outcome for p in members),
            FieldMetrics.from_outcomes(p.candidate.outcome for p in members),
        )
        for (column, value), members in sorted(grouped.items())
    ]


def compare_runs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    manifest_path: Path,
    labels_dir: Path,
    pages_dir: Path,
    thresholds: Thresholds | None = None,
    seed: int = 1,
    resamples: int = 10_000,
) -> Comparison:
    """Score both runs against the labels as they are now and compare them. Never reads a stored
    `score/`, so a label edited between two scorings cannot pass for a regression."""
    cards = [
        score_run(
            run_dir,
            manifest_path=manifest_path,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            git_sha="unknown",
            git_dirty=False,
        )
        for run_dir in (baseline_dir, candidate_dir)
    ]
    baseline, candidate = cards
    pairs, only_baseline, only_candidate = pair_fields(baseline, candidate)
    if not pairs:
        raise CompareError("no labeled field was scored in both runs: nothing to compare")

    left = {score.doc_id: score for score in baseline.docs}
    right = {score.doc_id: score for score in candidate.docs}
    documents = sorted(left.keys() & right.keys())
    status_changes = [
        (doc_id, left[doc_id].status, right[doc_id].status)
        for doc_id in documents
        if left[doc_id].status != right[doc_id].status
    ]
    pages_differ = []
    for doc_id in documents:
        before, after = _page_hashes(baseline_dir, doc_id), _page_hashes(candidate_dir, doc_id)
        if before and after and before != after:
            pages_differ.append(doc_id)

    regressions = sum(pair.flip == "regression" for pair in pairs)
    improvements = sum(pair.flip == "improvement" for pair in pairs)
    both_correct = sum(pair.baseline.correct and pair.candidate.correct for pair in pairs)
    both_wrong = len(pairs) - both_correct - regressions - improvements

    clusters: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for pair in pairs:
        counts = clusters[pair.owner]
        counts[0] += 1
        counts[1] += pair.baseline.correct
        counts[2] += pair.candidate.correct
    bootstrap = paired_bootstrap(
        [Cluster(*counts) for _, counts in sorted(clusters.items())], seed=seed, resamples=resamples
    )

    paired_left = [left[doc_id] for doc_id in documents]
    paired_right = [right[doc_id] for doc_id in documents]
    notes = []
    if only_baseline:
        notes.append(f"scored only in the baseline (left out): {', '.join(only_baseline)}")
    if only_candidate:
        notes.append(f"scored only in the candidate (left out): {', '.join(only_candidate)}")
    for doc_id, before, after in status_changes:
        notes.append(f"{doc_id}: the run status changed from {before} to {after}")
    if pages_differ:
        notes.append(
            "different stored pages were sent in the two runs, so flips on these documents may "
            f"come from the data and not from the change under test: {', '.join(pages_differ)}"
        )
    if bootstrap.clusters < FEW_DOCUMENTS:
        notes.append(
            f"only {bootstrap.clusters} independent document(s) (fewer than {FEW_DOCUMENTS}): "
            "fields inside a document are not independent, so the McNemar p-value is optimistic "
            "and the bootstrap interval is only indicative"
        )
    thresholds = thresholds or Thresholds()
    test = mcnemar_exact(regressions, improvements)
    return Comparison(
        baseline_id=str(baseline.manifest["run_id"]),
        candidate_id=str(candidate.manifest["run_id"]),
        differs=differences(baseline.manifest, candidate.manifest),
        documents=documents,
        only_in_baseline=only_baseline,
        only_in_candidate=only_candidate,
        status_changes=status_changes,
        pages_differ=pages_differ,
        pairs=pairs,
        both_correct=both_correct,
        both_wrong=both_wrong,
        mcnemar=test,
        bootstrap=bootstrap,
        baseline=FieldMetrics.from_outcomes(pair.baseline.outcome for pair in pairs),
        candidate=FieldMetrics.from_outcomes(pair.candidate.outcome for pair in pairs),
        baseline_rows=RowsScore.merge(score.rows for score in paired_left),
        candidate_rows=RowsScore.merge(score.rows for score in paired_right),
        baseline_errors=_errors(paired_left),
        candidate_errors=_errors(paired_right),
        per_field=_field_deltas(pairs),
        slices=_slice_deltas(pairs, manifest_path),
        notes=notes,
        thresholds=thresholds,
        decision=decide(
            test,
            baseline_correct=sum(pair.baseline.correct for pair in pairs),
            candidate_correct=sum(pair.candidate.correct for pair in pairs),
            pairs=len(pairs),
            thresholds=thresholds,
        ),
    )


# ---------------------------------------------------------------- reports

HEADLINE_METRICS: tuple[str, ...] = (
    "accuracy",
    "selective_accuracy",
    "coverage",
    "abstention_rate",
    "failure_rate",
)
ROW_METRICS: tuple[str, ...] = (
    "row_precision",
    "row_recall",
    "row_f1",
    "row_accuracy",
    "cell_accuracy",
)
VALUE_WIDTH = 40


def _rows_values(rows: RowsScore) -> dict[str, float]:
    return {name: getattr(rows, name) for name in ROW_METRICS}


def _pair_side(score: FieldScore) -> dict[str, Any]:
    shown = field_dict(score)
    return {
        "predicted": shown["predicted"],
        "status": shown["predicted_status"],
        "outcome": shown["outcome"],
        "error_class": shown["error_class"],
    }


def _flip_entry(pair: FieldPair) -> dict[str, Any]:
    return {
        "doc_id": pair.doc_id,
        "field": pair.name,
        "flip": pair.flip,
        "expected": field_dict(pair.baseline)["expected"],
        "baseline": _pair_side(pair.baseline),
        "candidate": _pair_side(pair.candidate),
    }


def _delta(before: float, after: float) -> dict[str, float]:
    return {"baseline": before, "candidate": after, "delta": after - before}


def _cell(value: Any) -> str:
    text = "(none)" if value is None else str(value)
    text = text.replace("|", "/").replace("\n", " ")
    return text if len(text) <= VALUE_WIDTH else text[: VALUE_WIDTH - 1] + "…"


def _side(side: dict[str, Any]) -> str:
    """How one run answered a field, for a table or a line."""
    status = side["status"]
    if status is None:
        return "(no prediction)"
    if status in ("abstained", "absent"):
        return f"({status})"
    return _cell(side["predicted"])


def _points(value: float) -> str:
    return f"{value * 100:+.1f} pts"


class _Report:
    """The comparison as data, text and markdown."""

    def __init__(self, comparison: Comparison) -> None:
        self.c = comparison

    # ------------------------------------------------------------ data

    def payload(self) -> dict[str, Any]:
        c = self.c
        decision, test, boot = c.decision, c.mcnemar, c.bootstrap
        headline = {
            name: _delta(metrics_dict(c.baseline)[name], metrics_dict(c.candidate)[name])
            for name in HEADLINE_METRICS
        }
        rows = {
            name: _delta(_rows_values(c.baseline_rows)[name], _rows_values(c.candidate_rows)[name])
            for name in ROW_METRICS
        }
        for name in ("matched", "missed", "spurious", "perfect"):
            rows[name] = _delta(getattr(c.baseline_rows, name), getattr(c.candidate_rows, name))
        return {
            "baseline": c.baseline_id,
            "candidate": c.candidate_id,
            "verdict": decision.verdict,
            "verdict_line": c.verdict_line,
            "thresholds": {"alpha": c.thresholds.alpha, "max_drop": c.thresholds.max_drop},
            "decision": {
                "significant": decision.significant,
                "material": decision.material,
                "drop": float(decision.drop),
            },
            "differs": [
                {"key": d.key, "baseline": d.baseline, "candidate": d.candidate} for d in c.differs
            ],
            "documents": {
                "paired": c.documents,
                "only_in_baseline": c.only_in_baseline,
                "only_in_candidate": c.only_in_candidate,
                "status_changes": [list(change) for change in c.status_changes],
                "pages_differ": c.pages_differ,
                "independent": boot.clusters,
            },
            "fields": {
                "paired": len(c.pairs),
                "both_correct": c.both_correct,
                "regressions": test.regressions,
                "improvements": test.improvements,
                "both_wrong": c.both_wrong,
            },
            "mcnemar": {
                "regressions": test.regressions,
                "improvements": test.improvements,
                "p_worse": float(test.p_worse),
                "p_better": float(test.p_better),
                "p_two_sided": float(test.p_two_sided),
            },
            "bootstrap": {
                "delta": boot.delta,
                "low": boot.low,
                "high": boot.high,
                "level": boot.level,
                "resamples": boot.resamples,
                "seed": boot.seed,
                "clusters": boot.clusters,
            },
            "headline": headline,
            "rows": rows,
            "errors": [
                {"class": name, **_delta(before, after)}
                for name, before, after in error_mix(c.baseline_errors, c.candidate_errors)
            ],
            "per_field": [
                {
                    "field": d.name,
                    "pairs": d.pairs,
                    "baseline_correct": d.baseline_correct,
                    "candidate_correct": d.candidate_correct,
                    "regressions": d.regressions,
                    "improvements": d.improvements,
                }
                for d in c.per_field
            ],
            "slices": [
                {
                    "column": s.column,
                    "value": s.value,
                    "labeled": s.baseline.labeled,
                    "accuracy": _delta(s.baseline.accuracy, s.candidate.accuracy),
                }
                for s in c.slices
            ],
            "flips": {
                kind: [_flip_entry(pair) for pair in c.flips(kind)]
                for kind in ("regression", "improvement", "changed_wrong")
            },
            "notes": c.notes,
        }

    # ------------------------------------------------------------ text

    def _flip_lines(self, kind: Flip, limit: int) -> list[str]:
        flips = self.c.flips(kind)
        lines = []
        for pair in flips[:limit]:
            entry = _flip_entry(pair)
            lines.append(
                f"  {pair.doc_id} {pair.name}: expected {_cell(entry['expected'])}; "
                f"baseline {_side(entry['baseline'])} -> candidate {_side(entry['candidate'])}"
                f" ({entry['candidate']['error_class'] or 'correct'})"
            )
        if len(flips) > limit:
            lines.append(f"  ... and {len(flips) - limit} more in comparison.json")
        return lines

    def text(self, max_flips: int) -> str:
        c = self.c
        test, boot = c.mcnemar, c.bootstrap
        lines = [
            c.verdict_line,
            f"compare: baseline {c.baseline_id}  vs  candidate {c.candidate_id}",
        ]
        if c.differs:
            lines.append("differs:")
            lines.extend(
                f"  {d.key}: {_cell(d.baseline)} -> {_cell(d.candidate)}" for d in c.differs
            )
        else:
            lines.append("differs: nothing (same model, prompt, schema and settings)")
        lines.append(
            f"paired: {len(c.documents)} document(s), {len(c.pairs)} labeled field(s), "
            f"{boot.clusters} independent"
        )
        lines.append(
            f"accuracy {c.baseline.accuracy:.3f} -> {c.candidate.accuracy:.3f}, "
            f"selective {c.baseline.selective_accuracy:.3f} -> "
            f"{c.candidate.selective_accuracy:.3f}, "
            f"coverage {c.baseline.coverage:.3f} -> {c.candidate.coverage:.3f}"
        )
        lines.append(
            f"rows: F1 {c.baseline_rows.row_f1:.3f} -> {c.candidate_rows.row_f1:.3f}, "
            f"cell accuracy {c.baseline_rows.cell_accuracy:.3f} -> "
            f"{c.candidate_rows.cell_accuracy:.3f}"
        )
        lines.append(
            f"McNemar exact: {test.regressions} regression(s), {test.improvements} "
            f"improvement(s); p_worse {float(test.p_worse):.4f}, "
            f"p_better {float(test.p_better):.4f}, two-sided {float(test.p_two_sided):.4f}"
        )
        lines.append(
            f"bootstrap ({boot.clusters} document(s), seed {boot.seed}, {boot.resamples} "
            f"resamples): delta {_points(boot.delta)}, {boot.level:.0%} interval "
            f"[{_points(boot.low)}, {_points(boot.high)}]"
        )
        lines.extend(f"note: {note}" for note in c.notes)
        errors = error_mix(c.baseline_errors, c.candidate_errors)
        if errors:
            lines.append("errors: " + ", ".join(f"{n} {a} -> {b}" for n, a, b in errors))
        for kind, title in (("regression", "regressions"), ("improvement", "improvements")):
            flips = c.flips(kind)  # type: ignore[arg-type]
            if flips:
                lines.append(f"{title} ({len(flips)}):")
                lines.extend(self._flip_lines(kind, max_flips))  # type: ignore[arg-type]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------ markdown

    def markdown(self, max_flips: int) -> str:
        c = self.c
        test, boot, t = c.mcnemar, c.bootstrap, c.thresholds
        data = self.payload()
        lines = [
            f"# Comparison: `{c.baseline_id}` (baseline) vs `{c.candidate_id}` (candidate)",
            "",
            f"**{c.verdict_line}**",
            "",
            f"Rule: a regression needs the one-sided exact McNemar p-value <= {t.alpha:g} AND "
            f"an accuracy drop >= {t.max_drop * 100:.1f} points over the paired fields.",
            "",
            "## What differs between the runs",
            "",
        ]
        if c.differs:
            lines += ["| setting | baseline | candidate |", "| --- | --- | --- |"]
            lines += [
                f"| {d.key} | {_cell(d.baseline)} | {_cell(d.candidate)} |" for d in c.differs
            ]
        else:
            lines.append("Nothing: same model, endpoint, prompt, schema and settings.")
        lines += [
            "",
            "## Documents",
            "",
            f"- paired: {len(c.documents)} ({boot.clusters} independent), "
            f"{len(c.pairs)} labeled field(s)",
        ]
        lines += [f"- {note}" for note in c.notes]
        lines += [
            "",
            "## Headline (paired fields)",
            "",
            "| metric | baseline | candidate | delta |",
            "| --- | --- | --- | --- |",
        ]
        for name, values in data["headline"].items():
            lines.append(
                f"| {name} | {values['baseline']:.3f} | {values['candidate']:.3f} "
                f"| {values['delta']:+.3f} |"
            )
        lines += [
            "",
            "## Paired test",
            "",
            "| | candidate correct | candidate not correct |",
            "| --- | --- | --- |",
            f"| baseline correct | {c.both_correct} | {test.regressions} (regressions) |",
            f"| baseline not correct | {test.improvements} (improvements) | {c.both_wrong} |",
            "",
            f"- McNemar (exact): p_worse {float(test.p_worse):.4f}, p_better "
            f"{float(test.p_better):.4f}, two-sided {float(test.p_two_sided):.4f}",
            f"- paired bootstrap over {boot.clusters} document(s) (seed {boot.seed}, "
            f"{boot.resamples} resamples): accuracy delta {_points(boot.delta)}, {boot.level:.0%} "
            f"interval [{_points(boot.low)}, {_points(boot.high)}] "
            "(reported, not part of the verdict)",
            "",
            "## By field",
            "",
            "| field | pairs | baseline correct | candidate correct | regressions | improvements |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {d.name} | {d.pairs} | {d.baseline_correct} | {d.candidate_correct} "
            f"| {d.regressions} | {d.improvements} |"
            for d in c.per_field
        ]
        lines += [
            "",
            "## Line items (aggregate, paired documents)",
            "",
            "| metric | baseline | candidate | delta |",
            "| --- | --- | --- | --- |",
        ]
        for name, values in data["rows"].items():
            lines.append(
                f"| {name} | {values['baseline']:.3f} | {values['candidate']:.3f} "
                f"| {values['delta']:+.3f} |"
            )
        lines += ["", "## Errors", ""]
        errors = error_mix(c.baseline_errors, c.candidate_errors)
        if errors:
            lines += ["| class | baseline | candidate |", "| --- | --- | --- |"]
            lines += [f"| {name} | {before} | {after} |" for name, before, after in errors]
        else:
            lines.append("None in either run.")
        for column in SLICE_COLUMNS:
            members = [s for s in c.slices if s.column == column]
            lines += [
                "",
                f"## Slice: {column}",
                "",
                "| value | labeled | baseline accuracy | candidate accuracy |",
                "| --- | --- | --- | --- |",
            ]
            lines += [
                f"| {s.value} | {s.baseline.labeled} | {s.baseline.accuracy:.3f} "
                f"| {s.candidate.accuracy:.3f} |"
                for s in members
            ]
        for kind, title in (
            ("regression", "Regressions (correct -> not correct)"),
            ("improvement", "Improvements (not correct -> correct)"),
        ):
            flips = c.flips(kind)  # type: ignore[arg-type]
            lines += ["", f"## {title}: {len(flips)}", ""]
            if not flips:
                lines.append("None.")
                continue
            lines += [
                "| document | field | expected | baseline | candidate | error class |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for pair in flips[:max_flips]:
                entry = _flip_entry(pair)
                lines.append(
                    f"| {pair.doc_id} | {pair.name} | {_cell(entry['expected'])} "
                    f"| {_side(entry['baseline'])} | {_side(entry['candidate'])} "
                    f"| {entry['candidate']['error_class'] or 'correct'} |"
                )
            if len(flips) > max_flips:
                lines += ["", f"... and {len(flips) - max_flips} more in comparison.json"]
        changed = c.flips("changed_wrong")
        lines += [
            "",
            f"Wrong in both runs but differently: {len(changed)} (see comparison.json).",
            "",
        ]
        return "\n".join(lines)
