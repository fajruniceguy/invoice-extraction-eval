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
from evalkit.scorecard import NO_VALUE, SLICE_COLUMNS, Scorecard, score_run
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

    def flips(self, kind: Flip) -> list[FieldPair]:
        return [pair for pair in self.pairs if pair.flip == kind]


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
        mcnemar=mcnemar_exact(regressions, improvements),
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
    )
