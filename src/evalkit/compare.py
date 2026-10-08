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
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb

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
