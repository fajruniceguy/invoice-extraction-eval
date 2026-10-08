from fractions import Fraction

import pytest

from evalkit.compare import (
    Bootstrap,
    Cluster,
    mcnemar_exact,
    paired_bootstrap,
)

# ------------------------------------------------------------------ McNemar


class TestMcNemar:
    # b = regressions, c = improvements; X ~ Binomial(b + c, 1/2). The expectations are literal
    # fractions worked out by hand from the binomial tails.
    @pytest.mark.parametrize(
        ("b", "c", "p_worse"),
        [
            (5, 0, Fraction(1, 32)),
            (4, 0, Fraction(1, 16)),
            (6, 0, Fraction(1, 64)),
            (7, 1, Fraction(9, 256)),
            (6, 1, Fraction(1, 16)),
            (3, 3, Fraction(21, 32)),
            (2, 5, Fraction(15, 16)),
            (0, 5, Fraction(1)),
            (1, 0, Fraction(1, 2)),
        ],
    )
    def test_the_evidence_that_the_candidate_is_worse(
        self, b: int, c: int, p_worse: Fraction
    ) -> None:
        assert mcnemar_exact(b, c).p_worse == p_worse

    @pytest.mark.parametrize(
        ("b", "c", "p_better"),
        [
            (0, 5, Fraction(1, 32)),
            (5, 0, Fraction(1)),
            (2, 5, Fraction(29, 128)),
            (3, 3, Fraction(21, 32)),
        ],
    )
    def test_the_evidence_that_the_candidate_is_better_is_the_mirror_image(
        self, b: int, c: int, p_better: Fraction
    ) -> None:
        assert mcnemar_exact(b, c).p_better == p_better

    @pytest.mark.parametrize(
        ("b", "c", "p_two_sided"),
        [
            (6, 0, Fraction(1, 32)),
            (0, 6, Fraction(1, 32)),
            (5, 0, Fraction(1, 16)),
            (2, 5, Fraction(29, 64)),
            (3, 3, Fraction(1)),  # capped at 1
            (1, 1, Fraction(1)),
        ],
    )
    def test_two_sided_doubles_the_smaller_tail_and_is_capped_at_one(
        self, b: int, c: int, p_two_sided: Fraction
    ) -> None:
        assert mcnemar_exact(b, c).p_two_sided == p_two_sided

    def test_no_flips_show_nothing(self) -> None:
        result = mcnemar_exact(0, 0)
        assert (result.p_worse, result.p_better, result.p_two_sided) == (1, 1, 1)
        assert (result.regressions, result.improvements) == (0, 0)

    def test_it_stays_exact_for_large_counts(self) -> None:
        result = mcnemar_exact(900, 100)
        assert result.p_worse < Fraction(1, 10**100)  # a fraction, not float underflow noise
        assert mcnemar_exact(500, 500).p_two_sided == 1

    @pytest.mark.parametrize(("b", "c"), [(-1, 0), (0, -2)])
    def test_negative_counts_are_refused(self, b: int, c: int) -> None:
        with pytest.raises(ValueError, match="negative"):
            mcnemar_exact(b, c)


# ------------------------------------------------------------------ bootstrap


def cluster(pairs: int, baseline: int, candidate: int) -> Cluster:
    return Cluster(pairs=pairs, baseline_correct=baseline, candidate_correct=candidate)


class TestBootstrap:
    def test_a_hand_checkable_case(self) -> None:
        # one cluster loses its only field (delta -1), the other keeps it (delta 0). Two draws:
        # AA = -1 (1/4), AB or BA = -0.5 (1/2), BB = 0 (1/4): the middle 95% spans -1 .. 0.
        result = paired_bootstrap([cluster(1, 1, 0), cluster(1, 1, 1)], seed=1, resamples=2000)
        assert result.delta == -0.5
        assert (result.low, result.high) == (-1.0, 0.0)
        assert result.clusters == 2 and result.resamples == 2000 and result.level == 0.95

    def test_identical_clusters_give_a_zero_width_interval(self) -> None:
        result = paired_bootstrap([cluster(13, 13, 8)] * 4, seed=3, resamples=500)
        assert result.low == result.high == result.delta == pytest.approx(-5 / 13)

    def test_the_interval_brackets_the_observed_difference(self) -> None:
        clusters = [cluster(13, 13, 13), cluster(13, 13, 5), cluster(13, 12, 13), cluster(9, 9, 9)]
        result = paired_bootstrap(clusters, seed=1)
        assert result.low <= result.delta <= result.high
        assert result.low < result.high
        assert result.low >= -1 and result.high <= 1

    def test_the_same_seed_gives_the_same_interval_another_seed_another(self) -> None:
        clusters = [cluster(13, 13, 9), cluster(13, 12, 13), cluster(8, 8, 3), cluster(5, 5, 5)]
        first = paired_bootstrap(clusters, seed=7, resamples=300)
        assert paired_bootstrap(clusters, seed=7, resamples=300) == first
        other = paired_bootstrap(clusters, seed=8, resamples=300)
        assert (other.low, other.high) != (first.low, first.high)
        assert other.seed == 8

    def test_a_narrower_level_gives_a_narrower_interval(self) -> None:
        clusters = [cluster(10, 10, 4), cluster(10, 10, 10), cluster(10, 9, 10), cluster(10, 8, 9)]
        wide = paired_bootstrap(clusters, seed=1, level=0.95)
        narrow = paired_bootstrap(clusters, seed=1, level=0.5)
        assert wide.low <= narrow.low and narrow.high <= wide.high
        assert (wide.low, wide.high) != (narrow.low, narrow.high)

    def test_it_returns_a_bootstrap_record(self) -> None:
        assert isinstance(paired_bootstrap([cluster(2, 1, 1)], seed=1, resamples=10), Bootstrap)

    @pytest.mark.parametrize(
        "kwargs", [{"resamples": 0}, {"level": 0.0}, {"level": 1.0}], ids=["none", "zero", "one"]
    )
    def test_nonsense_settings_are_refused(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap([cluster(2, 1, 1)], seed=1, **kwargs)  # type: ignore[arg-type]

    def test_nothing_to_resample_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no clusters"):
            paired_bootstrap([], seed=1)
