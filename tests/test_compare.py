import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from docextract.extract.runner import DocRecord, PageRef, RunRecord
from docextract.extract.schema import LineItem
from docextract.ingest.manifest import ManifestRow
from evalkit.compare import (
    Bootstrap,
    Cluster,
    CompareError,
    Comparison,
    Decision,
    Difference,
    FieldPair,
    Thresholds,
    compare_runs,
    decide,
    error_mix,
    mcnemar_exact,
    paired_bootstrap,
)
from evalkit.groundtruth import GTLineItem
from test_scorecard import absent, abstained, dataset, document, extracted, label, run_dir

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

    def test_the_interval_is_the_central_part_not_the_extremes(self) -> None:
        # one cluster in six differs; a resample draws it k ~ Binomial(6, 1/6) times, so
        # P(k >= 4) = 0.9% < 2.5% < P(k >= 3) = 6.2%: the 2.5% / 97.5% points are at k = 3, not 6
        lose_one = [cluster(1, 1, 0)] + [cluster(1, 1, 1)] * 5
        result = paired_bootstrap(lose_one, seed=1)
        assert (result.low, result.high) == (-0.5, 0.0)
        gain_one = [cluster(1, 0, 1)] + [cluster(1, 1, 1)] * 5
        result = paired_bootstrap(gain_one, seed=1)
        assert (result.low, result.high) == (0.0, 0.5)

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


# ------------------------------------------------------------------ two runs


def manifest_row(doc_id: str, **overrides: object) -> ManifestRow:
    values: dict[str, object] = {
        "doc_id": doc_id,
        "raw_files": (f"{doc_id}.pdf",),
        "content_group": doc_id,
        "doc_type": "invoice",
        "capture": "born_digital",
        "split": "dev",
    }
    values.update(overrides)
    return ManifestRow.model_validate(values)


def retag(folder: Path, **updates: object) -> None:
    """Change what run.json says about the run (prompt hash, settings...)."""
    run = RunRecord.model_validate_json((folder / "run.json").read_text(encoding="utf-8"))
    (folder / "run.json").write_text(
        run.model_copy(update=updates).model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def ok(doc_id: str, **fields: object) -> DocRecord:
    return DocRecord(doc_id=doc_id, status="ok", validated=document(**fields))


class Scenario:
    """doc_001 has seven labeled fields, doc_002 two. Baseline vs candidate on doc_001:
    vendor_name, buyer_name and discount regress; vendor_address improves; total is wrong both
    times in a different way; document_number and subtotal do not change. doc_002 is unchanged."""

    def __init__(self, tmp_path: Path) -> None:
        rows = [manifest_row("doc_001"), manifest_row("doc_002", doc_type="faktur", split="test")]
        labels = [
            label(
                "doc_001",
                vendor_name="Acme",
                vendor_address="Jl Mawar 1",
                buyer_name="Beta",
                document_number="INV-1",
                subtotal=Decimal("9"),
                total=Decimal("10"),
                discount=None,
            ),
            label("doc_002", vendor_name="Gamma", total=Decimal("20")),
        ]
        self.manifest, self.labels, self.pages = dataset(tmp_path, rows, labels)
        self.tmp = tmp_path
        self.second = ok("doc_002", vendor_name=extracted("Gamma"), total=extracted(Decimal("20")))
        self.baseline = run_dir(
            tmp_path,
            [
                ok(
                    "doc_001",
                    vendor_name=extracted("Acme"),
                    vendor_address=extracted("Other St"),
                    buyer_name=extracted("Beta"),
                    document_number=extracted("INV-1"),
                    subtotal=extracted(Decimal("8")),
                    total=extracted(Decimal("11")),
                    discount=absent(),
                ),
                self.second,
            ],
            name="baseline",
        )
        self.candidate = run_dir(
            tmp_path,
            [
                ok(
                    "doc_001",
                    vendor_name=extracted("Zeta"),
                    vendor_address=extracted("Jl Mawar 1"),
                    buyer_name=abstained(),
                    document_number=extracted("INV-1"),
                    subtotal=extracted(Decimal("8")),
                    total=extracted(Decimal("12")),
                    discount=extracted(Decimal("5")),
                ),
                self.second,
            ],
            name="candidate",
        )

    def compare(self, baseline: Path | None = None, candidate: Path | None = None) -> Comparison:
        return compare_runs(
            baseline or self.baseline,
            candidate or self.candidate,
            manifest_path=self.manifest,
            labels_dir=self.labels,
            pages_dir=self.pages,
        )


@pytest.fixture
def scenario(tmp_path: Path) -> Scenario:
    return Scenario(tmp_path)


def names(pairs: list[FieldPair]) -> list[tuple[str, str]]:
    return [(pair.doc_id, pair.name) for pair in pairs]


class TestFlips:
    def test_every_flip_is_listed_by_kind(self, scenario: Scenario) -> None:
        result = scenario.compare()
        assert names(result.flips("regression")) == [
            ("doc_001", "vendor_name"),
            ("doc_001", "buyer_name"),
            ("doc_001", "discount"),
        ]
        assert names(result.flips("improvement")) == [("doc_001", "vendor_address")]
        assert names(result.flips("changed_wrong")) == [("doc_001", "total")]
        assert names(result.flips("unchanged")) == [
            ("doc_001", "document_number"),
            ("doc_001", "subtotal"),
            ("doc_002", "vendor_name"),
            ("doc_002", "total"),
        ]
        assert len(result.pairs) == 9

    def test_the_two_by_two_table_adds_up(self, scenario: Scenario) -> None:
        result = scenario.compare()
        assert (result.both_correct, result.mcnemar.regressions) == (3, 3)
        assert (result.mcnemar.improvements, result.both_wrong) == (1, 2)
        assert result.both_correct + 3 + 1 + result.both_wrong == len(result.pairs)
        assert result.mcnemar.p_worse == Fraction(5, 16)  # P(X >= 3), X ~ Binomial(4, 1/2)

    def test_metrics_are_each_runs_own_over_the_paired_fields(self, scenario: Scenario) -> None:
        result = scenario.compare()
        assert (result.baseline.labeled, result.baseline.correct) == (9, 6)
        assert (result.candidate.labeled, result.candidate.correct) == (9, 4)
        assert result.candidate.abstained == 1 and result.candidate.should_be_absent == 1

    def test_the_error_mix_of_each_run_and_its_delta(self, scenario: Scenario) -> None:
        result = scenario.compare()
        assert result.baseline_errors == {"numeric_mismatch": 2, "string_mismatch": 1}
        assert result.candidate_errors == {
            "numeric_mismatch": 2,
            "string_mismatch": 1,
            "hallucinated_value": 1,
            "abstained_on_value": 1,
        }
        assert error_mix(result.baseline_errors, result.candidate_errors) == [
            ("numeric_mismatch", 2, 2),
            ("string_mismatch", 1, 1),
            ("hallucinated_value", 0, 1),
            ("abstained_on_value", 0, 1),
        ]

    def test_the_per_field_table_follows_the_schema_order(self, scenario: Scenario) -> None:
        by_name = {delta.name: delta for delta in scenario.compare().per_field}
        assert list(by_name) == [
            "vendor_name",
            "vendor_address",
            "buyer_name",
            "document_number",
            "subtotal",
            "discount",
            "total",
        ]
        vendor = by_name["vendor_name"]
        assert (vendor.pairs, vendor.baseline_correct, vendor.candidate_correct) == (2, 2, 1)
        assert (vendor.regressions, vendor.improvements) == (1, 0)
        assert by_name["vendor_address"].improvements == 1

    def test_slices_use_the_manifest_columns(self, scenario: Scenario) -> None:
        slices = {(s.column, s.value): s for s in scenario.compare().slices}
        dev = slices[("split", "dev")]
        assert (dev.baseline.labeled, dev.baseline.correct, dev.candidate.correct) == (7, 4, 2)
        test = slices[("split", "test")]
        assert (test.baseline.correct, test.candidate.correct) == (2, 2)
        assert slices[("doc_type", "faktur")].baseline.labeled == 2

    def test_identical_runs_have_no_flips(self, scenario: Scenario) -> None:
        result = scenario.compare(scenario.baseline, scenario.baseline)
        assert {pair.flip for pair in result.pairs} == {"unchanged"}
        assert result.mcnemar.p_worse == 1 and result.differs == []
        assert result.baseline == result.candidate

    def test_a_document_that_failed_regresses_every_field(self, scenario: Scenario) -> None:
        failed = run_dir(
            scenario.tmp,
            [scenario.second, DocRecord(doc_id="doc_001", status="failed", error="HTTP 500")],
            name="failing",
        )
        result = scenario.compare(scenario.baseline, failed)
        assert result.status_changes == [("doc_001", "ok", "failed")]
        assert result.candidate.no_prediction == 7
        assert len(result.flips("regression")) == 4  # the three that were right + discount
        assert any("doc_001: the run status changed from ok to failed" in n for n in result.notes)


class TestWhatDiffers:
    def test_the_prompt_the_model_and_the_settings_are_compared(self, scenario: Scenario) -> None:
        retag(
            scenario.candidate,
            prompt_sha256="q" * 64,
            model="other-model",
            settings={"cache": "off", "max_pages": 15, "image_max_long_edge": 1024},
        )
        result = scenario.compare()
        assert result.differs == [
            Difference("model", "fake-model", "other-model"),
            Difference("prompt_sha256", "p" * 64, "q" * 64),
            Difference("settings.image_max_long_edge", None, 1024),
        ]

    def test_the_endpoint_and_the_schema_are_compared_too(self, scenario: Scenario) -> None:
        retag(scenario.candidate, endpoint="http://other.invalid/v1", schema_sha256="t" * 64)
        assert scenario.compare().differs == [
            Difference("endpoint", "http://example.invalid/v1", "http://other.invalid/v1"),
            Difference("schema_sha256", "s" * 64, "t" * 64),
        ]

    def test_equal_runs_differ_in_nothing(self, scenario: Scenario) -> None:
        assert scenario.compare().differs == []

    def test_a_document_with_no_pages_in_one_run_is_not_a_page_mismatch(
        self, scenario: Scenario
    ) -> None:
        pages = [PageRef(file="p1.png", sha256="a" * 64)]
        sent = run_dir(
            scenario.tmp, [scenario.second.model_copy(update={"pages": pages})], name="sent"
        )
        refused = run_dir(
            scenario.tmp,
            [DocRecord(doc_id="doc_002", status="refused", error="not ingested")],
            name="refused",
        )
        assert scenario.compare(sent, refused).pages_differ == []

    def test_different_stored_pages_are_reported(self, scenario: Scenario) -> None:
        def with_pages(digest: str) -> DocRecord:
            pages = [PageRef(file="p1.png", sha256=digest * 64)]
            return scenario.second.model_copy(update={"pages": pages})

        one = run_dir(scenario.tmp, [with_pages("a")], name="one")
        two = run_dir(scenario.tmp, [with_pages("b")], name="two")
        same = run_dir(scenario.tmp, [with_pages("a")], name="same")
        assert scenario.compare(one, two).pages_differ == ["doc_002"]
        assert any("different stored pages" in n for n in scenario.compare(one, two).notes)
        assert scenario.compare(one, same).pages_differ == []


class TestPairing:
    def test_only_the_documents_scored_in_both_runs_are_compared(self, tmp_path: Path) -> None:
        rows = [manifest_row("doc_001"), manifest_row("doc_002"), manifest_row("doc_003")]
        tea = [GTLineItem(description="Tea", amount=Decimal("1"))]
        labels = [
            label("doc_001", total=Decimal("10")),
            label("doc_002", total=Decimal("10"), line_items=tea),
            label("doc_003", total=Decimal("10"), line_items=tea),
        ]
        manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)

        def record(doc_id: str, total: str) -> DocRecord:
            return ok(doc_id, total=extracted(Decimal(total)))

        # the documents scored in one run only are WRONG and miss their row: if they leaked into
        # the metrics the errors and the missed rows below would not be zero
        baseline = run_dir(tmp_path, [record("doc_001", "10"), record("doc_002", "99")], name="a")
        candidate = run_dir(tmp_path, [record("doc_001", "99"), record("doc_003", "99")], name="b")
        result = compare_runs(
            baseline, candidate, manifest_path=manifest, labels_dir=labels_dir, pages_dir=pages_dir
        )
        assert result.documents == ["doc_001"]
        assert (result.only_in_baseline, result.only_in_candidate) == (["doc_002"], ["doc_003"])
        assert names(result.pairs) == [("doc_001", "total")]
        assert (result.baseline.labeled, result.baseline.correct) == (1, 1)
        assert (result.candidate.labeled, result.candidate.correct) == (1, 0)
        assert result.baseline_errors == {}
        assert result.candidate_errors == {"numeric_mismatch": 1}
        assert (result.baseline_rows.missed, result.candidate_rows.missed) == (0, 0)
        assert any("only in the baseline" in n and "doc_002" in n for n in result.notes)
        assert any("only in the candidate" in n and "doc_003" in n for n in result.notes)

    def test_no_document_in_common_is_an_error(self, tmp_path: Path) -> None:
        rows = [manifest_row("doc_001"), manifest_row("doc_002")]
        labels = [label(row.doc_id, total=Decimal("10")) for row in rows]
        manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)
        a = run_dir(tmp_path, [ok("doc_001", total=extracted(Decimal("10")))], name="a")
        b = run_dir(tmp_path, [ok("doc_002", total=extracted(Decimal("10")))], name="b")
        with pytest.raises(CompareError, match="nothing to compare"):
            compare_runs(a, b, manifest_path=manifest, labels_dir=labels_dir, pages_dir=pages_dir)

    def test_a_variant_and_its_owner_are_one_independent_document(self, tmp_path: Path) -> None:
        rows = [
            manifest_row("doc_001"),
            ManifestRow(
                doc_id="doc_001-deg-light",
                capture="synthetic_degraded",
                variant="light",
                gt_from="doc_001",
                content_group="doc_001",
                doc_type="invoice",
                split="dev",
            ),
            manifest_row("doc_002"),
        ]
        labels = [label("doc_001", total=Decimal("10")), label("doc_002", total=Decimal("10"))]
        manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)
        records = [
            ok(doc_id, total=extracted(Decimal("10")))
            for doc_id in ("doc_001", "doc_001-deg-light", "doc_002")
        ]
        a = run_dir(tmp_path, records, name="a")
        result = compare_runs(
            a, a, manifest_path=manifest, labels_dir=labels_dir, pages_dir=pages_dir
        )
        assert len(result.pairs) == 3 and result.bootstrap.clusters == 2

    def test_a_blank_manifest_column_is_its_own_slice_value(self, tmp_path: Path) -> None:
        rows = [manifest_row("doc_001", capture=None), manifest_row("doc_002")]
        labels = [label(row.doc_id, total=Decimal("10")) for row in rows]
        manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)
        records = [ok(row.doc_id, total=extracted(Decimal("10"))) for row in rows]
        a = run_dir(tmp_path, records, name="a")
        result = compare_runs(
            a, a, manifest_path=manifest, labels_dir=labels_dir, pages_dir=pages_dir
        )
        captures = {s.value: s.baseline.labeled for s in result.slices if s.column == "capture"}
        assert captures == {"(none)": 1, "born_digital": 1}

    @pytest.mark.parametrize(("documents", "caution"), [(9, True), (10, False)])
    def test_few_independent_documents_come_with_a_caution(
        self, tmp_path: Path, documents: int, caution: bool
    ) -> None:
        rows = [manifest_row(f"doc_{i:03d}") for i in range(documents)]
        labels = [label(row.doc_id, total=Decimal("10")) for row in rows]
        manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)
        records = [ok(row.doc_id, total=extracted(Decimal("10"))) for row in rows]
        a = run_dir(tmp_path, records, name="a")
        result = compare_runs(
            a, a, manifest_path=manifest, labels_dir=labels_dir, pages_dir=pages_dir
        )
        assert result.bootstrap.clusters == documents
        assert any("independent document" in note for note in result.notes) is caution


# ------------------------------------------------------------------ the verdict


def gate(
    regressions: int,
    improvements: int,
    *,
    pairs: int,
    baseline_correct: int,
    candidate_correct: int,
    alpha: float = 0.05,
    max_drop: float = 0.02,
) -> Decision:
    return decide(
        mcnemar_exact(regressions, improvements),
        baseline_correct=baseline_correct,
        candidate_correct=candidate_correct,
        pairs=pairs,
        thresholds=Thresholds(alpha=alpha, max_drop=max_drop),
    )


class TestDecide:
    def test_a_significant_and_material_drop_is_a_regression(self) -> None:
        decision = gate(5, 0, pairs=39, baseline_correct=39, candidate_correct=34)
        assert decision.verdict == "regression"
        assert (decision.significant, decision.material) == (True, True)
        assert decision.drop == Fraction(5, 39)

    def test_p_exactly_at_alpha_fires_just_above_does_not(self) -> None:
        # p_worse(5, 0) is exactly 1/32 = 0.03125, which a float holds exactly
        kwargs = {"pairs": 39, "baseline_correct": 39, "candidate_correct": 34}
        assert gate(5, 0, alpha=0.03125, **kwargs).verdict == "regression"
        assert gate(5, 0, alpha=0.03124, **kwargs).verdict == "ok"

    def test_a_drop_exactly_at_max_drop_fires_just_below_does_not(self) -> None:
        kwargs = {"pairs": 10, "baseline_correct": 10, "candidate_correct": 9}
        # 1/10 is not exact in a float (1 - 0.9 is 0.09999999999999998): the gate must still fire
        assert gate(6, 0, max_drop=0.1, **kwargs).verdict == "regression"
        assert gate(6, 0, max_drop=0.11, **kwargs).verdict == "ok"

    def test_significant_but_small_is_ok(self) -> None:
        decision = gate(6, 0, pairs=1000, baseline_correct=1000, candidate_correct=994)
        assert (decision.significant, decision.material, decision.verdict) == (True, False, "ok")

    def test_large_but_not_significant_is_ok(self) -> None:
        decision = gate(3, 0, pairs=39, baseline_correct=39, candidate_correct=36)
        assert (decision.significant, decision.material, decision.verdict) == (False, True, "ok")

    def test_an_improvement_is_ok(self) -> None:
        decision = gate(0, 8, pairs=39, baseline_correct=30, candidate_correct=38)
        assert decision.verdict == "ok" and decision.drop < 0

    def test_no_drop_requirement_leaves_only_significance(self) -> None:
        kwargs = {"pairs": 39, "baseline_correct": 39, "candidate_correct": 34}
        assert gate(5, 0, max_drop=0.0, **kwargs).verdict == "regression"
        assert gate(1, 0, max_drop=0.0, **kwargs).verdict == "ok"

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_must_be_strictly_between_0_and_1(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            Thresholds(alpha=alpha)

    @pytest.mark.parametrize("max_drop", [-0.01, 1.01])
    def test_max_drop_must_be_a_fraction(self, max_drop: float) -> None:
        with pytest.raises(ValueError, match="max_drop"):
            Thresholds(max_drop=max_drop)

    def test_the_defaults_are_five_percent_and_two_points(self) -> None:
        assert (Thresholds().alpha, Thresholds().max_drop) == (0.05, 0.02)


def many_documents(tmp_path: Path, count: int, wrong: int) -> Scenario:
    """`count` one-field documents: the baseline gets all right, the candidate `wrong` of them."""
    scenario = Scenario.__new__(Scenario)
    rows = [manifest_row(f"doc_{i:03d}") for i in range(count)]
    labels = [label(row.doc_id, total=Decimal("10")) for row in rows]
    scenario.tmp = tmp_path
    scenario.manifest, scenario.labels, scenario.pages = dataset(tmp_path, rows, labels)

    def records(wrong_ids: int) -> list[DocRecord]:
        return [
            ok(row.doc_id, total=extracted(Decimal("99" if i < wrong_ids else "10")))
            for i, row in enumerate(rows)
        ]

    scenario.baseline = run_dir(tmp_path, records(0), name="baseline")
    scenario.candidate = run_dir(tmp_path, records(wrong), name="candidate")
    return scenario


class TestVerdictOnRuns:
    def test_a_degraded_run_is_a_regression(self, tmp_path: Path) -> None:
        result = many_documents(tmp_path, count=12, wrong=7).compare()
        assert result.decision.verdict == "regression"
        assert result.mcnemar.p_worse == Fraction(1, 128)
        assert result.decision.drop == Fraction(7, 12)
        assert result.verdict_line.startswith("VERDICT: REGRESSION (p_worse 0.0078 <= alpha 0.05;")

    def test_the_scenario_has_too_few_flips_to_be_significant(self, scenario: Scenario) -> None:
        result = scenario.compare()
        assert result.decision.verdict == "ok" and not result.decision.significant
        assert result.verdict_line == (
            "VERDICT: OK (p_worse 0.3125 > alpha 0.05; accuracy drop 22.2 pts >= 2.0)"
        )

    def test_the_thresholds_are_passed_through(self, scenario: Scenario) -> None:
        result = compare_runs(
            scenario.baseline,
            scenario.candidate,
            manifest_path=scenario.manifest,
            labels_dir=scenario.labels,
            pages_dir=scenario.pages,
            thresholds=Thresholds(alpha=0.5, max_drop=0.0),
        )
        assert result.decision.verdict == "regression"
        assert result.thresholds == Thresholds(alpha=0.5, max_drop=0.0)

    def test_swapping_the_runs_turns_a_regression_into_an_improvement(self, tmp_path: Path) -> None:
        scenario = many_documents(tmp_path, count=12, wrong=7)
        result = scenario.compare(scenario.candidate, scenario.baseline)
        assert result.decision.verdict == "ok" and result.mcnemar.improvements == 7
        assert result.mcnemar.p_better == Fraction(1, 128)


# ------------------------------------------------------------------ the reports


class TestReports:
    def test_json_is_byte_identical_for_the_same_inputs(self, scenario: Scenario) -> None:
        first, second = scenario.compare(), scenario.compare()
        assert first.to_json() == second.to_json()
        assert first.to_markdown() == second.to_markdown()
        assert first.to_text() == second.to_text()

    def test_json_holds_the_documented_sections(self, scenario: Scenario) -> None:
        data = json.loads(scenario.compare().to_json())
        assert list(data) == sorted(data)  # sorted keys
        assert data["baseline"] == "baseline" and data["candidate"] == "candidate"
        assert data["verdict"] == "ok" and data["thresholds"] == {"alpha": 0.05, "max_drop": 0.02}
        assert data["fields"] == {
            "paired": 9,
            "both_correct": 3,
            "regressions": 3,
            "improvements": 1,
            "both_wrong": 2,
        }
        assert data["mcnemar"]["p_worse"] == 0.3125
        assert data["documents"]["paired"] == ["doc_001", "doc_002"]
        assert data["bootstrap"]["clusters"] == 2 and data["bootstrap"]["seed"] == 1
        assert data["headline"]["accuracy"] == {
            "baseline": 6 / 9,
            "candidate": 4 / 9,
            "delta": 4 / 9 - 6 / 9,
        }
        assert {e["class"] for e in data["errors"]} >= {"hallucinated_value", "abstained_on_value"}

    def test_each_flip_carries_what_was_expected_and_what_each_run_said(
        self, scenario: Scenario
    ) -> None:
        flips = {
            (f["doc_id"], f["field"]): f
            for f in json.loads(scenario.compare().to_json())["flips"]["regression"]
        }
        assert flips[("doc_001", "vendor_name")] == {
            "doc_id": "doc_001",
            "field": "vendor_name",
            "flip": "regression",
            "expected": "Acme",
            "baseline": {
                "predicted": "Acme",
                "status": "extracted",
                "outcome": "correct",
                "error_class": None,
            },
            "candidate": {
                "predicted": "Zeta",
                "status": "extracted",
                "outcome": "value_wrong",
                "error_class": "string_mismatch",
            },
        }
        assert flips[("doc_001", "buyer_name")]["candidate"]["status"] == "abstained"
        discount = flips[("doc_001", "discount")]
        assert discount["expected"] is None
        assert discount["candidate"]["predicted"] == "5"
        assert discount["candidate"]["error_class"] == "hallucinated_value"

    def test_the_text_starts_with_the_verdict_and_lists_the_regressions(
        self, scenario: Scenario
    ) -> None:
        result = scenario.compare()
        text = result.to_text()
        assert text.splitlines()[0] == result.verdict_line
        assert "regressions (3):" in text and "improvements (1):" in text
        assert (
            "doc_001 vendor_name: expected Acme; baseline Acme -> candidate Zeta (string_mismatch)"
            in text
        )
        assert "baseline (absent) -> candidate 5 (hallucinated_value)" in text
        assert "differs: nothing" in text

    def test_long_flip_lists_are_cut_in_the_text_and_markdown_not_in_json(
        self, tmp_path: Path
    ) -> None:
        result = many_documents(tmp_path, count=12, wrong=7).compare()
        text, markdown = result.to_text(max_flips=2), result.to_markdown(max_flips=2)
        assert "... and 5 more in comparison.json" in text
        assert "... and 5 more in comparison.json" in markdown
        assert sum(line.startswith("  doc_") for line in text.splitlines()) == 2
        regressions = markdown.split("## Regressions")[1].split("## Improvements")[0]
        assert sum(line.startswith("| doc_") for line in regressions.splitlines()) == 2
        assert len(json.loads(result.to_json())["flips"]["regression"]) == 7

    def test_markdown_has_the_sections_and_the_flip_table(self, scenario: Scenario) -> None:
        markdown = scenario.compare().to_markdown()
        for heading in (
            "# Comparison: `baseline` (baseline) vs `candidate` (candidate)",
            "## What differs between the runs",
            "## Headline (paired fields)",
            "## Paired test",
            "## By field",
            "## Line items (aggregate, paired documents)",
            "## Errors",
            "## Slice: split",
            "## Regressions (correct -> not correct): 3",
            "## Improvements (not correct -> correct): 1",
        ):
            assert heading in markdown, heading
        assert (
            "| doc_001 | buyer_name | Beta | Beta | (abstained) | abstained_on_value |" in markdown
        )
        assert "| baseline correct | 3 | 3 (regressions) |" in markdown

    def test_a_pipe_in_a_value_cannot_break_the_table(self, scenario: Scenario) -> None:
        piped = run_dir(
            scenario.tmp,
            [
                ok(
                    "doc_001",
                    vendor_name=extracted("Zeta | Ltd\nBranch"),
                    buyer_name=extracted("Beta"),
                ),
                scenario.second,
            ],
            name="piped",
        )
        markdown = scenario.compare(scenario.baseline, piped).to_markdown()
        assert "| doc_001 | vendor_name | Acme | Acme | Zeta / Ltd Branch |" in markdown

    def test_a_failed_document_reads_as_no_prediction(self, scenario: Scenario) -> None:
        failed = run_dir(
            scenario.tmp,
            [scenario.second, DocRecord(doc_id="doc_001", status="failed", error="x")],
            name="failing",
        )
        text = scenario.compare(scenario.baseline, failed).to_text()
        assert "candidate (no prediction) (no_prediction)" in text

    def test_what_differs_is_shown(self, scenario: Scenario) -> None:
        retag(scenario.candidate, prompt_sha256="q" * 64)
        result = scenario.compare()
        assert f"prompt_sha256: {'p' * 39}…" in result.to_text()
        assert "| prompt_sha256 |" in result.to_markdown()

    def test_row_metrics_that_would_be_vacuous_read_n_a(self, tmp_path: Path) -> None:
        # no row of the candidate matches: "cell accuracy 1.000" would be a ratio over nothing
        tea = [
            GTLineItem(
                description="Tea", quantity=Decimal(2), unit_price=Decimal(5), amount=Decimal(10)
            )
        ]
        manifest, labels_dir, pages_dir = dataset(
            tmp_path,
            [manifest_row("doc_001")],
            [label("doc_001", total=Decimal("10"), line_items=tea)],
        )

        def row(description: str, quantity: str, price: str, amount: str) -> LineItem:
            return LineItem(
                description=description,
                quantity=Decimal(quantity),
                unit_price=Decimal(price),
                amount=Decimal(amount),
                status="extracted",
                confidence=1.0,
            )

        def run(name: str, line: LineItem) -> Path:
            record = ok("doc_001", total=extracted(Decimal("10")), line_items=[line])
            return run_dir(tmp_path, [record], name=name)

        result = compare_runs(
            run("a", row("Tea", "2", "5", "10")),
            run("b", row("Coffee", "7", "0.5", "0.01")),
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
        )
        assert (result.baseline_rows.matched, result.candidate_rows.matched) == (1, 0)
        assert result.candidate_rows.cells_compared == 0
        assert "cell accuracy 1.000 -> n/a" in result.to_text()
        markdown = result.to_markdown()
        assert "| cell_accuracy | 1.000 | n/a | n/a |" in markdown
        assert "| row_accuracy | 1.000 | n/a | n/a |" in markdown
        assert "| matched | 1 | 0 | -1 |" in markdown
        assert "| row_f1 | 1.000 | 0.000 | -1.000 |" in markdown
