from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from docextract.extract.runner import DocRecord, PageRef, RunRecord
from docextract.ingest.manifest import ManifestRow
from evalkit.compare import (
    Bootstrap,
    Cluster,
    CompareError,
    Comparison,
    Difference,
    FieldPair,
    compare_runs,
    error_mix,
    mcnemar_exact,
    paired_bootstrap,
)
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

    def test_equal_runs_differ_in_nothing(self, scenario: Scenario) -> None:
        assert scenario.compare().differs == []

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
        labels = [label(row.doc_id, total=Decimal("10")) for row in rows]
        manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)

        def record(doc_id: str, total: str) -> DocRecord:
            return ok(doc_id, total=extracted(Decimal(total)))

        baseline = run_dir(tmp_path, [record("doc_001", "10"), record("doc_002", "10")], name="a")
        candidate = run_dir(tmp_path, [record("doc_001", "99"), record("doc_003", "10")], name="b")
        result = compare_runs(
            baseline, candidate, manifest_path=manifest, labels_dir=labels_dir, pages_dir=pages_dir
        )
        assert result.documents == ["doc_001"]
        assert (result.only_in_baseline, result.only_in_candidate) == (["doc_002"], ["doc_003"])
        assert names(result.pairs) == [("doc_001", "total")]
        assert result.baseline.labeled == 1  # doc_002 / doc_003 are not in either side's metrics
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
