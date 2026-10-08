import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from docextract.extract.runner import DocRecord
from evalkit.compare import Thresholds, compare_runs
from evalkit.scorecard import score_run
from evalkit.sweep import (
    BASELINE,
    DEFAULT_TOLERANCE,
    SweepPoint,
    SweepRow,
    before_after,
    build_sweep,
    choose,
    to_json,
    to_markdown,
)
from test_compare import manifest_row, ok
from test_scorecard import dataset, extracted, label, run_dir

# ------------------------------------------------------------------ the rule, on literal rows


def row(
    label: str = "v",
    *,
    drop: Fraction = Fraction(0),
    cost: str | None = "0.05",
    latency: float | None = 50_000.0,
    failed: int = 0,
    verdict: str = "ok",
    independent: int = 12,
    p_worse: Fraction = Fraction(1, 2),
) -> SweepRow:
    return SweepRow(
        label=label,
        knob="effort",
        setting={"reasoning_effort": label},
        run_id=f"run-{label}",
        documents=10,
        failed=failed,
        accuracy=0.9 - float(drop),
        selective_accuracy=0.95,
        coverage=0.9,
        row_f1=0.9,
        cost_per_doc=None if cost is None else Decimal(cost),
        list_cost_per_doc=None if cost is None else Decimal(cost),
        latency_p50_ms=latency,
        latency_p95_ms=None if latency is None else latency * 1.5,
        input_tokens=1000,
        output_tokens=500,
        drop=drop,
        p_worse=p_worse,
        interval_low=-0.05,
        interval_high=0.03,
        independent=independent,
        verdict=verdict,
    )


def baseline_row(cost: str = "0.10", latency: float = 100_000.0) -> SweepRow:
    return row(BASELINE, cost=cost, latency=latency, verdict=BASELINE)


class TestChoose:
    def test_the_cheapest_eligible_variant_wins(self) -> None:
        rows = [baseline_row(), row("a", cost="0.06"), row("b", cost="0.04"), row("c", cost="0.08")]
        choice = choose(rows)
        assert choice.label == "b"
        assert choice.reasons == {"a": "eligible", "b": "eligible", "c": "eligible"}

    def test_a_failed_document_disqualifies_even_the_cheapest(self) -> None:
        rows = [baseline_row(), row("cheap", cost="0.01", failed=1), row("ok", cost="0.07")]
        choice = choose(rows)
        assert choice.label == "ok"
        assert choice.reasons["cheap"] == "1 document(s) failed or were refused"

    def test_a_loss_exactly_at_the_tolerance_is_accepted_a_hair_more_is_not(self) -> None:
        exact = choose([baseline_row(), row("a", drop=DEFAULT_TOLERANCE, cost="0.05")])
        assert exact.label == "a"
        over = choose([baseline_row(), row("a", drop=DEFAULT_TOLERANCE + Fraction(1, 10_000))])
        assert over.label == BASELINE
        assert "more than the 2.0-point tolerance" in over.reasons["a"]

    def test_a_large_but_not_significant_drop_is_rejected(self) -> None:
        # the gate says ok (p is large) but 8 points are gone: "cannot tell" is not "no loss"
        rows = [
            baseline_row(),
            row("a", drop=Fraction(8, 100), verdict="ok", p_worse=Fraction(3, 10)),
        ]
        choice = choose(rows)
        assert choice.label == BASELINE and "8.0 points" in choice.reasons["a"]

    def test_the_regression_gate_rejects_even_within_the_tolerance(self) -> None:
        rows = [baseline_row(), row("a", drop=Fraction(2, 100), verdict="regression")]
        choice = choose(rows)
        assert choice.label == BASELINE
        assert choice.reasons["a"] == "the regression gate fired against the baseline"

    def test_equal_cost_goes_to_the_faster_one_then_to_the_label(self) -> None:
        rows = [
            baseline_row(),
            row("slow", cost="0.05", latency=80_000.0),
            row("fast", cost="0.05", latency=40_000.0),
        ]
        assert choose(rows).label == "fast"
        tie = [baseline_row(), row("b", cost="0.05"), row("a", cost="0.05")]
        assert choose(tie).label == "a"

    def test_a_variant_must_be_cheaper_than_the_baseline(self) -> None:
        rows = [
            baseline_row(),
            row("same", cost="0.10", latency=100_000.0),
            row("dearer", cost="0.2"),
        ]
        choice = choose(rows)
        assert choice.label == BASELINE
        assert choice.reasons == {
            "same": "not cheaper than the baseline",
            "dearer": "not cheaper than the baseline",
        }

    def test_as_cheap_but_faster_counts_as_better(self) -> None:
        rows = [baseline_row(), row("faster", cost="0.10", latency=60_000.0)]
        assert choose(rows).label == "faster"

    def test_unknown_cost_or_latency_never_wins(self) -> None:
        assert choose([baseline_row(), row("a", cost=None)]).label == BASELINE
        no_latency = [baseline_row(), row("a", cost="0.10", latency=None)]
        assert choose(no_latency).label == BASELINE
        assert (
            choose(
                [baseline_row(cost="0.10", latency=100_000.0), row("a", cost="0.05", latency=None)]
            ).label
            == "a"
        )

    def test_nothing_qualifying_keeps_the_baseline_and_says_why_each_failed(self) -> None:
        rows = [baseline_row(), row("a", failed=2), row("b", drop=Fraction(5, 100))]
        choice = choose(rows)
        assert choice.label == BASELINE and "baseline stays" in choice.why
        assert set(choice.reasons) == {"a", "b"} and "eligible" not in choice.reasons.values()

    def test_the_tolerance_is_a_parameter(self) -> None:
        rows = [baseline_row(), row("a", drop=Fraction(4, 100))]
        assert choose(rows).label == BASELINE
        assert choose(rows, tolerance=Fraction(5, 100)).label == "a"

    def test_a_sweep_with_only_the_baseline_keeps_it(self) -> None:
        assert choose([baseline_row()]).label == BASELINE

    def test_the_justification_gives_the_numbers_and_the_caveat_when_evidence_is_thin(self) -> None:
        thin = choose([baseline_row(), row("a", cost="0.05", drop=Fraction(1, 100), independent=9)])
        assert "accuracy 0.890 against 0.900 (-1.0 points" in thin.why
        assert "cost per document $0.0500 against $0.1000 (-50%)" in thin.why
        assert "latency p50 50.0 s against 100.0 s (-50%)" in thin.why
        assert "Thin evidence: only 9 independent documents" in thin.why
        solid = choose([baseline_row(), row("a", cost="0.05", independent=10)])
        assert "Thin evidence" not in solid.why


class TestReportCaveat:
    def report(self, independent: int) -> str:
        rows = [baseline_row(), row("a", cost="0.05", independent=independent)]
        return to_markdown(rows, choose(rows), sweep_id="t", tolerance=DEFAULT_TOLERANCE)

    def test_under_ten_independent_documents_the_report_carries_the_caveat(self) -> None:
        assert "Caveat: the comparisons rest on as few as 9 independent documents" in self.report(9)

    def test_at_ten_there_is_none(self) -> None:
        assert "Caveat" not in self.report(10)

    def test_the_caveat_names_the_smallest_sample(self) -> None:
        rows = [baseline_row(), row("a", independent=12), row("b", independent=4)]
        text = to_markdown(rows, choose(rows), sweep_id="t", tolerance=DEFAULT_TOLERANCE)
        assert "as few as 4 independent documents" in text

    def test_a_sweep_of_only_the_baseline_has_nothing_to_caveat(self) -> None:
        rows = [baseline_row()]
        text = to_markdown(rows, choose(rows), sweep_id="t", tolerance=DEFAULT_TOLERANCE)
        assert "Caveat" not in text and "Chosen: **baseline**" in text


# ------------------------------------------------------------------ built from real run folders


def timed(doc_id: str, total: str, *, cost: str, latency: float) -> DocRecord:
    record = ok(doc_id, total=extracted(Decimal(total)))
    return record.model_copy(
        update={
            "cost_usd": Decimal(cost),
            "latency_ms": latency,
            "usage": {"input_tokens": 1000, "billable_output_tokens": 400},
        }
    )


class World:
    """Three documents; the baseline gets all right at $0.10 and 100 s. `cheap` is as accurate at
    half the cost and half the time, `lossy` gets one wrong, `broken` loses a document."""

    def __init__(self, tmp_path: Path) -> None:
        rows = [manifest_row(f"doc_{i:03d}") for i in range(3)]
        labels = [label(r.doc_id, total=Decimal("10")) for r in rows]
        self.manifest, self.labels, self.pages = dataset(tmp_path, rows, labels)
        self.tmp = tmp_path
        self.ids = [r.doc_id for r in rows]

    def run(
        self, name: str, *, cost: str, latency: float, wrong: int = 0, missing: int = 0
    ) -> Path:
        records: list[DocRecord] = []
        for index, doc_id in enumerate(self.ids):
            if index >= len(self.ids) - missing:
                records.append(DocRecord(doc_id=doc_id, status="failed", error="HTTP 500"))
                continue
            total = "99" if index < wrong else "10"
            records.append(timed(doc_id, total, cost=cost, latency=latency))
        return run_dir(self.tmp, records, name=name)

    def sweep(self, baseline: Path, points: list[SweepPoint]) -> list[SweepRow]:
        return build_sweep(
            baseline,
            points,
            manifest_path=self.manifest,
            labels_dir=self.labels,
            pages_dir=self.pages,
            resamples=500,
        )


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


class TestBuildSweep:
    def points(self, world: World) -> tuple[Path, list[SweepPoint]]:
        baseline = world.run("base", cost="0.10", latency=100_000.0)
        return baseline, [
            SweepPoint(
                "cheap", world.run("cheap", cost="0.05", latency=50_000.0), "effort", {"e": "low"}
            ),
            SweepPoint(
                "lossy", world.run("lossy", cost="0.03", latency=30_000.0, wrong=1), "effort"
            ),
            SweepPoint(
                "broken", world.run("broken", cost="0.02", latency=20_000.0, missing=1), "effort"
            ),
        ]

    def test_each_row_carries_its_runs_numbers_and_the_comparison(self, world: World) -> None:
        baseline, points = self.points(world)
        rows = world.sweep(baseline, points)
        assert [r.label for r in rows] == [BASELINE, "cheap", "lossy", "broken"]
        base, cheap, lossy, broken = rows
        assert (base.documents, base.failed, base.accuracy) == (3, 0, 1.0)
        assert base.drop == 0 and base.p_worse == 1 and base.verdict == BASELINE
        assert base.list_cost_per_doc == Decimal("0.1") and base.latency_p50_ms == 100_000.0
        assert (cheap.accuracy, cheap.drop, cheap.verdict) == (1.0, 0, "ok")
        assert cheap.list_cost_per_doc == Decimal("0.05") and cheap.latency_p50_ms == 50_000.0
        assert (cheap.input_tokens, cheap.output_tokens) == (3000, 1200)
        assert cheap.setting == {"e": "low"} and cheap.knob == "effort"
        assert lossy.drop == Fraction(1, 3) and lossy.accuracy == pytest.approx(2 / 3)
        assert broken.failed == 1 and broken.documents == 3

    def test_the_comparison_is_the_compare_modules(self, world: World) -> None:
        baseline, points = self.points(world)
        lossy = world.sweep(baseline, points)[2]
        direct = compare_runs(
            baseline,
            points[1].run_dir,
            manifest_path=world.manifest,
            labels_dir=world.labels,
            pages_dir=world.pages,
            seed=1,
            resamples=500,
        )
        assert lossy.p_worse == direct.mcnemar.p_worse
        assert (lossy.interval_low, lossy.interval_high) == (
            direct.bootstrap.low,
            direct.bootstrap.high,
        )
        assert lossy.independent == direct.bootstrap.clusters == 3

    def test_the_thresholds_reach_the_verdict(self, world: World) -> None:
        baseline, points = self.points(world)
        rows = build_sweep(
            baseline,
            points[1:2],
            manifest_path=world.manifest,
            labels_dir=world.labels,
            pages_dir=world.pages,
            thresholds=Thresholds(alpha=0.9, max_drop=0.0),
            resamples=200,
        )
        assert rows[1].verdict == "regression"  # 1 regression, p = 1/2 <= 0.9, drop > 0

    def test_the_rule_picks_the_accurate_cheap_run_not_the_cheapest_broken_ones(
        self, world: World
    ) -> None:
        baseline, points = self.points(world)
        rows = world.sweep(baseline, points)
        choice = choose(rows)
        assert choice.label == "cheap"
        assert "33.3 points" in choice.reasons["lossy"]
        assert choice.reasons["broken"] == "1 document(s) failed or were refused"

    def test_reports_are_byte_identical_twice(self, world: World) -> None:
        baseline, points = self.points(world)
        first, second = world.sweep(baseline, points), world.sweep(baseline, points)
        choice = choose(first)
        assert to_json(first, choice, tolerance=DEFAULT_TOLERANCE) == to_json(
            second, choose(second), tolerance=DEFAULT_TOLERANCE
        )
        a = to_markdown(first, choice, sweep_id="t", tolerance=DEFAULT_TOLERANCE)
        assert a == to_markdown(second, choose(second), sweep_id="t", tolerance=DEFAULT_TOLERANCE)

    def test_the_markdown_has_the_table_the_choice_and_the_caveat(self, world: World) -> None:
        baseline, points = self.points(world)
        rows = world.sweep(baseline, points)
        text = to_markdown(rows, choose(rows), sweep_id="t", tolerance=DEFAULT_TOLERANCE)
        assert text.startswith("# Sweep `t`: one setting at a time against the baseline")
        assert "Chosen: **cheap**." in text
        assert "| baseline | baseline | 3 | 0 | 1.000 | - | - | - |" in text
        assert "| cheap | e=low | 3 | 0 | 1.000 | +0.0 pts |" in text
        assert "| lossy |" in text and "-33.3 pts" in text
        assert "- broken: 1 document(s) failed or were refused" in text
        assert "Caveat: the comparisons rest on as few as 3 independent documents" in text

    def test_a_zero_change_is_plain_zero_never_negative_zero(self, world: World) -> None:
        baseline, points = self.points(world)
        rows = world.sweep(baseline, points)
        choice = choose(rows)
        text = to_markdown(rows, choice, sweep_id="t", tolerance=DEFAULT_TOLERANCE)
        assert "-0.0" not in text and "-0.0" not in choice.why
        assert "-0.0" not in to_json(rows, choice, tolerance=DEFAULT_TOLERANCE)

    def test_the_json_has_every_row_and_the_choice(self, world: World) -> None:
        baseline, points = self.points(world)
        rows = world.sweep(baseline, points)
        data = json.loads(to_json(rows, choose(rows), tolerance=DEFAULT_TOLERANCE))
        assert list(data) == sorted(data)
        assert data["chosen"] == "cheap" and data["tolerance"] == 0.02
        assert [r["label"] for r in data["rows"]] == [BASELINE, "cheap", "lossy", "broken"]
        assert data["rows"][1]["list_cost_per_doc"] == "0.050000"
        assert data["rows"][2]["accuracy_delta"] == pytest.approx(-1 / 3)
        assert data["rows"][3]["failed"] == 1


class TestBeforeAfter:
    def test_the_section_numbers_come_from_both_runs_and_their_comparison(
        self, world: World
    ) -> None:
        base = world.run("base", cost="0.10", latency=100_000.0)
        after = world.run("after", cost="0.05", latency=50_000.0, wrong=1)

        def card(path: Path):  # noqa: ANN202
            return score_run(
                path,
                manifest_path=world.manifest,
                labels_dir=world.labels,
                pages_dir=world.pages,
                git_sha="x",
                git_dirty=False,
            )

        comparison = compare_runs(
            base,
            after,
            manifest_path=world.manifest,
            labels_dir=world.labels,
            pages_dir=world.pages,
            resamples=200,
        )
        data = before_after(card(base), card(after), comparison)
        assert (data["baseline"], data["candidate"]) == ("base", "after")
        assert (data["documents_paired"], data["fields_paired"]) == (3, 3)
        metrics = data["metrics"]
        assert metrics["accuracy"]["before"] == 1.0
        assert metrics["accuracy"]["after"] == pytest.approx(2 / 3)
        assert metrics["accuracy"]["delta"] == pytest.approx(-1 / 3)
        assert metrics["list_cost_per_doc_usd"] == {"before": 0.1, "after": 0.05, "delta": -0.05}
        assert metrics["latency_p50_ms"]["delta"] == -50_000.0
        assert metrics["input_tokens_per_doc"]["before"] == 1000.0
        assert data["mcnemar"] == {"regressions": 1, "improvements": 0, "p_worse": 0.5}
        assert data["verdict"] in ("ok", "regression")
        assert data["bootstrap"]["documents"] == 3
        json.dumps(data)  # plain data: it goes straight into scorecard.json
