"""Run-to-run noise on hand-built repeats of one setting.

Three runs of one document with four labeled fields (vendor_name Acme, total 10, subtotal 9,
tax absent). What each run returns:

  field        run1       run2       run3
  vendor_name  Acme       Acme       Acme        right  right  right
  total        10         abstained  10          right  no     right
  subtotal     9          9          wrong 8     right  right  no
  tax          absent     absent     absent      right  right  right

so accuracy is 4/4, 3/4, 3/4; pairs (1,2) and (1,3) each have one field right only in run 1, and
(2,3) has one right only in each (2 flips). Unstable outcomes: total and subtotal.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from docextract.extract.runner import DocRecord
from docextract.extract.schema import Field
from evalkit.noise import NoiseError, NoiseReport, measure, to_markdown
from test_compare import manifest_row
from test_failures import abst, ext, none
from test_scorecard import dataset, document, label, run_dir

PRICE_IN, PRICE_OUT = Decimal("1"), Decimal("2")


def record(total: Field, subtotal: Field, *, tokens: int, latency: float) -> DocRecord:
    returned = document(vendor_name=ext("Acme"), total=total, subtotal=subtotal, tax=none())
    return DocRecord(
        doc_id="doc_a",
        status="ok",
        parsed=returned,
        validated=returned,
        latency_ms=latency,
        usage={"input_tokens": 1000, "billable_output_tokens": tokens},
    )


def build(tmp_path: Path, *, drop_subtotal_label: bool = False) -> list[Path]:
    labels = [
        label(
            "doc_a",
            vendor_name="Acme",
            total=Decimal("10"),
            subtotal=None if drop_subtotal_label else Decimal("9"),
            tax=None,
        )
    ]
    dataset(tmp_path, [manifest_row("doc_a")], labels)
    runs = [
        record(ext(Decimal("10")), ext(Decimal("9")), tokens=100, latency=10_000.0),
        record(abst(0.5, "evidence_not_grounded"), ext(Decimal("9")), tokens=150, latency=30_000.0),
        record(ext(Decimal("10")), ext(Decimal("8")), tokens=200, latency=20_000.0),
    ]
    return [run_dir(tmp_path, [one], name=f"run{i + 1}") for i, one in enumerate(runs)]


def report(tmp_path: Path) -> NoiseReport:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return measure(
        build(tmp_path),
        manifest_path=tmp_path / "data" / "manifest.csv",
        labels_dir=tmp_path / "ground_truth",
    )


class TestMeasure:
    def test_each_runs_headline(self, tmp_path: Path) -> None:
        one, two, three = report(tmp_path).runs
        assert (one.correct, one.labeled, one.answered) == (4, 4, 4)
        assert (two.correct, two.answered) == (3, 3)  # total was withheld
        assert (three.correct, three.answered) == (3, 4)
        assert one.accuracy == 1.0 and two.selective == 1.0 and three.coverage == 1.0

    def test_pairwise_flips_count_each_direction(self, tmp_path: Path) -> None:
        pairs = {(p.first, p.second): (p.first_only, p.second_only) for p in report(tmp_path).pairs}
        assert pairs == {
            ("run1", "run2"): (1, 0),
            ("run1", "run3"): (1, 0),
            ("run2", "run3"): (1, 1),
        }
        assert [p.flips for p in report(tmp_path / "again").pairs] == [1, 1, 2]

    def test_unstable_fields_are_those_whose_outcome_differs_anywhere(self, tmp_path: Path) -> None:
        result = report(tmp_path)
        assert [(u.name, u.outcomes) for u in result.unstable] == [
            ("subtotal", ("correct", "correct", "value_wrong")),
            ("total", ("correct", "abstained", "correct")),
        ]
        assert result.correctness_changed == 2

    def test_per_document_spread_of_tokens_and_latency(self, tmp_path: Path) -> None:
        (spread,) = report(tmp_path).docs
        assert spread.tokens == (100, 150, 200)
        assert spread.latency_ms == (10_000.0, 30_000.0, 20_000.0)

    def test_dollars_are_recomputed_at_one_price(self, tmp_path: Path) -> None:
        one, two, three = report(tmp_path).runs
        # 1000 input at $1 per million and 100/150/200 output at $2 per million
        assert one.cost_at(PRICE_IN, PRICE_OUT) == Decimal("0.0012")
        assert three.cost_at(PRICE_IN, PRICE_OUT) == Decimal("0.0014")
        assert two.cost_at(Decimal(0), Decimal(0)) == 0


class TestUnstableButNeverRight:
    def custom(
        self, tmp_path: Path, totals: list[Field], *, cached: int | None = None
    ) -> list[Path]:
        dataset(tmp_path, [manifest_row("doc_a")], [label("doc_a", total=Decimal("10"))])
        runs = []
        for index, total in enumerate(totals):
            one = DocRecord(
                doc_id="doc_a",
                status="ok",
                parsed=document(total=total),
                validated=document(total=total),
                latency_ms=1000.0 * (index + 1),
                cache_hit=index == cached,
                usage={"input_tokens": 10, "billable_output_tokens": 5},
            )
            runs.append(run_dir(tmp_path, [one], name=f"c{index}"))
        return runs

    def measure(self, tmp_path: Path, runs: list[Path]) -> NoiseReport:
        return measure(
            runs,
            manifest_path=tmp_path / "data" / "manifest.csv",
            labels_dir=tmp_path / "ground_truth",
        )

    def test_wrong_then_withheld_then_wrong_is_unstable_but_its_correctness_never_changes(
        self, tmp_path: Path
    ) -> None:
        wrong, withheld = ext(Decimal("8")), abst(0.5, "evidence_not_grounded")
        result = self.measure(tmp_path, self.custom(tmp_path, [wrong, withheld, wrong]))
        assert len(result.unstable) == 1 and result.correctness_changed == 0
        assert all(pair.flips == 0 for pair in result.pairs)

    def test_a_cached_answer_has_no_latency_to_report(self, tmp_path: Path) -> None:
        ok = ext(Decimal("10"))
        result = self.measure(tmp_path, self.custom(tmp_path, [ok, ok, ok], cached=1))
        (spread,) = result.docs
        assert spread.latency_ms == (1000.0, 3000.0)  # the cached run (2000 ms) is not live
        assert spread.tokens == (5, 5, 5)


class TestRefusals:
    def test_one_run_is_not_a_noise_measurement(self, tmp_path: Path) -> None:
        runs = build(tmp_path)
        with pytest.raises(NoiseError, match="at least two"):
            measure(
                runs[:1],
                manifest_path=tmp_path / "data" / "manifest.csv",
                labels_dir=tmp_path / "ground_truth",
            )

    def test_a_run_that_lost_a_document_is_not_a_repeat_of_the_others(self, tmp_path: Path) -> None:
        runs = build(tmp_path)
        failed = run_dir(
            tmp_path, [DocRecord(doc_id="doc_a", status="failed", error="HTTP 500")], name="lost"
        )
        with pytest.raises(NoiseError, match="does not cover the same labeled fields"):
            measure(
                [runs[0], failed],
                manifest_path=tmp_path / "data" / "manifest.csv",
                labels_dir=tmp_path / "ground_truth",
            )


class TestMarkdown:
    def render(self, tmp_path: Path, notes: str | None = None) -> str:
        tmp_path.mkdir(parents=True, exist_ok=True)
        return to_markdown(report(tmp_path), price_in=PRICE_IN, price_out=PRICE_OUT, notes=notes)

    def test_it_states_the_independent_sample_and_the_price(self, tmp_path: Path) -> None:
        text = self.render(tmp_path)
        assert "so the independent sample is 1 documents, not the number of runs" in text
        assert "recomputed from tokens at $1/$2 per million" in text

    def test_ranges_and_flip_rates(self, tmp_path: Path) -> None:
        text = self.render(tmp_path)
        assert "Accuracy ranges from 0.750 to 1.000 (25.0 points)" in text
        assert "output tokens from 100 to 200 (100% more)" in text
        assert "Between two runs 1 to 2 of 4 fields flip (25.0% to 50.0%)" in text
        assert "2 fields are right in some runs and not in others" in text

    def test_the_p95_remark_names_the_slowest_call_when_that_is_what_p95_is(
        self, tmp_path: Path
    ) -> None:
        assert "it is the single slowest call, so one document decides it" in self.render(tmp_path)

    def test_the_same_runs_give_the_same_bytes(self, tmp_path: Path) -> None:
        assert self.render(tmp_path / "a") == self.render(tmp_path / "b")

    def test_notes_become_a_reading_section(self, tmp_path: Path) -> None:
        assert self.render(tmp_path, notes=" Why. ").endswith("## Reading\n\nWhy.\n")
