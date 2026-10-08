"""Calibration on the hand-built world of test_failures (see the table there) and on literal point
lists whose bins, ECE, Brier and risk-coverage are worked out by hand.

The 8 fields of the world, as the threshold sees them:
  withheld (stay abstained at every threshold): doc_a total and doc_b vendor_name, total (HARD
      code), doc_b subtotal (the model abstained itself)                                -> 4
  answered (always returned): doc_a tax (absent, confidence 0.9, right)                  -> 1
  governed: doc_a vendor_name 0.95 right, subtotal 0.5 right (its raw 0.8), document_number
      0.85 wrong                                                                         -> 3
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from evalkit.calibration import (
    TARGET_SELECTIVE,
    Operating,
    Point,
    analyse,
    aurc,
    brier,
    choose_threshold,
    classify,
    curve,
    ece,
    operating_point,
    pool,
    reliability,
    reliability_svg,
    risk_coverage_svg,
    to_markdown,
)
from evalkit.failures import RunCases, collect
from test_failures import build


def world(tmp_path: Path) -> RunCases:
    return collect(
        build(tmp_path),
        manifest_path=tmp_path / "data" / "manifest.csv",
        labels_dir=tmp_path / "ground_truth",
    )


class TestPool:
    def test_every_field_is_withheld_answered_or_governed(self, tmp_path: Path) -> None:
        result = world(tmp_path)
        kinds = {(c.doc_id, c.name): classify(c) for c in result.cases}
        assert kinds == {
            ("doc_a", "vendor_name"): "governed",
            ("doc_a", "total"): "withheld",
            ("doc_a", "subtotal"): "governed",
            ("doc_a", "tax"): "answered",
            ("doc_a", "document_number"): "governed",
            ("doc_b", "vendor_name"): "withheld",
            ("doc_b", "total"): "withheld",
            ("doc_b", "subtotal"): "withheld",
        }

    def test_a_threshold_only_abstention_is_governed_but_a_hard_one_never_is(
        self, tmp_path: Path
    ) -> None:
        by = {(c.doc_id, c.name): classify(c) for c in world(tmp_path).cases}
        assert by[("doc_a", "subtotal")] == "governed"  # abstained for confidence_below_threshold
        assert by[("doc_a", "total")] == "withheld"  # abstained for evidence_not_grounded

    def test_the_pool_counts_and_correctness_of_the_models_own_value(self, tmp_path: Path) -> None:
        data = pool([world(tmp_path)])
        assert (data.labeled, data.withheld, len(data.answered), len(data.governed)) == (8, 4, 1, 3)
        assert sorted((p.confidence, p.correct) for p in data.governed) == [
            (0.5, True),  # subtotal: lifted, the model's value 9 was right
            (0.85, False),
            (0.95, True),
        ]
        assert sorted((p.confidence, p.correct) for p in data.governed_raw) == [
            (0.8, True),
            (0.85, False),
            (0.95, True),
        ]
        assert (data.runs, data.documents) == (1, 2)

    def test_repeated_runs_add_points_not_documents(self, tmp_path: Path) -> None:
        one = world(tmp_path)
        data = pool([one, one])
        assert (data.runs, data.documents, data.labeled) == (2, 2, 16)


class TestOperatingPoints:
    # t=0.3: all 3 governed kept -> returned 4, right 3 (tax, vendor_name, subtotal); t=0.6 drops
    # subtotal (0.5); t=0.9 drops document_number (0.85) as well
    @pytest.mark.parametrize(
        ("threshold", "answered", "correct"),
        [(0.3, 4, 3), (0.5, 4, 3), (0.6, 3, 2), (0.85, 3, 2), (0.9, 2, 2), (0.96, 1, 1)],
    )
    def test_literal_points(
        self, tmp_path: Path, threshold: float, answered: int, correct: int
    ) -> None:
        point = operating_point(pool([world(tmp_path)]), threshold)
        assert (point.answered, point.correct, point.labeled) == (answered, correct, 8)

    def test_headline_counts_a_withheld_field_as_an_error_selective_does_not(
        self, tmp_path: Path
    ) -> None:
        point = operating_point(pool([world(tmp_path)]), 0.3)
        assert point.accuracy == 3 / 8 and point.selective == 3 / 4 and point.coverage == 4 / 8

    def test_the_threshold_is_inclusive(self, tmp_path: Path) -> None:
        data = pool([world(tmp_path)])
        assert (
            operating_point(data, 0.85).answered == 3 and operating_point(data, 0.86).answered == 2
        )

    def test_the_curve_has_a_point_per_distinct_governed_confidence_and_zero(
        self, tmp_path: Path
    ) -> None:
        points = curve(pool([world(tmp_path)]))
        assert [p.threshold for p in points] == [0.0, 0.5, 0.85, 0.95]
        assert [p.coverage for p in points] == [4 / 8, 4 / 8, 3 / 8, 2 / 8]  # never rises

    def test_nothing_answered_has_no_selective_accuracy(self) -> None:
        data = pool([])
        assert operating_point(data, 0.5).answered == 0


class TestThresholdRule:
    def points(self, tmp_path: Path):  # noqa: ANN202
        return curve(pool([world(tmp_path)]))

    def test_the_lowest_threshold_that_reaches_the_target(self, tmp_path: Path) -> None:
        # selective: 0.75 (t=0, 0.5), 0.667 (0.85), 1.0 (0.95)
        chosen = choose_threshold(self.points(tmp_path), 0.95)
        assert chosen is not None and chosen.point.threshold == 0.95 and chosen.target == 0.95

    def test_a_lower_target_is_reached_sooner(self, tmp_path: Path) -> None:
        chosen = choose_threshold(self.points(tmp_path), 0.7)
        assert chosen is not None and chosen.point.threshold == 0.0

    def test_no_threshold_may_reach_it(self, tmp_path: Path) -> None:
        assert choose_threshold(self.points(tmp_path), 1.01) is None

    def test_the_documented_target(self) -> None:
        assert TARGET_SELECTIVE == 0.95

    def test_the_input_order_does_not_matter(self, tmp_path: Path) -> None:
        points = self.points(tmp_path)
        assert choose_threshold(list(reversed(points)), 0.95) == choose_threshold(points, 0.95)


class TestTargetBoundary:
    def test_a_selective_accuracy_exactly_at_the_target_qualifies(self) -> None:
        exactly = Operating(threshold=0.5, answered=4, correct=3, labeled=8)  # 0.75
        chosen = choose_threshold([exactly], 0.75)
        assert chosen is not None and chosen.point.threshold == 0.5
        assert choose_threshold([exactly], 0.76) is None


POINTS = [Point(0.95, True), Point(0.55, True), Point(0.5, False), Point(1.0, True)]


class TestReliability:
    def test_bins_hold_mean_confidence_and_accuracy(self) -> None:
        bins = reliability(POINTS)
        assert [(b.low, b.high, b.count) for b in bins] == [(0.5, 0.7, 2), (0.95, 1.0, 2)]
        assert bins[0].mean_confidence == pytest.approx(0.525) and bins[0].accuracy == 0.5
        assert bins[1].mean_confidence == pytest.approx(0.975) and bins[1].accuracy == 1.0

    def test_the_lower_edge_is_inclusive_and_one_belongs_to_the_last_bin(self) -> None:
        bins = reliability([Point(0.5, True), Point(1.0, True), Point(0.45, False)])
        assert [(b.low, b.count) for b in bins] == [(0.0, 1), (0.5, 1), (0.95, 1)]

    def test_ece_is_the_count_weighted_gap(self) -> None:
        # (2 * |0.5 - 0.525| + 2 * |1.0 - 0.975|) / 4
        assert ece(reliability(POINTS)) == pytest.approx(0.025)

    def test_ece_weights_each_bin_by_its_size(self) -> None:
        # three points at 0.6 with one right (gap 0.2667), one at 0.96 right (gap 0.04)
        points = [Point(0.6, True), Point(0.6, False), Point(0.6, False), Point(0.96, True)]
        assert ece(reliability(points)) == pytest.approx((3 * (0.6 - 1 / 3) + 0.04) / 4)

    def test_brier_is_the_mean_squared_error_of_the_confidence(self) -> None:
        # (0.05^2 + 0.45^2 + 0.5^2 + 0) / 4
        assert brier(POINTS) == pytest.approx(0.11375)

    def test_nothing_in_nothing_out(self) -> None:
        assert reliability([]) == [] and ece([]) is None and brier([]) is None


class TestAurc:
    def test_it_is_the_mean_selective_risk_in_decreasing_confidence(self) -> None:
        # risk after each answer: 0, 0, 0, 1/4 -> (0 + 0 + 0 + 0.25) / 4
        assert aurc(POINTS) == pytest.approx(0.0625)

    def test_ties_are_answered_together(self) -> None:
        assert aurc([Point(0.9, True), Point(0.9, False)]) == pytest.approx(0.5)

    def test_a_confidence_that_ranks_wrongly_is_worse_than_no_ranking(self) -> None:
        right = aurc([Point(0.9, True), Point(0.5, False)])
        wrong = aurc([Point(0.9, False), Point(0.5, True)])
        assert right is not None and wrong is not None and right < wrong

    def test_no_points_no_area(self) -> None:
        assert aurc([]) is None


class TestReport:
    def report(self, tmp_path: Path):  # noqa: ANN202
        return analyse([world(tmp_path)], abstain_below=0.3)

    def test_the_report_holds_the_rule_the_threshold_in_force_and_the_choice(
        self, tmp_path: Path
    ) -> None:
        report = self.report(tmp_path)
        assert report.in_force_threshold == 0.3 and report.in_force.answered == 4
        assert report.chosen is not None and report.chosen.point.threshold == 0.95

    def test_markdown_states_the_independent_sample_and_what_a_threshold_can_move(
        self, tmp_path: Path
    ) -> None:
        text = to_markdown(self.report(tmp_path), run_ids=["r1"])
        assert "so the independent sample is 2 documents, not 8 fields" in text
        assert "4 instances are **fixed abstentions**" in text
        assert "1 are `absent` fields" in text and "3 are **governed**" in text

    def test_markdown_shows_in_force_and_chosen_side_by_side(self, tmp_path: Path) -> None:
        text = to_markdown(self.report(tmp_path), run_ids=["r1"])
        assert "| in force (`ABSTAIN_BELOW`) | 0.30 | 0.375 | 0.750 | 0.500 |" in text
        assert "| chosen by the rule | 0.95 | 0.250 | 1.000 | 0.250 |" in text

    def test_when_no_threshold_reaches_the_target_it_says_so_and_keeps_the_current_one(
        self, tmp_path: Path
    ) -> None:
        import dataclasses

        report = dataclasses.replace(self.report(tmp_path), chosen=None)
        text = to_markdown(report, run_ids=["r1"])
        assert "| chosen by the rule | none | n/a | n/a | n/a |" in text
        assert "the current `ABSTAIN_BELOW` stays" in text

    def test_the_absent_fields_are_left_out_of_the_reliability_table(self, tmp_path: Path) -> None:
        report = self.report(tmp_path)
        assert sum(b.count for b in report.final_bins) == 3  # governed only, not 4
        assert "(1 of 1 right) are left out" in to_markdown(report, run_ids=["r1"])

    def test_same_input_same_bytes_and_notes_are_a_reading_section(self, tmp_path: Path) -> None:
        a = to_markdown(analyse([world(tmp_path / "a")], abstain_below=0.3), run_ids=["r1"])
        b = to_markdown(analyse([world(tmp_path / "b")], abstain_below=0.3), run_ids=["r1"])
        assert a == b
        noted = to_markdown(
            analyse([world(tmp_path / "c")], abstain_below=0.3), run_ids=["r1"], notes=" Why. "
        )
        assert noted.endswith("## Reading\n\nWhy.\n")


class TestSvg:
    def report(self, tmp_path: Path):  # noqa: ANN202
        return analyse([world(tmp_path)], abstain_below=0.3)

    def test_both_diagrams_are_well_formed_xml(self, tmp_path: Path) -> None:
        report = self.report(tmp_path)
        for text in (reliability_svg(report), risk_coverage_svg(report)):
            root = ET.fromstring(text)
            assert root.tag.endswith("svg")

    def test_the_reliability_diagram_has_a_point_per_bin_and_a_diagonal(
        self, tmp_path: Path
    ) -> None:
        report = self.report(tmp_path)
        text = reliability_svg(report)
        bins = len(report.final_bins) + len(report.raw_bins)
        assert text.count("<circle") == bins + 2  # two legend dots
        assert "stroke-dasharray" in text

    def test_the_chosen_marker_exists_only_when_a_threshold_was_chosen(
        self, tmp_path: Path
    ) -> None:
        import dataclasses

        report = self.report(tmp_path)
        assert ">chosen<" in risk_coverage_svg(report) and ">in force<" in risk_coverage_svg(report)
        assert ">chosen<" not in risk_coverage_svg(dataclasses.replace(report, chosen=None))

    def test_the_risk_coverage_axis_is_zoomed_and_says_where_it_starts(
        self, tmp_path: Path
    ) -> None:
        # the lowest selective accuracy of the world is 2/3, so the axis starts at 0.60
        assert "axis starts at 0.60" in risk_coverage_svg(self.report(tmp_path))

    def test_every_point_and_the_target_line_lie_inside_the_plot(self, tmp_path: Path) -> None:
        import dataclasses

        report = self.report(tmp_path)
        everything_above_the_target = dataclasses.replace(
            report, curve=[Operating(threshold=0.0, answered=10, correct=10, labeled=10)]
        )
        for variant in (report, everything_above_the_target):
            root = ET.fromstring(risk_coverage_svg(variant))
            ns = "{http://www.w3.org/2000/svg}"
            ys = [float(c.get("cy")) for c in root.iter(ns + "circle")]
            ys += [
                float(t.get("y")) for t in root.iter(ns + "text") if t.text and "target" in t.text
            ]
            assert ys and all(36.0 <= y <= 348.0 for y in ys), ys

    def test_the_diagrams_are_deterministic(self, tmp_path: Path) -> None:
        one = analyse([world(tmp_path / "a")], abstain_below=0.3)
        two = analyse([world(tmp_path / "b")], abstain_below=0.3)
        assert reliability_svg(one) == reliability_svg(two)
        assert risk_coverage_svg(one) == risk_coverage_svg(two)
