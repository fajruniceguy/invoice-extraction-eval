from decimal import Decimal
from typing import Any

import pytest

from docextract.extract.runner import DocRecord, DocStatus
from evalkit.perf import PerfStats, percentile


def record(
    status: DocStatus = "ok",
    *,
    cost: str = "0",
    avoided: str = "0",
    latency: float = 1000.0,
    cache_hit: bool = False,
    tokens: tuple[int, int] | None = (100, 40),
    repairs: int = 0,
) -> DocRecord:
    usage: dict[str, Any] | None = None
    if tokens is not None:
        usage = {"input_tokens": tokens[0], "billable_output_tokens": tokens[1]}
    return DocRecord(
        doc_id="d",
        status=status,
        cost_usd=Decimal(cost),
        avoided_cost_usd=Decimal(avoided),
        latency_ms=latency,
        cache_hit=cache_hit,
        usage=usage,
        repairs=repairs,
    )


class TestPercentile:
    # nearest-rank: the smallest value with at least q% of the values at or below it
    @pytest.mark.parametrize(
        ("values", "p50", "p95"),
        [
            ([7.0], 7.0, 7.0),
            ([1.0, 2.0], 1.0, 2.0),
            ([1.0, 2.0, 3.0], 2.0, 3.0),
            ([1.0, 2.0, 3.0, 4.0], 2.0, 4.0),
            ([float(i) for i in range(1, 21)], 10.0, 19.0),
            ([float(i) for i in range(1, 101)], 50.0, 95.0),
        ],
    )
    def test_literal_values(self, values: list[float], p50: float, p95: float) -> None:
        assert percentile(values, 50) == p50
        assert percentile(values, 95) == p95

    def test_the_input_need_not_be_sorted(self) -> None:
        assert percentile([30.0, 10.0, 20.0], 50) == 20.0

    def test_it_is_always_a_value_that_happened(self) -> None:
        assert percentile([10.0, 20.0], 50) == 10.0  # not the interpolated 15

    def test_no_values_no_percentile(self) -> None:
        assert percentile([], 50) is None

    def test_100_is_the_maximum_and_1_the_minimum_of_a_long_list(self) -> None:
        values = [float(i) for i in range(1, 51)]
        assert percentile(values, 100) == 50.0 and percentile(values, 1) == 1.0

    @pytest.mark.parametrize("q", [0, 101, -5])
    def test_q_must_be_between_1_and_100(self, q: int) -> None:
        with pytest.raises(ValueError, match="between 1 and 100"):
            percentile([1.0], q)


class TestPerfStats:
    def test_nothing_attempted_is_all_zero_and_unknown(self) -> None:
        stats = PerfStats.from_records([])
        assert stats == PerfStats()
        assert stats.cost_per_doc is None and stats.list_cost_per_doc is None
        assert stats.latency_p50_ms is None

    def test_only_documents_that_were_attempted_count(self) -> None:
        stats = PerfStats.from_records(
            [
                record("ok", cost="0.10"),
                record("failed", cost="0.20", tokens=(50, 10)),
                record("refused", cost="9"),  # no call was made
                record("not_run", cost="9"),
            ]
        )
        assert (stats.documents, stats.ok, stats.failed) == (2, 1, 1)
        assert stats.cost_usd == Decimal("0.30")  # a failed call can still be billed
        assert (stats.input_tokens, stats.output_tokens) == (150, 50)

    def test_cost_is_shown_paid_and_at_list_price(self) -> None:
        stats = PerfStats.from_records(
            [
                record(cost="0.10"),
                record(cost="0", avoided="0.30", cache_hit=True, latency=0.0),
            ]
        )
        assert stats.cost_usd == Decimal("0.10")
        assert stats.list_cost_usd == Decimal("0.40")
        assert stats.cost_per_doc == Decimal("0.05")
        assert stats.list_cost_per_doc == Decimal("0.2")

    def test_cost_per_document_is_rounded_half_up_to_six_places(self) -> None:
        stats = PerfStats.from_records([record(cost="0.1"), record(), record()])
        assert stats.cost_per_doc == Decimal("0.033333")
        half = PerfStats.from_records([record(cost="0.0000025"), record(cost="0")])
        assert half.cost_per_doc == Decimal("0.000001")  # 0.00000125 -> 0.000001

    def test_latency_is_over_live_calls_only(self) -> None:
        stats = PerfStats.from_records(
            [
                record(latency=100.0),
                record(latency=300.0),
                record(latency=0.0, cache_hit=True),  # a replay: 0 ms, not a latency
                record("failed", latency=99999.0),  # failed: not a completed live call
            ]
        )
        assert (stats.live_calls, stats.cached_documents) == (2, 1)
        assert (stats.latency_p50_ms, stats.latency_p95_ms) == (100.0, 300.0)
        assert stats.latency_mean_ms == 200.0

    def test_with_no_live_call_there_is_no_latency(self) -> None:
        stats = PerfStats.from_records([record(latency=0.0, cache_hit=True)])
        assert stats.live_calls == 0
        assert stats.latency_p50_ms is None and stats.latency_p95_ms is None
        assert stats.latency_mean_ms is None

    def test_a_record_without_usage_adds_no_tokens(self) -> None:
        stats = PerfStats.from_records([record(tokens=None), record(tokens=(10, 5))])
        assert (stats.input_tokens, stats.output_tokens) == (10, 5)

    def test_repairs_are_summed(self) -> None:
        stats = PerfStats.from_records([record(repairs=1), record(repairs=2), record()])
        assert stats.repairs == 3
