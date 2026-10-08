"""Cost and latency of a run, from the per-document records.

Which documents count: every one that was attempted (status ok or failed), labeled or not, because
a failed call can still have been billed. Refused and not_run documents made no call.

  * Cost is shown twice: what was PAID (a cache replay costs 0) and what it would cost at LIST
    PRICE (paid + avoided), so a run answered from the cache still says what it would cost.
  * Latency is over LIVE calls only (status ok, not answered from the cache). A replay takes 0 ms
    and would drag p50 down to nothing. With no live call there is no latency (`None`, shown n/a).
  * Percentiles are nearest-rank: the smallest value with at least q% of the values at or below it
    (no interpolation, so a percentile is always a latency that really happened). With 3 values p50
    is the middle one and p95 the largest.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from docextract.extract.runner import DocRecord

_PER_DOC_PLACES = Decimal("0.000001")


def percentile(values: Sequence[float], q: int) -> float | None:
    """Nearest-rank percentile (q in 1..100) of `values`, or None when there are none. Integer
    arithmetic, so no float rounding can move a rank."""
    if not 1 <= q <= 100:
        raise ValueError(f"q must be between 1 and 100, got {q}")
    if not values:
        return None
    ordered = sorted(values)
    rank = (q * len(ordered) + 99) // 100  # ceil(q * n / 100)
    return ordered[rank - 1]


def _per_doc(total: Decimal, documents: int) -> Decimal | None:
    if not documents:
        return None
    return (total / documents).quantize(_PER_DOC_PLACES, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class PerfStats:
    documents: int = 0  # attempted: ok + failed
    ok: int = 0
    failed: int = 0
    live_calls: int = 0  # ok documents not answered from the cache
    cached_documents: int = 0  # ok documents answered from the cache
    cost_usd: Decimal = Decimal(0)  # paid
    list_cost_usd: Decimal = Decimal(0)  # paid + avoided
    input_tokens: int = 0
    output_tokens: int = 0  # billable
    repairs: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_mean_ms: float | None = None

    @property
    def cost_per_doc(self) -> Decimal | None:
        return _per_doc(self.cost_usd, self.documents)

    @property
    def list_cost_per_doc(self) -> Decimal | None:
        return _per_doc(self.list_cost_usd, self.documents)

    @classmethod
    def from_records(cls, records: Iterable[DocRecord]) -> "PerfStats":
        attempted = [record for record in records if record.status in ("ok", "failed")]
        live = [record for record in attempted if record.status == "ok" and not record.cache_hit]
        latencies = [record.latency_ms for record in live]
        usages = [record.usage for record in attempted if record.usage is not None]
        return cls(
            documents=len(attempted),
            ok=sum(record.status == "ok" for record in attempted),
            failed=sum(record.status == "failed" for record in attempted),
            live_calls=len(live),
            cached_documents=sum(
                record.status == "ok" and record.cache_hit for record in attempted
            ),
            cost_usd=sum((record.cost_usd for record in attempted), Decimal(0)),
            list_cost_usd=sum(
                (record.cost_usd + record.avoided_cost_usd for record in attempted), Decimal(0)
            ),
            input_tokens=sum(int(usage.get("input_tokens", 0)) for usage in usages),
            output_tokens=sum(int(usage.get("billable_output_tokens", 0)) for usage in usages),
            repairs=sum(record.repairs for record in attempted),
            latency_p50_ms=percentile(latencies, 50),
            latency_p95_ms=percentile(latencies, 95),
            latency_mean_ms=sum(latencies) / len(latencies) if latencies else None,
        )
