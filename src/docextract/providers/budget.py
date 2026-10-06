"""Run-level cost cap (MAX_RUN_COST_USD).

How it behaves:
  - The client records the cost of every completed HTTP call: repairs included, and calls
    whose output then failed to parse (they still cost money). Calls that failed before any
    usage came back cost nothing and are not recorded.
  - Before it starts a call (and before each repair) the client checks the budget and raises
    `BudgetExceededError` once `spent > limit`. Spending exactly the limit is still allowed.
  - The call that crosses the limit completes, because its cost is only known afterwards.
    With concurrency, calls already in flight also complete, so the overshoot is bounded by
    concurrency x the cost of one call.
  - Calls answered from the response cache cost nothing and are never refused.
  - A response with no usage block cannot be costed. Such calls are counted as unmetered and
    logged, so a provider that omits usage cannot silently defeat the cap.

Batch / eval runners: catch `BudgetExceededError`, stop dispatching new work, flush the
partial results, and exit non-zero. It is deliberately not a `ProviderError`, so a loop that
swallows `ProviderError` per document cannot swallow the stop signal by accident.

Not thread-safe: it is meant for one asyncio event loop. Several clients may share one budget.
"""

from decimal import Decimal
from typing import Self

from docextract.config import Settings


class BudgetExceededError(Exception):
    """The run's cost cap was exceeded; no further calls are allowed."""

    def __init__(self, spent: Decimal, limit: Decimal) -> None:
        super().__init__(
            f"run cost cap exceeded: spent ${spent:.6f} of the ${limit:.6f} limit "
            "(MAX_RUN_COST_USD)"
        )
        self.spent = spent
        self.limit = limit


class RunBudget:
    def __init__(self, limit: Decimal | None = None) -> None:
        if limit is not None and limit <= 0:
            raise ValueError("a budget limit must be positive (use None for no limit)")
        self.limit = limit
        self.spent = Decimal(0)
        self.calls = 0  # calls whose cost was recorded
        self.unmetered_calls = 0  # calls that returned no usage, so their cost is unknown
        self.cached_calls = 0  # calls answered from the response cache (cost 0)

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings.max_run_cost_usd)

    @property
    def exceeded(self) -> bool:
        return self.limit is not None and self.spent > self.limit

    @property
    def remaining(self) -> Decimal | None:
        return None if self.limit is None else self.limit - self.spent

    def record(self, cost: Decimal, *, metered: bool = True) -> None:
        if cost < 0:
            raise ValueError("a call cannot cost a negative amount")
        self.spent += cost
        if metered:
            self.calls += 1
        else:
            self.unmetered_calls += 1

    def record_cached(self) -> None:
        """A call served from the response cache: free, and counted apart from real calls."""
        self.cached_calls += 1

    def ensure_available(self) -> None:
        if self.limit is not None and self.spent > self.limit:
            raise BudgetExceededError(self.spent, self.limit)
