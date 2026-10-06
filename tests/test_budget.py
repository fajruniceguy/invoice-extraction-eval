from decimal import Decimal

import pytest

from docextract.config import Settings
from docextract.providers.budget import BudgetExceededError, RunBudget
from docextract.providers.openai_compat import ProviderError

D = Decimal


class TestRunBudget:
    def test_starts_empty(self) -> None:
        budget = RunBudget(D(10))
        assert (budget.spent, budget.calls, budget.unmetered_calls) == (0, 0, 0)
        assert budget.limit == 10
        assert budget.remaining == 10
        assert not budget.exceeded

    def test_record_accumulates_cost_and_counts_calls(self) -> None:
        budget = RunBudget(D(10))
        budget.record(D("1.25"))
        budget.record(D("0.75"))
        assert budget.spent == D(2)
        assert budget.calls == 2
        assert budget.remaining == D(8)

    def test_unlimited_budget_tracks_spend_but_never_refuses(self) -> None:
        budget = RunBudget()
        budget.record(D("1000000"))
        budget.ensure_available()
        assert budget.limit is None
        assert budget.remaining is None
        assert not budget.exceeded
        assert budget.spent == D("1000000")

    def test_spending_exactly_the_limit_is_not_exceeding_it(self) -> None:
        budget = RunBudget(D(5))
        budget.record(D(5))
        assert not budget.exceeded
        budget.ensure_available()  # does not raise
        assert budget.remaining == 0

    def test_one_cent_over_is_exceeded(self) -> None:
        budget = RunBudget(D(5))
        budget.record(D("5.01"))
        assert budget.exceeded
        assert budget.remaining == D("-0.01")
        with pytest.raises(BudgetExceededError):
            budget.ensure_available()

    def test_error_carries_the_figures_and_names_the_setting(self) -> None:
        budget = RunBudget(D("0.05"))
        budget.record(D("0.0612"))
        with pytest.raises(BudgetExceededError) as exc:
            budget.ensure_available()
        assert exc.value.spent == D("0.0612")
        assert exc.value.limit == D("0.05")
        assert "MAX_RUN_COST_USD" in str(exc.value)
        assert "0.061200" in str(exc.value)

    def test_calls_without_usage_are_counted_separately(self) -> None:
        budget = RunBudget(D(5))
        budget.record(D(0), metered=False)
        budget.record(D(1))
        assert (budget.calls, budget.unmetered_calls, budget.spent) == (1, 1, D(1))

    def test_a_negative_cost_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            RunBudget(D(5)).record(D(-1))

    @pytest.mark.parametrize("limit", [D(0), D(-1)])
    def test_a_non_positive_limit_is_rejected(self, limit: Decimal) -> None:
        with pytest.raises(ValueError, match="positive"):
            RunBudget(limit)

    def test_from_settings(self) -> None:
        settings = Settings(
            _env_file=None,
            llm_base_url="http://x/v1",
            llm_model="m",
            llm_price_input_per_mtok="1",
            max_run_cost_usd="2.5",
        )
        assert RunBudget.from_settings(settings).limit == D("2.5")
        uncapped = Settings(_env_file=None, llm_base_url="http://x/v1", llm_model="m")
        assert RunBudget.from_settings(uncapped).limit is None


def test_budget_exceeded_is_not_a_provider_error() -> None:
    # a batch loop that swallows ProviderError per document must not swallow the stop signal
    assert not issubclass(BudgetExceededError, ProviderError)

    def swallow_provider_errors() -> None:
        try:
            raise BudgetExceededError(D(2), D(1))
        except ProviderError:  # pragma: no cover - must not be reached
            pytest.fail("BudgetExceededError was caught as a ProviderError")

    with pytest.raises(BudgetExceededError):
        swallow_provider_errors()
