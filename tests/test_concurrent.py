import asyncio

import pytest

from docextract.extract.concurrent import run_bounded


class Gauge:
    """Counts how many workers run at once."""

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0
        self.started: list[int] = []

    async def work(self, item: int, delay: float = 0.02) -> int:
        self.started.append(item)
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(delay)
        finally:
            self.running -= 1
        return item * 10


@pytest.mark.parametrize("concurrency", [1, 2, 3, 8])
async def test_at_most_n_workers_run_at_once(concurrency: int) -> None:
    gauge = Gauge()
    results = await run_bounded(list(range(8)), gauge.work, concurrency=concurrency)
    assert gauge.peak == min(concurrency, 8)
    assert results == [i * 10 for i in range(8)]


async def test_results_keep_the_input_order_even_when_later_items_finish_first() -> None:
    gauge = Gauge()

    async def slow_first(item: int) -> int:
        return await gauge.work(item, delay=0.1 if item == 0 else 0.0)

    assert await run_bounded([0, 1, 2], slow_first, concurrency=3) == [0, 10, 20]


async def test_with_one_worker_items_start_in_input_order() -> None:
    gauge = Gauge()
    await run_bounded(list(range(5)), gauge.work, concurrency=1)
    assert gauge.started == [0, 1, 2, 3, 4]


async def test_an_empty_list_is_fine() -> None:
    gauge = Gauge()
    assert await run_bounded([], gauge.work, concurrency=3) == []


async def test_skip_is_asked_after_the_semaphore_so_a_running_worker_can_stop_the_rest() -> None:
    stopped = False
    done: list[int] = []

    async def work(item: int) -> str:
        nonlocal stopped
        await asyncio.sleep(0.01)
        if item == 1:
            stopped = True  # like the cost cap
        done.append(item)
        return f"ran {item}"

    results = await run_bounded(
        list(range(6)),
        work,
        concurrency=2,
        skip=lambda: stopped,
        skipped=lambda item: f"skipped {item}",
    )
    # 0 and 1 started together; 1 raised the flag; 2.. had not won the semaphore yet
    assert results[:2] == ["ran 0", "ran 1"]
    assert results[2:] == [f"skipped {i}" for i in range(2, 6)]
    assert done == [0, 1]


async def test_a_failing_worker_cancels_the_others_and_raises_as_itself() -> None:
    cancelled: list[int] = []

    async def work(item: int) -> int:
        if item == 1:
            raise KeyError("boom")
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(item)
            raise
        return item

    with pytest.raises(KeyError, match="boom"):
        await run_bounded([0, 1, 2], work, concurrency=3)
    assert sorted(cancelled) == [0, 2]


@pytest.mark.parametrize("concurrency", [0, -1])
async def test_concurrency_must_be_positive(concurrency: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        await run_bounded([1], Gauge().work, concurrency=concurrency)


async def test_skip_and_skipped_go_together() -> None:
    with pytest.raises(ValueError, match="go together"):
        await run_bounded([1], Gauge().work, concurrency=1, skip=lambda: False)
    with pytest.raises(ValueError, match="go together"):
        await run_bounded([1], Gauge().work, concurrency=1, skipped=lambda item: 0)
