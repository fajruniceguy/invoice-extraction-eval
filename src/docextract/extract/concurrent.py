"""Run async work over a list with at most N tasks at a time, keeping the input order.

Used by the extraction runner (`--concurrency`) and the API's batch endpoint, so both bound
concurrency the same way: one semaphore, FIFO, results in input order.

`skip` is asked AFTER a task has won the semaphore, so a stop signal raised by a running worker
(the run cost cap) reaches every document that has not started yet: those get `skipped(item)`
instead of a call. A worker that raises cancels the others and its exception propagates as
itself, not wrapped in an ExceptionGroup.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def run_bounded(
    items: Sequence[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
    skip: Callable[[], bool] | None = None,
    skipped: Callable[[T], R] | None = None,
) -> list[R]:
    """`[await worker(item) for item in items]`, at most `concurrency` at a time."""
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    if (skip is None) != (skipped is None):
        raise ValueError("skip and skipped go together")
    semaphore = asyncio.Semaphore(concurrency)
    results: list[R | None] = [None] * len(items)

    async def one(index: int, item: T) -> None:
        async with semaphore:
            if skip is not None and skipped is not None and skip():
                results[index] = skipped(item)
            else:
                results[index] = await worker(item)

    try:
        async with asyncio.TaskGroup() as group:
            for index, item in enumerate(items):
                group.create_task(one(index, item))
    except ExceptionGroup as failure:
        raise failure.exceptions[0] from None
    return results  # type: ignore[return-value]  # every slot was filled
