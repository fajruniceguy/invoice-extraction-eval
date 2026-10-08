"""One JSON line per document request: tokens, cost and latency, for the API (and anything else
that wants an audit trail of what the model cost).

A line holds the request id, the route, the document's position in its request, the outcome, the
model, how many pages went out, input and billable output tokens, cost paid and cost avoided by the
response cache, whether the cache answered, latency, HTTP attempts and repairs. It never holds a
file name, a document id derived from one, or any of the document's content: the caller names its
own files, and a log outlives the request. Lines go to the `docextract.calls` logger and, when a
path is set, are appended to that JSONL file.

A log that cannot be written must not fail the request it describes: the error is reported once
per failure on the logger and the request goes on.
"""

import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docextract.extract.runner import DocRecord

logger = logging.getLogger("docextract.calls")

_ERROR_CHARS = 200


def _now() -> datetime:
    return datetime.now(UTC)


class CallLog:
    def __init__(
        self,
        path: Path | None = None,
        *,
        model: str,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.path = path
        self.model = model
        self._clock = clock
        self._lock = threading.Lock()

    def entry(
        self, record: DocRecord, *, request_id: str, route: str, index: int
    ) -> dict[str, Any]:
        usage = record.usage or {}
        return {
            "ts": self._clock().isoformat(timespec="milliseconds"),
            "request_id": request_id,
            "route": route,
            "index": index,
            "status": record.status,
            "model": self.model,
            "pages": len(record.pages),
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("billable_output_tokens", 0)),
            "cost_usd": str(record.cost_usd),
            "avoided_cost_usd": str(record.avoided_cost_usd),
            "cache_hit": record.cache_hit,
            "latency_ms": round(record.latency_ms, 1),
            "attempts": record.attempts,
            "repairs": record.repairs,
            "error": None if record.error is None else record.error[:_ERROR_CHARS],
        }

    def log(
        self, record: DocRecord, *, request_id: str, route: str, index: int = 0
    ) -> dict[str, Any]:
        """Write the line for `record` and return it."""
        entry = self.entry(record, request_id=request_id, route=route, index=index)
        line = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        logger.info(line)
        if self.path is not None:
            try:
                with self._lock:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(line + "\n")
            except OSError as exc:
                logger.warning("the call log %s cannot be written: %s", self.path, exc)
        return entry
