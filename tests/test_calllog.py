import json
import logging
import threading
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from docextract.calllog import CallLog
from docextract.extract.runner import DocRecord, PageRef

FIXED = datetime(2026, 10, 8, 12, 0, 0, 123000, tzinfo=UTC)
KEYS = {
    "ts",
    "request_id",
    "route",
    "index",
    "status",
    "model",
    "pages",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "avoided_cost_usd",
    "cache_hit",
    "latency_ms",
    "attempts",
    "repairs",
    "error",
}


def record(doc_id: str = "my-secret-invoice-name", **overrides: object) -> DocRecord:
    values: dict[str, object] = {
        "doc_id": doc_id,
        "status": "ok",
        "pages": [PageRef(file="p1.png", sha256="a" * 64), PageRef(file="p2.png", sha256="b" * 64)],
        "raw_text": '{"vendor_name": "ACME CONFIDENTIAL LTD"}',
        "usage": {"input_tokens": 1500, "billable_output_tokens": 700},
        "cost_usd": Decimal("0.0123"),
        "avoided_cost_usd": Decimal("0.5"),
        "cache_hit": False,
        "latency_ms": 1234.5678,
        "attempts": 2,
        "repairs": 1,
    }
    values.update(overrides)
    return DocRecord.model_validate(values)


def make(tmp_path: Path | None = None, **kwargs: object) -> CallLog:
    path = None if tmp_path is None else tmp_path / "logs" / "calls.jsonl"
    return CallLog(path, model="test-model", clock=lambda: FIXED, **kwargs)  # type: ignore[arg-type]


def test_a_line_has_exactly_the_documented_keys_and_values(tmp_path: Path) -> None:
    entry = make(tmp_path).log(record(), request_id="req-1", route="/extract", index=3)
    assert set(entry) == KEYS
    assert entry == {
        "ts": "2026-10-08T12:00:00.123+00:00",
        "request_id": "req-1",
        "route": "/extract",
        "index": 3,
        "status": "ok",
        "model": "test-model",
        "pages": 2,
        "input_tokens": 1500,
        "output_tokens": 700,
        "cost_usd": "0.0123",
        "avoided_cost_usd": "0.5",
        "cache_hit": False,
        "latency_ms": 1234.6,
        "attempts": 2,
        "repairs": 1,
        "error": None,
    }


def test_the_file_gets_one_valid_json_line_per_request(tmp_path: Path) -> None:
    log = make(tmp_path)
    log.log(record(), request_id="r1", route="/extract")
    log.log(
        record(status="failed", error="HTTP 500"), request_id="r2", route="/extract/batch", index=1
    )
    lines = (tmp_path / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]  # creates the folder, every line valid JSON
    assert [p["request_id"] for p in parsed] == ["r1", "r2"]
    assert [p["status"] for p in parsed] == ["ok", "failed"]
    assert parsed[1]["error"] == "HTTP 500" and parsed[1]["index"] == 1
    assert all(list(p) == sorted(p) for p in parsed)  # sorted keys: stable bytes


def test_neither_the_file_name_nor_the_content_is_logged(tmp_path: Path) -> None:
    make(tmp_path).log(record(), request_id="r1", route="/extract")
    text = (tmp_path / "logs" / "calls.jsonl").read_text(encoding="utf-8")
    assert "my-secret-invoice-name" not in text
    assert "ACME CONFIDENTIAL" not in text and "p1.png" not in text and "a" * 64 not in text


def test_a_long_error_is_cut(tmp_path: Path) -> None:
    entry = make(tmp_path).log(
        record(status="failed", error="x" * 5000), request_id="r", route="/extract"
    )
    assert entry["error"] == "x" * 200


def test_a_cached_answer_and_a_missing_usage_are_logged_as_such() -> None:
    entry = make().log(
        record(cache_hit=True, usage=None, cost_usd=Decimal(0), latency_ms=0.0),
        request_id="r",
        route="/extract",
    )
    assert entry["cache_hit"] is True and entry["cost_usd"] == "0"
    assert (entry["input_tokens"], entry["output_tokens"]) == (0, 0)


def test_with_no_path_the_line_goes_to_the_logger_only(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="docextract.calls"):
        make().log(record(), request_id="only-logger", route="/extract")
    messages = [r.getMessage() for r in caplog.records if r.name == "docextract.calls"]
    assert len(messages) == 1 and json.loads(messages[0])["request_id"] == "only-logger"


def test_the_logger_gets_the_same_line_as_the_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="docextract.calls"):
        make(tmp_path).log(record(), request_id="same", route="/extract")
    file_line = (tmp_path / "logs" / "calls.jsonl").read_text(encoding="utf-8").strip()
    assert [r.getMessage() for r in caplog.records if r.name == "docextract.calls"] == [file_line]


def test_several_writers_never_mix_their_lines(tmp_path: Path) -> None:
    log = make(tmp_path)

    def write(n: int) -> None:
        for i in range(25):
            log.log(record(), request_id=f"t{n}-{i}", route="/extract/batch", index=i)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = (tmp_path / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    assert len({json.loads(line)["request_id"] for line in lines}) == 100


def test_appending_keeps_what_an_earlier_process_wrote(tmp_path: Path) -> None:
    make(tmp_path).log(record(), request_id="first", route="/extract")
    make(tmp_path).log(record(), request_id="second", route="/extract")
    lines = (tmp_path / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["request_id"] for line in lines] == ["first", "second"]


def test_a_log_that_cannot_be_written_does_not_fail_the_request(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the log folder should be", encoding="utf-8")
    log = CallLog(blocker / "calls.jsonl", model="test-model", clock=lambda: FIXED)
    with caplog.at_level(logging.WARNING, logger="docextract.calls"):
        entry = log.log(record(), request_id="r", route="/extract")
    assert entry["request_id"] == "r"
    assert any("cannot be written" in r.getMessage() for r in caplog.records)
