import asyncio
import io
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

import docextract.api.app as api_module
from docextract.api.app import create_app
from docextract.api.uploads import pages_from_uploads
from docextract.config import Settings
from docextract.extract.prompt import SYSTEM_PROMPT
from docextract.extract.runner import DocRecord, extract_record
from docextract.providers.openai_compat import OpenAICompatClient
from fakes.fake_llm import FakeLLMServer, Reply, canned_answer, chat_reply
from files import A4, make_sized_pdf

TOKEN = "s3cret-token"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a real LLM_* / API_* variable (or a real key) into these tests."""
    for name in list(os.environ):
        if name.startswith(
            ("LLM_", "PRELABEL_", "MAX_RUN_COST", "API_", "CALL_LOG")
        ) and name not in (
            "LLM_CACHE",
            "LLM_CACHE_DIR",
        ):
            monkeypatch.delenv(name)


def make_settings(server: FakeLLMServer, **changes: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_base_url": server.base_url,
        "llm_model": "fake-extractor",
        "llm_api_key": "sk-test-key",
        "llm_max_retries": 0,
        "llm_max_repairs": 1,
    }
    values.update(changes)
    return Settings(_env_file=None, **values)


@asynccontextmanager
async def serve(config: Settings, **kwargs: Any) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    app = create_app(config, **kwargs)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as http:
            yield http, app


def capped() -> dict[str, Any]:
    """Prices that make every call cost $1.50 against a $1.00 cap (as in the runner tests)."""
    return {
        "llm_price_input_per_mtok": "1000",
        "llm_price_output_per_mtok": "0",
        "max_run_cost_usd": "1.00",
    }


def png(width: int = 80, height: int = 60, colour: int = 200) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (colour, 100, 50)).save(buffer, "PNG")
    return buffer.getvalue()


def pdf(tmp_path: Path, pages: int = 2, name: str = "doc.pdf") -> bytes:
    return make_sized_pdf(tmp_path / name, [A4] * pages).read_bytes()


def upload(*items: tuple[str, bytes, str], field: str = "files") -> list[tuple[str, Any]]:
    return [(field, (name, data, mime)) for name, data, mime in items]


def one_png(name: str = "a.png") -> list[tuple[str, Any]]:
    return upload((name, png(), "image/png"))


def image_count(request_body: dict[str, Any]) -> int:
    parts = request_body["messages"][1]["content"]
    return sum(part["type"] == "image_url" for part in parts)


def peak_in_flight(requests: list[Any], delay: float) -> int:
    starts = sorted(request.received_at for request in requests)
    return max(sum(1 for other in starts if start <= other < start + delay) for start in starts)


def slow_replies(count: int, delay: float) -> list[Reply]:
    return [Reply(200, chat_reply(canned_answer()).body, delay_s=delay)] * count


# ------------------------------------------------------------------ /health


class TestHealth:
    async def test_it_reports_status_and_budget_but_no_secret(
        self, fake_llm: FakeLLMServer
    ) -> None:
        config = make_settings(fake_llm, api_token=TOKEN, api_concurrency=3, **capped())
        async with serve(config) as (http, _):
            response = await http.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok" and body["model"] == "fake-extractor"
        assert body["concurrency"] == 3 and body["max_batch"] == 20
        assert body["budget"] == {"spent_usd": "0", "limit_usd": "1.00", "exceeded": False}
        assert body["calls"] == 0 and body["cached_calls"] == 0
        assert "sk-test-key" not in response.text and TOKEN not in response.text

    async def test_it_reports_what_a_run_record_would_say_about_the_server(
        self, fake_llm: FakeLLMServer
    ) -> None:
        from docextract.extract.prompt import prompt_sha256
        from docextract.extract.schema import DocumentExtraction
        from docextract.providers.cache import schema_hash

        config = make_settings(
            fake_llm, llm_reasoning_effort="low", llm_max_pages=7, **capped()
        )
        async with serve(config) as (http, _):
            body = (await http.get("/health")).json()
        assert body["endpoint"] == fake_llm.base_url
        assert body["prompt_sha256"] == prompt_sha256(SYSTEM_PROMPT)
        assert body["schema_sha256"] == schema_hash(DocumentExtraction)
        assert body["settings"] == {
            "structured_mode": config.llm_structured_mode,
            "reasoning_effort": "low",
            "image_max_long_edge": None,
            "max_pages": 7,
            "max_repairs": 1,
            "cache": "off",
            "max_run_cost_usd": "1.00",
            "temperature": 0.0,
        }

    async def test_it_needs_no_token_and_carries_a_request_id(
        self, fake_llm: FakeLLMServer
    ) -> None:
        async with serve(make_settings(fake_llm, api_token=TOKEN)) as (http, _):
            response = await http.get("/health")
        assert response.status_code == 200
        assert len(response.headers["x-request-id"]) == 12


# ------------------------------------------------------------------ /extract


class TestExtract:
    async def test_a_pdf_becomes_a_record_with_every_page_in_one_call(
        self, fake_llm: FakeLLMServer, tmp_path: Path
    ) -> None:
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post(
                "/extract", files=upload(("invoice.pdf", pdf(tmp_path, 2), "application/pdf"))
            )
        assert response.status_code == 200, response.text
        record = response.json()
        assert (
            record["status"] == "ok"
            and record["doc_id"] == f"upload-{response.headers['x-request-id']}"
        )
        assert record["validated"]["vendor_name"]["value"] == "CORNER CAFE DEMO"
        assert len(record["pages"]) == 2 and record["repairs"] == 0
        assert len(fake_llm.requests) == 1 and image_count(fake_llm.requests[0].body) == 2
        assert fake_llm.requests[0].body["messages"][0]["content"] == SYSTEM_PROMPT

    async def test_a_png(self, fake_llm: FakeLLMServer) -> None:
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post("/extract", files=one_png())
        assert response.status_code == 200 and len(response.json()["pages"]) == 1

    async def test_several_files_are_one_document_in_order(self, fake_llm: FakeLLMServer) -> None:
        files = upload(
            ("1.png", png(80, 60, 10), "image/png"), ("2.png", png(60, 80, 90), "image/png")
        )
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post("/extract", files=files)
        assert response.status_code == 200 and len(response.json()["pages"]) == 2
        assert len(fake_llm.requests) == 1 and image_count(fake_llm.requests[0].body) == 2

    async def test_compact_drops_the_bulky_fields(self, fake_llm: FakeLLMServer) -> None:
        async with serve(make_settings(fake_llm)) as (http, _):
            full = (await http.post("/extract", files=one_png())).json()
            compact = (await http.post("/extract?compact=true", files=one_png())).json()
        assert full["raw_text"] and full["parsed"] is not None
        for dropped in ("raw_text", "parsed", "repair_history"):
            assert dropped not in compact
        assert compact["validated"] == full["validated"]

    async def test_doc_id_and_page_range_are_honored(
        self, fake_llm: FakeLLMServer, tmp_path: Path
    ) -> None:
        files = upload(("five.pdf", pdf(tmp_path, 5), "application/pdf"))
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post(
                "/extract", files=files, data={"doc_id": "my-doc.1", "page_range": "2-3"}
            )
        record = response.json()
        assert response.status_code == 200 and record["doc_id"] == "my-doc.1"
        assert len(record["pages"]) == 2 and image_count(fake_llm.requests[0].body) == 2

    async def test_the_record_equals_the_one_the_runner_would_write(
        self, fake_llm: FakeLLMServer, tmp_path: Path
    ) -> None:
        data = pdf(tmp_path, 2)
        config = make_settings(fake_llm)
        async with serve(config) as (http, _):
            via_api = (
                await http.post(
                    "/extract",
                    files=upload(("a.pdf", data, "application/pdf")),
                    data={"doc_id": "same"},
                )
            ).json()
        pages = pages_from_uploads([data], tmp_path / "work", max_pages=15)
        async with OpenAICompatClient(config) as client:
            direct, _ = await extract_record(
                "same",
                pages,
                client=client,
                settings=config,
                cache=None,
                system_prompt=SYSTEM_PROMPT,
            )
        expected = direct.model_dump(mode="json", exclude={"latency_ms"})
        via_api.pop("latency_ms")
        assert via_api == expected
        assert DocRecord.model_validate(expected).pages == direct.pages  # same page hashes


class TestExtractErrors:
    async def test_no_files(self, fake_llm: FakeLLMServer) -> None:
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post("/extract")
        assert response.status_code == 400
        error = response.json()["error"]
        assert (
            error["code"] == "no_files" and error["request_id"] == response.headers["x-request-id"]
        )

    async def test_an_unreadable_file_is_a_400_and_costs_nothing(
        self, fake_llm: FakeLLMServer
    ) -> None:
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post(
                "/extract", files=upload(("x.png", b"not an image", "image/png"))
            )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unreadable_file"
        assert response.json()["record"]["status"] == "refused"
        assert fake_llm.requests == []

    async def test_a_file_over_the_limit_is_a_413(self, fake_llm: FakeLLMServer) -> None:
        big = b"\x89PNG" + b"0" * (1024 * 1024 + 10)
        async with serve(make_settings(fake_llm, api_max_upload_mb=1)) as (http, _):
            response = await http.post("/extract", files=upload(("big.png", big, "image/png")))
        assert response.status_code == 413 and response.json()["error"]["code"] == "file_too_large"
        assert fake_llm.requests == []

    async def test_a_file_exactly_at_the_limit_gets_past_the_size_check(
        self, fake_llm: FakeLLMServer
    ) -> None:
        limit = 1024 * 1024
        async with serve(make_settings(fake_llm, api_max_upload_mb=1)) as (http, _):
            at_limit = await http.post(
                "/extract", files=upload(("a.png", b"0" * limit, "image/png"))
            )
            over = await http.post(
                "/extract", files=upload(("b.png", b"0" * (limit + 1), "image/png"))
            )
        # exactly at the limit it is read, found to be no image (400); one byte more is a 413
        assert at_limit.status_code == 400 and at_limit.json()["error"]["code"] == "unreadable_file"
        assert over.status_code == 413

    async def test_too_many_pages_is_a_422(self, fake_llm: FakeLLMServer, tmp_path: Path) -> None:
        files = upload(("long.pdf", pdf(tmp_path, 21), "application/pdf"))
        async with serve(make_settings(fake_llm, llm_max_pages=20)) as (http, _):
            response = await http.post("/extract", files=files)
        assert response.status_code == 422 and response.json()["error"]["code"] == "too_many_pages"
        assert fake_llm.requests == []

    async def test_a_bad_doc_id_is_a_400(self, fake_llm: FakeLLMServer) -> None:
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post("/extract", files=one_png(), data={"doc_id": "../etc"})
        assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_doc_id"

    async def test_a_provider_failure_is_a_502_with_the_failed_record(
        self, fake_llm: FakeLLMServer
    ) -> None:
        fake_llm.enqueue(Reply(500, {"error": {"message": "upstream exploded"}}))
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post("/extract", files=one_png())
        assert response.status_code == 502
        body = response.json()
        assert body["error"]["code"] == "provider_error" and "HTTP 500" in body["error"]["message"]
        assert body["record"]["status"] == "failed"

    async def test_a_retried_429_is_one_successful_answer(self, fake_llm: FakeLLMServer) -> None:
        fake_llm.enqueue(Reply(429, {"error": {"message": "slow down"}}))
        config = make_settings(
            fake_llm, llm_max_retries=2, llm_backoff_base_s=0.01, llm_backoff_max_s=0.02
        )
        async with serve(config) as (http, _):
            response = await http.post("/extract", files=one_png())
        assert response.status_code == 200 and response.json()["attempts"] == 2
        assert len(fake_llm.requests) == 2

    async def test_the_cost_cap_makes_the_next_request_a_503(self, fake_llm: FakeLLMServer) -> None:
        async with serve(make_settings(fake_llm, **capped())) as (http, _):
            first = await http.post("/extract", files=one_png())
            second = await http.post("/extract", files=one_png())
            health = (await http.get("/health")).json()
        assert first.status_code == 200  # the call that crosses the cap completes
        assert second.status_code == 503 and second.json()["error"]["code"] == "budget_exceeded"
        assert len(fake_llm.requests) == 1
        assert health["budget"]["exceeded"] is True and health["budget"]["spent_usd"] == "1.5"


class TestToken:
    async def test_without_a_token_configured_everything_is_open(
        self, fake_llm: FakeLLMServer
    ) -> None:
        async with serve(make_settings(fake_llm)) as (http, _):
            assert (await http.post("/extract", files=one_png())).status_code == 200

    @pytest.mark.parametrize("route", ["/extract", "/extract/batch"])
    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Authorization": "Bearer wrong"},
            {"Authorization": TOKEN},
            {"Authorization": "Basic x"},
        ],
        ids=["missing", "wrong", "no-scheme", "wrong-scheme"],
    )
    async def test_a_missing_or_wrong_token_is_a_401_before_any_work(
        self, fake_llm: FakeLLMServer, route: str, headers: dict[str, str]
    ) -> None:
        async with serve(make_settings(fake_llm, api_token=TOKEN)) as (http, _):
            response = await http.post(route, files=one_png(), headers=headers)
        assert response.status_code == 401 and response.headers["www-authenticate"] == "Bearer"
        assert response.json()["error"]["code"] == "unauthorized"
        assert fake_llm.requests == []

    @pytest.mark.parametrize("route", ["/extract", "/extract/batch"])
    async def test_the_right_token_is_accepted(self, fake_llm: FakeLLMServer, route: str) -> None:
        async with serve(make_settings(fake_llm, api_token=TOKEN)) as (http, _):
            response = await http.post(
                route, files=one_png(), headers={"Authorization": f"Bearer {TOKEN}"}
            )
        assert response.status_code == 200


# ------------------------------------------------------------------ /extract/batch


def batch_files(*names: str) -> list[tuple[str, Any]]:
    return upload(*[(name, png(80, 60, 20 + i), "image/png") for i, name in enumerate(names)])


class TestBatch:
    async def test_documents_come_back_in_upload_order_named_after_their_files(
        self, fake_llm: FakeLLMServer
    ) -> None:
        async with serve(make_settings(fake_llm, api_concurrency=3)) as (http, _):
            response = await http.post(
                "/extract/batch", files=batch_files("b.png", "a.png", "c.png")
            )
        assert response.status_code == 200
        body = response.json()
        assert [d["doc_id"] for d in body["documents"]] == ["b", "a", "c"]
        assert {d["status"] for d in body["documents"]} == {"ok"}
        run = body["run"]
        assert (run["documents"], run["ok"], run["failed"], run["refused"]) == (3, 3, 0, 0)
        assert run["stopped"] is None and run["concurrency"] == 3 and run["wall_ms"] > 0
        assert run["input_tokens"] == 3 * 1500 and run["cost_usd"] == "0"
        assert len(fake_llm.requests) == 3

    async def test_duplicate_and_hostile_names_become_safe_distinct_ids(
        self, fake_llm: FakeLLMServer
    ) -> None:
        files = batch_files("x.png", "x.png", "../../etc/passwd.png", "weird name!!.png", "---.png")
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post("/extract/batch", files=files)
        ids = [d["doc_id"] for d in response.json()["documents"]]
        assert ids == ["x", "x-2", "passwd", "weird-name", "doc-5"]

    async def test_one_bad_file_is_refused_while_the_others_succeed(
        self, fake_llm: FakeLLMServer
    ) -> None:
        files = upload(
            ("good1.png", png(80, 60, 1), "image/png"),
            ("bad.png", b"garbage", "image/png"),
            ("good2.png", png(80, 60, 2), "image/png"),
        )
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post("/extract/batch", files=files)
        body = response.json()
        assert response.status_code == 200
        assert [d["status"] for d in body["documents"]] == ["ok", "refused", "ok"]
        assert "cannot open the image" in body["documents"][1]["error"]
        assert (body["run"]["ok"], body["run"]["refused"]) == (2, 1)
        assert len(fake_llm.requests) == 2

    async def test_a_file_over_the_limit_is_refused_alone(self, fake_llm: FakeLLMServer) -> None:
        big = b"\x89PNG" + b"0" * (1024 * 1024 + 10)
        files = upload(("big.png", big, "image/png"), ("ok.png", png(), "image/png"))
        async with serve(make_settings(fake_llm, api_max_upload_mb=1)) as (http, _):
            body = (await http.post("/extract/batch", files=files)).json()
        assert [d["status"] for d in body["documents"]] == ["refused", "ok"]
        assert "larger than 1 MB" in body["documents"][0]["error"]

    async def test_a_provider_failure_marks_only_that_document_failed(
        self, fake_llm: FakeLLMServer
    ) -> None:
        fake_llm.enqueue(Reply(500, {"error": {"message": "upstream exploded"}}))
        async with serve(make_settings(fake_llm, api_concurrency=1)) as (http, _):
            body = (await http.post("/extract/batch", files=batch_files("a.png", "b.png"))).json()
        assert [d["status"] for d in body["documents"]] == ["failed", "ok"]
        assert body["run"]["failed"] == 1 and body["run"]["stopped"] is None

    async def test_more_files_than_the_batch_limit_is_a_413(self, fake_llm: FakeLLMServer) -> None:
        async with serve(make_settings(fake_llm, api_max_batch=2)) as (http, _):
            response = await http.post(
                "/extract/batch", files=batch_files("a.png", "b.png", "c.png")
            )
        assert response.status_code == 413 and response.json()["error"]["code"] == "batch_too_large"
        assert fake_llm.requests == []

    async def test_exactly_the_batch_limit_is_accepted(self, fake_llm: FakeLLMServer) -> None:
        async with serve(make_settings(fake_llm, api_max_batch=2)) as (http, _):
            response = await http.post("/extract/batch", files=batch_files("a.png", "b.png"))
        assert response.status_code == 200 and response.json()["run"]["ok"] == 2

    async def test_no_files(self, fake_llm: FakeLLMServer) -> None:
        async with serve(make_settings(fake_llm)) as (http, _):
            response = await http.post("/extract/batch")
        assert response.status_code == 400 and response.json()["error"]["code"] == "no_files"

    async def test_compact_applies_to_every_document(self, fake_llm: FakeLLMServer) -> None:
        async with serve(make_settings(fake_llm)) as (http, _):
            body = (
                await http.post("/extract/batch?compact=true", files=batch_files("a.png", "b.png"))
            ).json()
        assert all("raw_text" not in d and "parsed" not in d for d in body["documents"])
        assert all(d["validated"] is not None for d in body["documents"])


class TestBatchConcurrency:
    async def test_documents_in_flight_never_exceed_the_limit(
        self, fake_llm: FakeLLMServer
    ) -> None:
        fake_llm.enqueue(*slow_replies(6, 0.3))
        names = [f"{i}.png" for i in range(6)]
        async with serve(make_settings(fake_llm, api_concurrency=2)) as (http, _):
            response = await http.post("/extract/batch", files=batch_files(*names))
        assert response.json()["run"]["ok"] == 6
        assert peak_in_flight(fake_llm.requests, 0.3) == 2

    async def test_parallel_batches_finish_sooner_than_serial_ones(
        self, fake_llm: FakeLLMServer
    ) -> None:
        names = [f"{i}.png" for i in range(4)]
        timings = {}
        for limit in (1, 4):
            fake_llm.enqueue(*slow_replies(4, 0.2))
            async with serve(make_settings(fake_llm, api_concurrency=limit)) as (http, _):
                started = time.monotonic()
                await http.post("/extract/batch", files=batch_files(*names))
                timings[limit] = time.monotonic() - started
        assert timings[1] >= 4 * 0.2 and timings[4] < 0.75 * timings[1]

    async def test_the_limit_is_shared_by_simultaneous_requests(
        self, fake_llm: FakeLLMServer
    ) -> None:
        fake_llm.enqueue(*slow_replies(6, 0.3))
        async with serve(make_settings(fake_llm, api_concurrency=2)) as (http, _):
            first, second = await asyncio.gather(
                http.post("/extract/batch", files=batch_files("a.png", "b.png", "c.png")),
                http.post("/extract/batch", files=batch_files("d.png", "e.png", "f.png")),
            )
        assert first.json()["run"]["ok"] == 3 and second.json()["run"]["ok"] == 3
        assert len(fake_llm.requests) == 6
        assert peak_in_flight(fake_llm.requests, 0.3) == 2  # not 4: one cap for the whole server

    async def test_a_single_extract_also_counts_against_the_shared_limit(
        self, fake_llm: FakeLLMServer
    ) -> None:
        fake_llm.enqueue(*slow_replies(4, 0.3))
        async with serve(make_settings(fake_llm, api_concurrency=2)) as (http, _):
            await asyncio.gather(
                http.post("/extract/batch", files=batch_files("a.png", "b.png", "c.png")),
                http.post("/extract", files=one_png()),
            )
        assert peak_in_flight(fake_llm.requests, 0.3) == 2

    async def test_health_answers_while_a_page_is_being_rendered(
        self, fake_llm: FakeLLMServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def slow_render(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.6)  # a blocked event loop would hold /health for all of this
            return pages_from_uploads(*args, **kwargs)

        monkeypatch.setattr(api_module, "pages_from_uploads", slow_render)
        async with serve(make_settings(fake_llm)) as (http, _):
            slow = asyncio.create_task(http.post("/extract", files=one_png()))
            await asyncio.sleep(0.15)
            started = time.monotonic()
            health = await http.get("/health")
            waited = time.monotonic() - started
            assert (await slow).status_code == 200
        assert health.status_code == 200 and waited < 0.3


class TestBatchCostCap:
    async def test_after_the_cap_the_documents_that_had_not_started_are_not_run(
        self, fake_llm: FakeLLMServer
    ) -> None:
        names = ["a.png", "b.png", "c.png"]
        async with serve(make_settings(fake_llm, api_concurrency=1, **capped())) as (http, _):
            response = await http.post("/extract/batch", files=batch_files(*names))
        body = response.json()
        assert response.status_code == 200  # a batch always answers 200
        assert [d["status"] for d in body["documents"]] == ["ok", "not_run", "not_run"]
        assert body["run"]["stopped"] == "cost_cap" and body["run"]["not_run"] == 2
        assert body["documents"][2]["error"] == "the cost cap was reached"
        assert len(fake_llm.requests) == 1

    async def test_calls_already_in_flight_finish(self, fake_llm: FakeLLMServer) -> None:
        names = [f"{i}.png" for i in range(5)]
        async with serve(make_settings(fake_llm, api_concurrency=2, **capped())) as (http, _):
            body = (await http.post("/extract/batch", files=batch_files(*names))).json()
        assert [d["status"] for d in body["documents"]] == ["ok", "ok"] + ["not_run"] * 3
        assert len(fake_llm.requests) == 2 and body["run"]["cost_usd"] == "3.0"


# ------------------------------------------------------------------ the call log


class TestCallLog:
    def read(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    async def test_every_request_and_every_batch_document_is_logged(
        self, fake_llm: FakeLLMServer, tmp_path: Path
    ) -> None:
        log = tmp_path / "logs" / "calls.jsonl"
        async with serve(make_settings(fake_llm, call_log_path=log)) as (http, _):
            single = await http.post("/extract", files=one_png("secret-name.png"))
            batch = await http.post(
                "/extract/batch", files=batch_files("confidential-1.png", "confidential-2.png")
            )
        lines = self.read(log)
        assert len(lines) == 3
        # batch documents finish in any order: compare as a set
        assert sorted((line["route"], line["index"]) for line in lines) == [
            ("/extract", 0),
            ("/extract/batch", 0),
            ("/extract/batch", 1),
        ]
        by_route = {line["route"]: line for line in lines}
        assert by_route["/extract"]["request_id"] == single.headers["x-request-id"]
        assert {line["request_id"] for line in lines if line["route"] == "/extract/batch"} == {
            batch.headers["x-request-id"]
        }
        first = lines[0]
        assert first["status"] == "ok" and first["model"] == "fake-extractor"
        assert (first["input_tokens"], first["output_tokens"]) == (1500, 700)
        assert first["pages"] == 1 and first["cache_hit"] is False

    async def test_neither_file_names_nor_content_reach_the_log(
        self, fake_llm: FakeLLMServer, tmp_path: Path
    ) -> None:
        log = tmp_path / "calls.jsonl"
        async with serve(make_settings(fake_llm, call_log_path=log)) as (http, _):
            await http.post("/extract", files=one_png("secret-name.png"))
            await http.post("/extract/batch", files=batch_files("confidential.png"))
        text = log.read_text(encoding="utf-8")
        assert "secret-name" not in text and "confidential" not in text
        assert "CORNER CAFE" not in text

    async def test_refused_and_failed_documents_are_logged_too(
        self, fake_llm: FakeLLMServer, tmp_path: Path
    ) -> None:
        log = tmp_path / "calls.jsonl"
        async with serve(make_settings(fake_llm, call_log_path=log)) as (http, _):
            await http.post("/extract", files=upload(("x.png", b"junk", "image/png")))
            await http.post("/extract/batch", files=upload(("x.png", b"junk", "image/png")))
        assert [line["status"] for line in self.read(log)] == ["refused", "refused"]

    async def test_without_a_log_path_requests_still_work(self, fake_llm: FakeLLMServer) -> None:
        async with serve(make_settings(fake_llm)) as (http, _):
            assert (await http.post("/extract", files=one_png())).status_code == 200


# ------------------------------------------------------------------ the injected client


async def test_an_injected_client_is_used_and_left_open(fake_llm: FakeLLMServer) -> None:
    config = make_settings(fake_llm)
    client = OpenAICompatClient(config)
    async with serve(config, client=client) as (http, app):
        assert app.state.api.client is client
        assert (await http.post("/extract", files=one_png())).status_code == 200
    # the app did not close it: it can still be used
    async with client:
        pass


async def test_the_cache_makes_a_repeat_free(fake_llm: FakeLLMServer, tmp_path: Path) -> None:
    config = make_settings(
        fake_llm,
        llm_cache="on",
        llm_cache_dir=tmp_path / "cache",
        **capped() | {"max_run_cost_usd": "10"},
    )
    data = png()
    async with serve(config) as (http, _):
        first = (await http.post("/extract", files=upload(("a.png", data, "image/png")))).json()
        second = (await http.post("/extract", files=upload(("b.png", data, "image/png")))).json()
    assert first["cache_hit"] is False and first["cost_usd"] == "1.5"
    assert second["cache_hit"] is True and second["cost_usd"] == "0"
    assert second["avoided_cost_usd"] == "1.5"
    assert len(fake_llm.requests) == 1
