"""Integration tests: the real client talking real HTTP to tests/fakes/fake_llm.py.

Unlike test_provider_client.py (httpx.MockTransport), these go through actual sockets, so
they also cover connection handling, headers on the wire and the smoke script end to end.
Sleep is injected so backoff is recorded rather than waited out.
"""

import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from docextract.config import Settings
from docextract.extract.schema import DocumentExtraction
from docextract.providers.openai_compat import (
    OpenAICompatClient,
    ProviderError,
    ResponseParseError,
    Usage,
    build_image_message,
)
from docextract.validate.arithmetic import validate_line_items
from fakes.fake_llm import FakeLLMServer, Reply, canned_answer, chat_reply

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_receipt.png"
MESSAGES = [
    {"role": "system", "content": "You extract receipts."},
    build_image_message("Extract the fields.", [FIXTURE]),
]
BAD_TEXT = "Sure! Here is the receipt: the total was 17.28."


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def make_client(server: FakeLLMServer, sleeps: Sleeps, **overrides: Any) -> OpenAICompatClient:
    settings = Settings(
        _env_file=None,
        llm_base_url=server.base_url,
        llm_model="fake-model",
        llm_api_key="sk-not-real",
        llm_backoff_base_s=1.0,
        llm_backoff_max_s=30.0,
        **overrides,
    )
    return OpenAICompatClient(settings, sleep=sleeps, rng=lambda: 1.0)


# ---------------------------------------------------------------- structured output


async def test_json_schema_rejection_falls_back_and_the_extraction_is_valid(
    fake_llm: FakeLLMServer,
) -> None:
    fake_llm.reject_json_schema = True
    async with make_client(fake_llm, Sleeps()) as client:
        result = await client.chat(MESSAGES, DocumentExtraction)

    assert result.structured_mode_used == "json_object"
    assert result.attempts == 2
    assert [r.response_format for r in fake_llm.requests] == ["json_schema", "json_object"]

    doc = result.parsed
    assert doc is not None
    assert doc.vendor_name.value == "CORNER CAFE DEMO"
    assert doc.buyer_name.status == "absent"
    assert doc.total.value == Decimal("17.28")
    assert len(doc.line_items) == 3
    # the raw output passes the arithmetic validator untouched
    assert all(row.flagged_cells == [] for row in validate_line_items(doc).line_items)


async def test_image_auth_and_schema_really_cross_the_wire(fake_llm: FakeLLMServer) -> None:
    async with make_client(fake_llm, Sleeps()) as client:
        await client.chat(MESSAGES, DocumentExtraction)

    request = fake_llm.requests[0]
    assert request.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-not-real"
    user_parts = request.body["messages"][-1]["content"]
    image_part = next(part for part in user_parts if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    schema = request.body["response_format"]["json_schema"]["schema"]
    assert "total" in schema["properties"]
    assert "$ref" not in str(schema)
    assert "flagged_cells" not in str(schema)  # derived by validate/, never asked of the model


async def test_the_fallback_is_remembered_by_the_client(fake_llm: FakeLLMServer) -> None:
    fake_llm.reject_json_schema = True
    async with make_client(fake_llm, Sleeps()) as client:
        await client.chat(MESSAGES, DocumentExtraction)
        second = await client.chat(MESSAGES, DocumentExtraction)

    assert second.attempts == 1
    assert [r.response_format for r in fake_llm.requests] == [
        "json_schema",
        "json_object",
        "json_object",
    ]


# ---------------------------------------------------------------- repair


async def test_malformed_json_is_repaired_over_a_real_connection(fake_llm: FakeLLMServer) -> None:
    fake_llm.enqueue(chat_reply(BAD_TEXT))  # the next reply after this one is the canned answer
    async with make_client(fake_llm, Sleeps()) as client:
        result = await client.chat(MESSAGES, DocumentExtraction)

    assert result.repairs == 1
    assert result.attempts == 2
    assert result.parsed is not None
    assert result.parsed.total.value == Decimal("17.28")
    assert result.usage == Usage(
        input_tokens=3000, output_tokens=1400, total_tokens=4400, reported=True
    )

    repair = fake_llm.requests[1].body["messages"]
    assert repair[-2] == {"role": "assistant", "content": BAD_TEXT}
    assert repair[-1]["role"] == "user"
    assert "not valid JSON" in repair[-1]["content"]
    assert repair[:2] == fake_llm.requests[0].body["messages"][:2]  # original prompt intact


async def test_schema_mismatch_is_repaired_and_the_prompt_names_the_problem(
    fake_llm: FakeLLMServer,
) -> None:
    fake_llm.enqueue(chat_reply({"vendor_name": "only this"}))
    async with make_client(fake_llm, Sleeps()) as client:
        result = await client.chat(MESSAGES, DocumentExtraction)

    assert result.repairs == 1
    assert (
        "does not match DocumentExtraction" in fake_llm.requests[1].body["messages"][-1]["content"]
    )


async def test_repair_exhausted_raises_with_everything_that_was_spent(
    fake_llm: FakeLLMServer,
) -> None:
    fake_llm.enqueue(chat_reply(BAD_TEXT), chat_reply(BAD_TEXT + " (again)"))
    async with make_client(fake_llm, Sleeps(), llm_max_repairs=1) as client:
        with pytest.raises(ResponseParseError) as exc:
            await client.chat(MESSAGES, DocumentExtraction)

    assert exc.value.repairs == 1
    assert exc.value.raw_text == BAD_TEXT + " (again)"
    assert exc.value.attempts == 2
    assert exc.value.usage == Usage(
        input_tokens=3000, output_tokens=1400, total_tokens=4400, reported=True
    )
    assert len(fake_llm.requests) == 2


# ---------------------------------------------------------------- retries and backoff


async def test_429_backs_off_using_retry_after_then_succeeds(fake_llm: FakeLLMServer) -> None:
    fake_llm.enqueue(Reply(429, {"error": {"message": "slow down"}}, {"Retry-After": "2"}))
    sleeps = Sleeps()
    async with make_client(fake_llm, sleeps) as client:
        result = await client.chat(MESSAGES, DocumentExtraction)

    assert sleeps.delays == [2.0]
    assert result.attempts == 2
    assert result.parsed is not None
    assert fake_llm.requests[0].body == fake_llm.requests[1].body  # the retry is identical


async def test_5xx_is_retried_with_exponential_backoff(fake_llm: FakeLLMServer) -> None:
    fake_llm.enqueue(Reply(503, "overloaded"), Reply(502, "bad gateway"))
    sleeps = Sleeps()
    async with make_client(fake_llm, sleeps) as client:
        result = await client.chat(MESSAGES, DocumentExtraction)

    assert sleeps.delays == [1.0, 2.0]
    assert result.attempts == 3


async def test_persistent_5xx_gives_up_with_the_attempt_count(fake_llm: FakeLLMServer) -> None:
    fake_llm.enqueue(*[Reply(500, "boom") for _ in range(3)])
    async with make_client(fake_llm, Sleeps(), llm_max_retries=2) as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(MESSAGES, DocumentExtraction)

    assert exc.value.status_code == 500
    assert exc.value.attempts == 3
    assert exc.value.usage is None  # nothing was spent


async def test_a_dropped_connection_is_retried(fake_llm: FakeLLMServer) -> None:
    fake_llm.enqueue(Reply(drop=True))
    sleeps = Sleeps()
    async with make_client(fake_llm, sleeps) as client:
        result = await client.chat(MESSAGES, DocumentExtraction)

    assert result.attempts == 2
    assert sleeps.delays == [1.0]
    assert len(fake_llm.requests) == 2
    assert result.parsed is not None


async def test_a_persistently_dropped_connection_is_a_provider_error(
    fake_llm: FakeLLMServer,
) -> None:
    fake_llm.enqueue(Reply(drop=True), Reply(drop=True))
    sleeps = Sleeps()
    async with make_client(fake_llm, sleeps, llm_max_retries=1) as client:
        with pytest.raises(ProviderError, match="failed after 2 attempt") as exc:
            await client.chat(MESSAGES, DocumentExtraction)

    assert exc.value.attempts == 2
    assert exc.value.status_code is None
    assert exc.value.usage is None  # nothing was spent
    assert len(sleeps.delays) == 1


# ---------------------------------------------------------------- reasoning tokens


async def test_google_style_usage_over_the_wire_is_reconciled_and_costed(
    fake_llm: FakeLLMServer,
) -> None:
    # completion_tokens (700) leaves out 300 reasoning tokens that total_tokens still counts
    fake_llm.enqueue(
        chat_reply(
            canned_answer(),
            usage={"prompt_tokens": 1500, "completion_tokens": 700, "total_tokens": 2500},
        )
    )
    async with make_client(
        fake_llm,
        Sleeps(),
        llm_price_input_per_mtok="0.30",
        llm_price_output_per_mtok="2.50",
    ) as client:
        result = await client.chat(MESSAGES, DocumentExtraction)

    assert result.usage.reasoning_tokens == 300
    assert result.usage.reasoning_inferred is True
    assert result.usage.billable_output_tokens == 1000
    assert result.cost_usd == Decimal("0.00295")  # 1500 x 0.30/M + 1000 x 2.50/M


async def test_reasoning_effort_reaches_the_wire(fake_llm: FakeLLMServer) -> None:
    async with make_client(fake_llm, Sleeps(), llm_reasoning_effort="low") as client:
        await client.chat(MESSAGES, DocumentExtraction)
    assert fake_llm.requests[0].body["reasoning_effort"] == "low"


async def test_reasoning_effort_is_absent_from_the_wire_when_unset(fake_llm: FakeLLMServer) -> None:
    async with make_client(fake_llm, Sleeps()) as client:
        await client.chat(MESSAGES, DocumentExtraction)
    assert "reasoning_effort" not in fake_llm.requests[0].body


# ---------------------------------------------------------------- smoke script end to end


def run_smoke(
    server_url: str | None, tmp_path: Path, **env_extra: str
) -> subprocess.CompletedProcess[str]:
    """Run scripts/smoke.py in a subprocess. Always with an EMPTY env file and a scrubbed
    environment, so it can never pick up the real .env (and with it the real API key)."""
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("LLM_", "MAX_RUN_COST"))
    }
    if server_url is not None:
        env.update(
            LLM_BASE_URL=server_url,
            LLM_MODEL="fake-model",
            LLM_API_KEY="sk-not-real",
            LLM_PRICE_INPUT_PER_MTOK="0.30",
            LLM_PRICE_OUTPUT_PER_MTOK="2.50",
        )
    env.update(env_extra)
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "smoke.py"),
            str(FIXTURE),
            "--env-file",
            str(empty_env),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=ROOT,
        timeout=120,
    )


def test_smoke_script_runs_end_to_end_with_fallback(
    fake_llm: FakeLLMServer, tmp_path: Path
) -> None:
    fake_llm.reject_json_schema = True
    done = run_smoke(fake_llm.base_url, tmp_path)

    assert done.returncode == 0, done.stderr
    # the JSON the script printed is a valid extraction
    printed = done.stdout[done.stdout.index("{") : done.stdout.rindex("}") + 1]
    assert DocumentExtraction.model_validate_json(printed).total.value == Decimal("17.28")
    assert "tokens in/out : 1500 / 700" in done.stdout
    assert "cost          : $0.002200" in done.stdout  # 1500 x 0.30/M + 700 x 2.50/M
    assert "2 attempt(s)" in done.stdout
    assert "structured    : json_object" in done.stdout
    assert "sk-not-real" not in done.stdout + done.stderr


def test_smoke_script_shows_reasoning_tokens_separately_and_costs_them(
    fake_llm: FakeLLMServer, tmp_path: Path
) -> None:
    fake_llm.enqueue(
        chat_reply(
            canned_answer(),
            usage={"prompt_tokens": 1500, "completion_tokens": 700, "total_tokens": 2500},
        )
    )
    done = run_smoke(fake_llm.base_url, tmp_path, LLM_REASONING_EFFORT="low")

    assert done.returncode == 0, done.stderr
    assert "tokens in/out : 1500 / 1000" in done.stdout  # 700 completion + 300 reasoning
    assert "output = 700 completion + 300 reasoning" in done.stdout
    assert "300 tokens, inferred from total_tokens" in done.stdout
    assert "cost          : $0.002950" in done.stdout
    assert fake_llm.requests[0].body["reasoning_effort"] == "low"  # env var -> wire


def test_smoke_script_reports_a_failed_call_cleanly(
    fake_llm: FakeLLMServer, tmp_path: Path
) -> None:
    fake_llm.enqueue(Reply(401, {"error": {"message": "invalid key"}}))
    done = run_smoke(fake_llm.base_url, tmp_path)

    assert done.returncode == 1
    assert "LLM call failed" in done.stderr
    assert "HTTP 401" in done.stderr
    assert "Traceback" not in done.stderr
    assert "sk-not-real" not in done.stdout + done.stderr


def test_smoke_script_reports_unusable_model_output_cleanly(
    fake_llm: FakeLLMServer, tmp_path: Path
) -> None:
    fake_llm.enqueue(chat_reply(BAD_TEXT), chat_reply(BAD_TEXT))
    done = run_smoke(fake_llm.base_url, tmp_path)

    assert done.returncode == 1
    assert "Model output was not usable" in done.stderr
    assert BAD_TEXT in done.stderr  # the raw output is shown
    assert "Traceback" not in done.stderr


def test_smoke_script_without_configuration_exits_2(tmp_path: Path) -> None:
    done = run_smoke(None, tmp_path)

    assert done.returncode == 2
    assert "LLM_MODEL" in done.stderr
    assert "LLM_BASE_URL" in done.stderr
    assert "Traceback" not in done.stderr
