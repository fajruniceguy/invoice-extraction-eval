import asyncio
import json
import logging
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, create_model

from docextract.config import Settings
from docextract.extract.schema import Field as ExtractedField
from docextract.providers.budget import BudgetExceededError, RunBudget
from docextract.providers.openai_compat import (
    OpenAICompatClient,
    ProviderError,
    ResponseParseError,
    Usage,
    build_image_message,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
USER = [{"role": "user", "content": "extract"}]
API_KEY = "sk-test-secret"


class Receipt(BaseModel):
    vendor: str
    total: Decimal


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_base_url": "https://llm.test/v1/",
        "llm_model": "test-model",
        "llm_api_key": API_KEY,
        "llm_backoff_base_s": 1.0,
        "llm_backoff_max_s": 30.0,
        "llm_max_retries": 3,
        "llm_max_repairs": 0,  # most tests script exactly one reply; repair tests opt in
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def completion(
    content: Any = '{"vendor": "ACME", "total": "9.72"}',
    *,
    usage: Any = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ]
    }
    payload["usage"] = {"prompt_tokens": 100, "completion_tokens": 20} if usage is None else usage
    return payload


def ok(payload: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(200, json=payload if payload is not None else completion())


def failing(status: int, n: int, text: str = "error") -> list[httpx.Response]:
    return [httpx.Response(status, text=text) for _ in range(n)]


class Harness:
    """Records requests and sleeps; replays responses from a script or a handler."""

    def __init__(
        self,
        script: list[Any] | None = None,
        handler: Callable[[httpx.Request], Any] | None = None,
        rng: Callable[[], float] = lambda: 1.0,  # 1.0 = no jitter reduction: delay == ceiling
        budget: RunBudget | None = None,
        **settings_overrides: Any,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []
        self._script = list(script or [])
        self._handler = handler

        async def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            result = self._handler(request) if self._handler else self._script.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        async def fake_sleep(delay: float) -> None:
            self.sleeps.append(delay)

        self.client = OpenAICompatClient(
            make_settings(**settings_overrides),
            transport=httpx.MockTransport(respond),
            sleep=fake_sleep,
            rng=rng,
            budget=budget,
        )

    def body(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


# ---------------------------------------------------------------- happy path


async def test_plain_chat_returns_text_usage_latency_and_cost() -> None:
    h = Harness(
        [ok(completion("hello", usage={"prompt_tokens": 1_000_000, "completion_tokens": 500_000}))],
        llm_price_input_per_mtok="0.30",
        llm_price_output_per_mtok="2.50",
    )
    async with h.client as client:
        result = await client.chat(USER)

    assert result.text == "hello"
    assert result.parsed is None
    assert result.usage == Usage(input_tokens=1_000_000, output_tokens=500_000, reported=True)
    assert result.cost_usd == Decimal("1.55")
    assert result.attempts == 1
    assert result.finish_reason == "stop"
    assert result.structured_mode_used == "none"
    assert result.model == "test-model"
    assert result.latency_ms >= 0
    assert result.raw["choices"][0]["message"]["content"] == "hello"


async def test_request_shape() -> None:
    h = Harness([ok()])
    async with h.client as client:
        await client.chat(USER, temperature=0.2, max_tokens=512)

    request = h.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    body = h.body()
    assert body["model"] == "test-model"
    assert body["messages"] == USER
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 512
    assert "response_format" not in body


async def test_base_url_without_trailing_slash() -> None:
    h = Harness([ok()], llm_base_url="https://llm.test/v1beta/openai")
    async with h.client as client:
        await client.chat(USER)
    assert str(h.requests[0].url) == "https://llm.test/v1beta/openai/chat/completions"


async def test_no_authorization_header_without_a_key() -> None:
    h = Harness([ok()], llm_api_key="")
    async with h.client as client:
        await client.chat(USER)
    assert "authorization" not in h.requests[0].headers


async def test_max_tokens_is_omitted_by_default() -> None:
    h = Harness([ok()])
    async with h.client as client:
        await client.chat(USER)
    assert "max_tokens" not in h.body()


async def test_multi_image_message_is_sent_in_order() -> None:
    h = Harness([ok()])
    message = build_image_message("read these pages", [PNG, JPEG, PNG])
    async with h.client as client:
        await client.chat([message])

    parts = h.body()["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "read these pages"}
    urls = [part["image_url"]["url"] for part in parts[1:]]
    assert len(urls) == 3
    assert [u.split(";")[0] for u in urls] == [
        "data:image/png",
        "data:image/jpeg",
        "data:image/png",
    ]


async def test_content_given_as_list_of_parts_is_joined() -> None:
    parts = [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]
    h = Harness([ok(completion(parts))])
    async with h.client as client:
        assert (await client.chat(USER)).text == "hello"


async def test_missing_usage_is_flagged_not_invented() -> None:
    payload = completion("hi")
    del payload["usage"]
    h = Harness([ok(payload)], llm_price_input_per_mtok="5")
    async with h.client as client:
        result = await client.chat(USER)
    assert result.usage == Usage(input_tokens=0, output_tokens=0, reported=False)
    assert result.cost_usd == Decimal(0)


async def test_latency_covers_the_whole_call() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.1)
        return ok()

    client = OpenAICompatClient(make_settings(), transport=httpx.MockTransport(slow))
    async with client:
        result = await client.chat(USER)
    # sleeps can return early by a timer tick (~15 ms on Windows), so leave real margin
    assert 80 <= result.latency_ms < 5_000


# ---------------------------------------------------------------- structured output


async def test_json_schema_is_tried_first_and_parsed_into_the_model() -> None:
    h = Harness([ok()])
    async with h.client as client:
        result = await client.chat(USER, Receipt)

    response_format = h.body()["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Receipt"
    schema = response_format["json_schema"]["schema"]
    assert set(schema["properties"]) == {"vendor", "total"}
    assert "title" not in schema
    assert result.parsed == Receipt(vendor="ACME", total=Decimal("9.72"))
    assert result.structured_mode_used == "json_schema"
    assert h.body()["messages"] == USER  # the prompt is untouched on the json_schema tier


async def test_falls_back_to_json_object_when_json_schema_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})
        return ok()

    h = Harness(handler=handler)
    async with h.client as client:
        result = await client.chat(USER, Receipt)

    assert result.parsed == Receipt(vendor="ACME", total=Decimal("9.72"))
    assert result.structured_mode_used == "json_object"
    assert result.attempts == 2
    assert [h.body(i)["response_format"]["type"] for i in range(2)] == [
        "json_schema",
        "json_object",
    ]
    fallback_messages = h.body(1)["messages"]
    assert fallback_messages[0]["role"] == "system"
    assert "JSON" in fallback_messages[0]["content"]
    assert '"vendor"' in fallback_messages[0]["content"]  # the schema is spelled out in the prompt
    assert fallback_messages[1:] == USER


async def test_schema_instruction_is_appended_to_an_existing_system_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["response_format"]["type"] == "json_schema":
            return httpx.Response(422, text="nope")
        return ok()

    h = Harness(handler=handler)
    messages = [{"role": "system", "content": "You extract receipts."}, *USER]
    async with h.client as client:
        await client.chat(messages, Receipt)

    sent = h.body(1)["messages"]
    assert len(sent) == 2
    assert sent[0]["content"].startswith("You extract receipts.")
    assert '"vendor"' in sent[0]["content"]
    assert messages[0]["content"] == "You extract receipts."  # caller's list is not mutated


async def test_fallback_is_remembered_for_later_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["response_format"]["type"] == "json_schema":
            return httpx.Response(400, text="unsupported")
        return ok()

    h = Harness(handler=handler)
    async with h.client as client:
        await client.chat(USER, Receipt)
        assert len(h.requests) == 2
        second = await client.chat(USER, Receipt)

    assert len(h.requests) == 3  # one request, not two
    assert h.body(2)["response_format"]["type"] == "json_object"
    assert second.attempts == 1


async def test_fallback_is_not_remembered_when_json_object_fails_too() -> None:
    h = Harness(handler=lambda request: httpx.Response(400, text="bad image"))
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER, Receipt)
        assert exc.value.status_code == 400
        assert exc.value.attempts == 2
        with pytest.raises(ProviderError):
            await client.chat(USER, Receipt)

    # a 400 caused by something else must not permanently downgrade the client
    assert h.body(2)["response_format"]["type"] == "json_schema"


async def test_pinned_json_schema_mode_does_not_fall_back() -> None:
    h = Harness([httpx.Response(400, text="unsupported")], llm_structured_mode="json_schema")
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER, Receipt)
    assert exc.value.attempts == 1
    assert len(h.requests) == 1


async def test_prompt_mode_sends_no_response_format_and_the_schema_in_the_text() -> None:
    h = Harness([ok()], llm_structured_mode="prompt")
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    body = h.body()
    assert "response_format" not in body
    assert body["messages"][0]["role"] == "system" and '"vendor"' in body["messages"][0]["content"]
    assert result.structured_mode_used == "prompt" and len(h.requests) == 1
    assert result.parsed is not None


async def test_prompt_mode_never_falls_back_or_negotiates() -> None:
    h = Harness([httpx.Response(400, text="nope")], llm_structured_mode="prompt")
    async with h.client as client:
        with pytest.raises(ProviderError, match="HTTP 400"):
            await client.chat(USER, Receipt)
    assert len(h.requests) == 1


async def test_repair_in_prompt_mode_keeps_the_schema_instruction_and_no_response_format() -> None:
    h = Harness(
        [ok(completion(BAD_JSON)), ok(completion(GOOD))],
        llm_structured_mode="prompt",
        llm_max_repairs=1,
    )
    async with h.client as client:
        await client.chat(USER, Receipt)
    for index in (0, 1):
        assert "response_format" not in h.body(index)
    assert '"vendor"' in h.body(1)["messages"][0]["content"]


async def test_temperature_is_sent_by_default_and_can_be_left_out() -> None:
    sent = Harness([ok()])
    async with sent.client as client:
        await client.chat(USER, temperature=0.3)
    assert sent.body()["temperature"] == 0.3

    left_out = Harness([ok()], llm_send_temperature=False)
    async with left_out.client as client:
        await client.chat(USER, Receipt, temperature=0.3)
    assert "temperature" not in left_out.body()
    assert left_out.body()["model"] == "test-model"  # nothing else about the body changed


async def test_pinned_json_object_mode_goes_straight_to_json_object() -> None:
    h = Harness([ok()], llm_structured_mode="json_object")
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert h.body()["response_format"] == {"type": "json_object"}
    assert result.structured_mode_used == "json_object"


async def test_non_400_errors_do_not_trigger_the_fallback() -> None:
    h = Harness([httpx.Response(401, text="unauthorized")])
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER, Receipt)
    assert exc.value.status_code == 401
    assert len(h.requests) == 1


# ---------------------------------------------------------------- retries


async def test_429_is_retried_with_exponential_backoff() -> None:
    h = Harness([httpx.Response(429), httpx.Response(429), ok()])
    async with h.client as client:
        result = await client.chat(USER)
    assert result.attempts == 3
    assert h.sleeps == [1.0, 2.0]


async def test_retry_after_header_is_honored() -> None:
    h = Harness([httpx.Response(429, headers={"Retry-After": "2"}), ok()])
    async with h.client as client:
        await client.chat(USER)
    assert h.sleeps == [2.0]


async def test_retry_after_is_capped_by_backoff_max() -> None:
    h = Harness([httpx.Response(429, headers={"Retry-After": "999"}), ok()], llm_backoff_max_s=5)
    async with h.client as client:
        await client.chat(USER)
    assert h.sleeps == [5.0]


async def test_unparseable_retry_after_falls_back_to_backoff() -> None:
    h = Harness(
        [httpx.Response(503, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), ok()]
    )
    async with h.client as client:
        await client.chat(USER)
    assert h.sleeps == [1.0]


async def test_backoff_doubles_then_is_capped() -> None:
    h = Harness(failing(500, 4), llm_max_retries=3, llm_backoff_base_s=10, llm_backoff_max_s=25)
    async with h.client as client:
        with pytest.raises(ProviderError):
            await client.chat(USER)
    assert h.sleeps == [10.0, 20.0, 25.0]


async def test_backoff_jitter_scales_the_delay_between_half_and_full() -> None:
    low = Harness(failing(500, 2), llm_max_retries=1, rng=lambda: 0.0)
    async with low.client as client:
        with pytest.raises(ProviderError):
            await client.chat(USER)
    assert low.sleeps == [0.5]  # rng=0 -> half the ceiling

    high = Harness(failing(500, 2), llm_max_retries=1, rng=lambda: 1.0)
    async with high.client as client:
        with pytest.raises(ProviderError):
            await client.chat(USER)
    assert high.sleeps == [1.0]  # rng=1 -> the full ceiling


async def test_5xx_exhaustion_raises_with_attempt_count() -> None:
    h = Harness(failing(503, 3, "overloaded"), llm_max_retries=2)
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER)
    assert exc.value.status_code == 503
    assert exc.value.attempts == 3
    assert exc.value.latency_ms >= 0
    assert len(h.requests) == 3
    assert len(h.sleeps) == 2  # no sleep after the final attempt


async def test_exhaustion_error_carries_the_providers_own_explanation() -> None:
    body = '{"error": {"code": 503, "message": "The model is overloaded. Please try again later."}}'
    h = Harness(failing(503, 2, body), llm_max_retries=1)
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER)
    message = str(exc.value)
    assert "failed after 2 attempt(s): HTTP 503" in message
    assert "The model is overloaded" in message


async def test_exhaustion_error_shows_the_last_body_not_the_first() -> None:
    h = Harness(
        [httpx.Response(503, text="first"), httpx.Response(503, text="last")], llm_max_retries=1
    )
    async with h.client as client:
        with pytest.raises(ProviderError, match="last") as exc:
            await client.chat(USER)
    assert "first" not in str(exc.value)


async def test_exhaustion_error_with_an_empty_body_has_no_dangling_colon() -> None:
    h = Harness([httpx.Response(503)], llm_max_retries=0)
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER)
    assert str(exc.value) == "LLM call failed after 1 attempt(s): HTTP 503"


async def test_exhaustion_error_body_is_capped() -> None:
    h = Harness(failing(500, 1, "x" * 10_000), llm_max_retries=0)
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER)
    assert len(str(exc.value)) < 700


async def test_a_transport_error_with_an_empty_message_has_no_dangling_colon() -> None:
    h = Harness([httpx.ReadTimeout("")], llm_max_retries=0)
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER)
    assert str(exc.value) == "LLM call failed after 1 attempt(s): ReadTimeout"


async def test_exhaustion_after_transport_errors_has_no_body_to_show() -> None:
    h = Harness([httpx.ReadTimeout("slow")], llm_max_retries=0)
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER)
    assert str(exc.value) == "LLM call failed after 1 attempt(s): ReadTimeout: slow"


async def test_zero_retries_means_one_attempt() -> None:
    h = Harness([httpx.Response(500)], llm_max_retries=0)
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER)
    assert exc.value.attempts == 1
    assert h.sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
async def test_client_errors_are_not_retried(status: int) -> None:
    h = Harness([httpx.Response(status, text="client error")], llm_structured_mode="json_object")
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER, Receipt)
    assert exc.value.status_code == status
    assert exc.value.attempts == 1
    assert h.sleeps == []
    assert str(status) in str(exc.value)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("slow"),
        httpx.ConnectTimeout("slow"),
        httpx.ConnectError("refused"),
        httpx.RemoteProtocolError("dropped"),
    ],
)
async def test_transport_errors_are_retried(error: Exception) -> None:
    h = Harness([error, ok()])
    async with h.client as client:
        result = await client.chat(USER)
    assert result.attempts == 2
    assert h.sleeps == [1.0]


async def test_persistent_timeout_raises_provider_error() -> None:
    h = Harness([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")], llm_max_retries=1)
    async with h.client as client:
        with pytest.raises(ProviderError, match="ReadTimeout") as exc:
            await client.chat(USER)
    assert exc.value.status_code is None
    assert exc.value.attempts == 2


async def test_bad_base_url_scheme_fails_fast() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.UnsupportedProtocol("Request URL has an unsupported protocol")

    h = Harness(handler=handler)
    async with h.client as client:
        with pytest.raises(ProviderError, match="LLM_BASE_URL") as exc:
            await client.chat(USER)
    assert exc.value.attempts == 1
    assert h.sleeps == []


async def test_retries_do_not_change_the_request() -> None:
    h = Harness([httpx.Response(500), ok()])
    async with h.client as client:
        await client.chat(USER, Receipt)
    assert h.body(0) == h.body(1)


# ---------------------------------------------------------------- bad responses


async def test_invalid_json_raises_parse_error_that_keeps_text_and_cost() -> None:
    h = Harness([ok(completion("I cannot read this receipt."))])
    async with h.client as client:
        with pytest.raises(ResponseParseError) as exc:
            await client.chat(USER, Receipt)
    assert exc.value.raw_text == "I cannot read this receipt."
    assert exc.value.usage == Usage(input_tokens=100, output_tokens=20, reported=True)
    assert exc.value.attempts == 1
    assert exc.value.latency_ms >= 0


async def test_truncated_output_is_called_out() -> None:
    h = Harness([ok(completion('{"vendor": "ACM', finish_reason="length"))])
    async with h.client as client:
        with pytest.raises(ResponseParseError, match="truncated"):
            await client.chat(USER, Receipt)


async def test_schema_mismatch_names_the_failing_field() -> None:
    h = Harness([ok(completion('{"vendor": "ACME", "total": "lots"}'))])
    async with h.client as client:
        with pytest.raises(ResponseParseError, match=r"total") as exc:
            await client.chat(USER, Receipt)
    assert "Receipt" in str(exc.value)
    assert exc.value.raw_text == '{"vendor": "ACME", "total": "lots"}'


async def test_fenced_json_is_accepted() -> None:
    h = Harness([ok(completion('```json\n{"vendor": "ACME", "total": 9.72}\n```'))])
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.parsed == Receipt(vendor="ACME", total=Decimal("9.72"))


class Totals(BaseModel):
    subtotal: Decimal
    tax: Decimal
    total: Decimal
    confidence: float
    page: int


class MoneyField(BaseModel):
    total: ExtractedField[Decimal]


async def test_money_reaches_the_model_exactly_and_non_money_fields_still_validate() -> None:
    content = '{"subtotal": 0.1, "tax": 0.2, "total": 1250000.10, "confidence": 0.95, "page": 1}'
    h = Harness([ok(completion(content))])
    async with h.client as client:
        result = await client.chat(USER, Totals)

    parsed = result.parsed
    assert parsed is not None
    assert parsed.subtotal + parsed.tax == Decimal("0.3")
    assert parsed.total == Decimal("1250000.10")
    assert parsed.total.as_tuple().exponent == -2  # scale preserved: not 1250000.1
    assert parsed.confidence == 0.95
    assert type(parsed.confidence) is float
    assert parsed.page == 1
    assert type(parsed.page) is int


async def test_decimal_survives_inside_a_field_wrapper() -> None:
    content = (
        '{"total": {"value": 12345678901234567.89, "confidence": 0.9,'
        ' "status": "extracted", "page": 1}}'
    )
    h = Harness([ok(completion(content))])
    async with h.client as client:
        result = await client.chat(USER, MoneyField)
    assert result.parsed is not None
    assert result.parsed.total.value == Decimal("12345678901234567.89")  # beyond float precision
    assert result.parsed.total.confidence == 0.9


async def test_non_integral_page_is_still_rejected_after_decimal_parsing() -> None:
    content = '{"subtotal": 1, "tax": 1, "total": 2, "confidence": 0.5, "page": 1.5}'
    h = Harness([ok(completion(content))])
    async with h.client as client:
        with pytest.raises(ResponseParseError, match="page"):
            await client.chat(USER, Totals)


async def test_response_without_choices_is_a_parse_error_with_usage() -> None:
    h = Harness([httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 5}})])
    async with h.client as client:
        with pytest.raises(ResponseParseError, match="no choices") as exc:
            await client.chat(USER)
    assert exc.value.usage == Usage(input_tokens=5, output_tokens=0, reported=True)


async def test_empty_content_is_a_parse_error_when_a_schema_is_requested() -> None:
    h = Harness([ok(completion(None))])
    async with h.client as client:
        with pytest.raises(ResponseParseError, match="empty"):
            await client.chat(USER, Receipt)


async def test_non_json_http_body_is_a_parse_error() -> None:
    h = Harness([httpx.Response(200, text="<html>gateway</html>")])
    async with h.client as client:
        with pytest.raises(ResponseParseError, match="non-JSON") as exc:
            await client.chat(USER)
    assert exc.value.raw_text == "<html>gateway</html>"


# ---------------------------------------------------------------- reasoning tokens

# $1 per input token and $2 per output token keeps the arithmetic obvious
DOLLAR_PRICES: dict[str, Any] = {
    "llm_price_input_per_mtok": "1000000",
    "llm_price_output_per_mtok": "2000000",
}


def usage_reply(prompt: int, completion: int, total: int | None, reasoning: int | None) -> Any:
    block: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion}
    if total is not None:
        block["total_tokens"] = total
    if reasoning is not None:
        block["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return ok(completion_payload(block))


def completion_payload(
    usage: dict[str, Any], content: str = '{"vendor": "ACME", "total": "9.72"}'
) -> dict[str, Any]:
    payload = completion(content)
    payload["usage"] = usage
    return payload


async def test_openai_style_cost_does_not_count_reasoning_twice() -> None:
    # 100 in, 300 out of which 200 are reasoning (already inside completion_tokens)
    h = Harness([usage_reply(100, 300, total=400, reasoning=200)], **DOLLAR_PRICES)
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.usage.reasoning_tokens == 200
    assert result.usage.billable_output_tokens == 300
    assert result.cost_usd == Decimal(100 * 1 + 300 * 2)  # not 100 + 500 * 2


async def test_google_style_cost_adds_the_reasoning_the_provider_left_out() -> None:
    # completion_tokens (100) excludes the 200 reasoning tokens; total_tokens includes them
    h = Harness([usage_reply(100, 100, total=400, reasoning=200)], **DOLLAR_PRICES)
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.usage.output_tokens == 100
    assert result.usage.reasoning_tokens == 200
    assert result.usage.billable_output_tokens == 300
    assert result.cost_usd == Decimal(100 * 1 + 300 * 2)


async def test_unreported_reasoning_is_inferred_and_costed() -> None:
    h = Harness([usage_reply(100, 100, total=500, reasoning=None)], **DOLLAR_PRICES)
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.usage.reasoning_tokens == 300
    assert result.usage.reasoning_inferred is True
    assert result.cost_usd == Decimal(100 * 1 + 400 * 2)


async def test_without_total_tokens_cost_uses_completion_tokens_as_reported() -> None:
    h = Harness([usage_reply(100, 300, total=None, reasoning=200)], **DOLLAR_PRICES)
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.usage.output_includes_reasoning is None  # unverified
    assert result.cost_usd == Decimal(100 * 1 + 300 * 2)


async def test_reasoning_is_totalled_across_a_repair_and_each_call_is_judged_alone() -> None:
    h = Harness(
        [
            ok(
                completion_payload(
                    {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 400}, BAD_JSON
                )
            ),
            ok(
                completion_payload(
                    {
                        "prompt_tokens": 10,
                        "completion_tokens": 30,
                        "total_tokens": 40,
                        "completion_tokens_details": {"reasoning_tokens": 20},
                    },
                    GOOD,
                )
            ),
        ],
        llm_max_repairs=1,
        **DOLLAR_PRICES,
    )
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.usage.input_tokens == 110
    assert result.usage.reasoning_tokens == 220  # 200 inferred + 20 reported
    assert result.usage.billable_output_tokens == 130 + 200
    assert result.cost_usd == Decimal(110 * 1 + 330 * 2)


async def test_reasoning_spend_survives_in_errors_too() -> None:
    h = Harness(
        [
            ok(
                completion_payload(
                    {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 400}, BAD_JSON
                )
            )
        ],
        llm_max_repairs=0,
        **DOLLAR_PRICES,
    )
    async with h.client as client:
        with pytest.raises(ResponseParseError) as exc:
            await client.chat(USER, Receipt)
    assert exc.value.usage is not None
    assert exc.value.usage.reasoning_outside_output == 200
    assert exc.value.usage.cost_usd(make_settings(**DOLLAR_PRICES)) == Decimal(100 * 1 + 300 * 2)


# ---------------------------------------------------------------- reasoning effort


async def test_reasoning_effort_is_not_sent_by_default() -> None:
    h = Harness([ok()])
    async with h.client as client:
        await client.chat(USER, Receipt)
    assert "reasoning_effort" not in h.body()


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high"])
async def test_reasoning_effort_is_passed_through_when_set(effort: str) -> None:
    h = Harness([ok()], llm_reasoning_effort=effort)
    async with h.client as client:
        await client.chat(USER)
    assert h.body()["reasoning_effort"] == effort


async def test_blank_reasoning_effort_counts_as_unset() -> None:
    h = Harness([ok()], llm_reasoning_effort="   ")
    async with h.client as client:
        await client.chat(USER)
    assert "reasoning_effort" not in h.body()


async def test_reasoning_effort_is_sent_on_fallback_and_repair_requests_too() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, text="unsupported")
        return ok(completion(BAD_JSON if len(h.requests) <= 2 else GOOD))

    h = Harness(handler=handler, llm_reasoning_effort="low", llm_max_repairs=1)
    async with h.client as client:
        await client.chat(USER, Receipt)
    assert len(h.requests) == 3  # json_schema (rejected), json_object, repair
    assert [h.body(i)["reasoning_effort"] for i in range(3)] == ["low", "low", "low"]


async def test_reasoning_effort_is_resent_on_retries() -> None:
    h = Harness([httpx.Response(503), ok()], llm_reasoning_effort="high")
    async with h.client as client:
        await client.chat(USER)
    assert [h.body(i)["reasoning_effort"] for i in range(2)] == ["high", "high"]


# ---------------------------------------------------------------- repair

GOOD = '{"vendor": "ACME", "total": "9.72"}'
BAD_JSON = "Sure! The receipt total is 9.72 from ACME."
BAD_SCHEMA = '{"vendor": "ACME", "total": "lots"}'


async def test_malformed_json_is_repaired_with_one_more_call() -> None:
    h = Harness([ok(completion(BAD_JSON)), ok(completion(GOOD))], llm_max_repairs=1)
    async with h.client as client:
        result = await client.chat(USER, Receipt)

    assert result.parsed == Receipt(vendor="ACME", total=Decimal("9.72"))
    assert result.repairs == 1
    assert result.attempts == 2
    assert result.text == GOOD  # the text of the call that finally worked
    # the repair re-asks with: original messages, the bad output, then the problem
    sent = h.body(1)["messages"]
    assert sent[0] == USER[0]
    assert sent[1] == {"role": "assistant", "content": BAD_JSON}
    assert sent[2]["role"] == "user"
    assert "not valid JSON" in sent[2]["content"]
    assert "corrected JSON" in sent[2]["content"]
    assert len(sent) == 3


async def test_schema_mismatch_is_repaired_and_the_prompt_names_the_field() -> None:
    h = Harness([ok(completion(BAD_SCHEMA)), ok(completion(GOOD))], llm_max_repairs=1)
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.parsed is not None
    assert result.repairs == 1
    prompt = h.body(1)["messages"][-1]["content"]
    assert "does not match Receipt" in prompt
    assert "total" in prompt


async def test_repair_totals_usage_cost_and_attempts_across_calls() -> None:
    h = Harness(
        [
            ok(completion(BAD_JSON)),
            ok(completion(GOOD, usage={"prompt_tokens": 300, "completion_tokens": 50})),
        ],
        llm_max_repairs=1,
        llm_price_input_per_mtok="1000000",  # $1 per token keeps the arithmetic obvious
        llm_price_output_per_mtok="2000000",
    )
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.usage == Usage(input_tokens=400, output_tokens=70, reported=True)
    assert result.cost_usd == Decimal(400 * 1 + 70 * 2)


async def test_repair_exhausted_raises_with_aggregated_totals_and_last_text() -> None:
    h = Harness([ok(completion(BAD_JSON)), ok(completion(BAD_SCHEMA))], llm_max_repairs=1)
    async with h.client as client:
        with pytest.raises(ResponseParseError) as exc:
            await client.chat(USER, Receipt)
    assert len(h.requests) == 2  # one original + exactly one repair
    assert exc.value.raw_text == BAD_SCHEMA  # the LAST output
    assert exc.value.repairs == 1
    assert exc.value.attempts == 2
    assert exc.value.usage == Usage(input_tokens=200, output_tokens=40, reported=True)


async def test_max_repairs_zero_never_re_asks() -> None:
    h = Harness([ok(completion(BAD_JSON))], llm_max_repairs=0)
    async with h.client as client:
        with pytest.raises(ResponseParseError) as exc:
            await client.chat(USER, Receipt)
    assert len(h.requests) == 1
    assert exc.value.repairs == 0


async def test_conversation_does_not_grow_across_several_repairs() -> None:
    h = Harness(
        [ok(completion(BAD_JSON)), ok(completion(BAD_SCHEMA)), ok(completion(GOOD))],
        llm_max_repairs=2,
    )
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.repairs == 2
    assert len(h.requests) == 3
    third = h.body(2)["messages"]
    assert len(third) == 3  # original + last bad output + problem, not the whole history
    assert third[1]["content"] == BAD_SCHEMA  # the most recent bad output, not the first


async def test_truncated_output_is_not_repaired() -> None:
    h = Harness([ok(completion('{"vendor": "AC', finish_reason="length"))], llm_max_repairs=3)
    async with h.client as client:
        with pytest.raises(ResponseParseError, match="truncated"):
            await client.chat(USER, Receipt)
    assert len(h.requests) == 1


async def test_plain_chat_is_never_repaired() -> None:
    h = Harness([ok(completion(BAD_JSON))], llm_max_repairs=3)
    async with h.client as client:
        result = await client.chat(USER)
    assert result.text == BAD_JSON
    assert result.repairs == 0
    assert len(h.requests) == 1


async def test_a_clean_first_reply_makes_no_repair_call() -> None:
    h = Harness([ok(completion(GOOD))], llm_max_repairs=3)
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.repairs == 0
    assert len(h.requests) == 1


async def test_http_failure_during_a_repair_still_reports_the_first_calls_spend() -> None:
    h = Harness(
        [ok(completion(BAD_JSON)), *failing(503, 3)],
        llm_max_repairs=1,
        llm_max_retries=2,
    )
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER, Receipt)
    assert not isinstance(exc.value, ResponseParseError)
    assert exc.value.status_code == 503
    assert exc.value.attempts == 4  # 1 original + 3 tries at the repair
    assert exc.value.usage == Usage(input_tokens=100, output_tokens=20, reported=True)


async def test_http_failure_with_nothing_spent_has_no_usage() -> None:
    h = Harness(failing(503, 1), llm_max_retries=0)
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER, Receipt)
    assert exc.value.usage is None


async def test_repair_goes_straight_to_the_tier_that_worked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, text="unsupported")
        return ok(completion(BAD_JSON if len(h.requests) <= 2 else GOOD))

    h = Harness(handler=handler, llm_max_repairs=1)
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert [h.body(i)["response_format"]["type"] for i in range(3)] == [
        "json_schema",  # rejected
        "json_object",  # fallback, bad output
        "json_object",  # repair does not retry json_schema
    ]
    assert result.repairs == 1
    assert result.structured_mode_used == "json_object"
    assert result.attempts == 3


async def test_repair_in_json_object_mode_keeps_the_schema_instruction() -> None:
    h = Harness(
        [ok(completion(BAD_JSON)), ok(completion(GOOD))],
        llm_structured_mode="json_object",
        llm_max_repairs=1,
    )
    async with h.client as client:
        await client.chat(USER, Receipt)
    repair_messages = h.body(1)["messages"]
    assert repair_messages[0]["role"] == "system"
    assert '"vendor"' in repair_messages[0]["content"]
    assert repair_messages[-1]["role"] == "user"


async def test_the_failure_a_repair_papered_over_is_recorded() -> None:
    h = Harness([ok(completion(BAD_SCHEMA)), ok(completion(GOOD))], llm_max_repairs=1)
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.repairs == 1
    (record,) = result.repair_history
    assert record.raw_text == BAD_SCHEMA
    assert "does not match Receipt" in record.problem
    assert "total" in record.problem  # which field failed


async def test_there_is_no_history_when_nothing_was_repaired() -> None:
    h = Harness([ok(completion(GOOD))], llm_max_repairs=1)
    async with h.client as client:
        assert (await client.chat(USER, Receipt)).repair_history == ()
    h = Harness([ok(completion(BAD_JSON))], llm_max_repairs=1)
    async with h.client as client:
        assert (await client.chat(USER)).repair_history == ()  # plain chat is never repaired


async def test_history_lists_every_repaired_failure_oldest_first() -> None:
    h = Harness(
        [ok(completion(BAD_JSON)), ok(completion(BAD_SCHEMA)), ok(completion(GOOD))],
        llm_max_repairs=2,
    )
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert [record.raw_text for record in result.repair_history] == [BAD_JSON, BAD_SCHEMA]
    assert "not valid JSON" in result.repair_history[0].problem
    assert "does not match Receipt" in result.repair_history[1].problem


async def test_exhausted_repairs_attach_the_earlier_failures_to_the_error() -> None:
    h = Harness([ok(completion(BAD_JSON)), ok(completion(BAD_SCHEMA))], llm_max_repairs=1)
    async with h.client as client:
        with pytest.raises(ResponseParseError) as exc:
            await client.chat(USER, Receipt)
    assert exc.value.raw_text == BAD_SCHEMA  # the final failure is the exception itself...
    assert [r.raw_text for r in exc.value.history] == [BAD_JSON]  # ...earlier ones are history
    assert "not valid JSON" in exc.value.history[0].problem


async def test_a_parse_error_with_no_repairs_has_an_empty_history() -> None:
    h = Harness([ok(completion(BAD_JSON))], llm_max_repairs=0)
    async with h.client as client:
        with pytest.raises(ResponseParseError) as exc:
            await client.chat(USER, Receipt)
    assert exc.value.history == ()


async def test_history_is_kept_when_a_repair_reply_is_an_unusable_body() -> None:
    h = Harness(
        [ok(completion(BAD_JSON)), httpx.Response(200, text="<html>gateway</html>")],
        llm_max_repairs=1,
    )
    async with h.client as client:
        with pytest.raises(ResponseParseError, match="non-JSON") as exc:
            await client.chat(USER, Receipt)
    assert exc.value.repairs == 1
    assert [r.raw_text for r in exc.value.history] == [BAD_JSON]


async def test_callers_message_list_is_not_mutated_by_repairs() -> None:
    h = Harness([ok(completion(BAD_JSON)), ok(completion(GOOD))], llm_max_repairs=1)
    messages = [dict(m) for m in USER]
    async with h.client as client:
        await client.chat(messages, Receipt)
    assert messages == USER


async def test_very_long_error_text_is_capped_in_the_repair_prompt() -> None:
    # the error text names each failing field; give the schema huge field names so the
    # message really is enormous (garbage model output alone gives a short error)
    huge = create_model("Huge", **{f"{letter}" * 2_000: (int, ...) for letter in "abcde"})
    h = Harness([ok(completion("{}")), ok(completion("{}"))], llm_max_repairs=1)
    async with h.client as client:
        with pytest.raises(ResponseParseError) as exc:
            await client.chat(USER, huge)
    assert len(str(exc.value)) > 5_000  # the premise: the uncapped message is huge
    prompt = h.body(1)["messages"][-1]["content"]
    assert len(prompt) < 1_500  # but what goes back to the model is capped


# ---------------------------------------------------------------- run budget

# with DOLLAR_PRICES one default completion() (100 in, 20 out) costs 100 x $1 + 20 x $2 = $140


async def test_calls_are_refused_once_spend_exceeds_the_cap() -> None:
    h = Harness([ok(), ok(), ok()], max_run_cost_usd="300", **DOLLAR_PRICES)
    async with h.client as client:
        await client.chat(USER)  # $140
        await client.chat(USER)  # $280, still within the cap
        await client.chat(USER)  # $420: this call completes, but crosses the line
        with pytest.raises(BudgetExceededError) as exc:
            await client.chat(USER)
    assert len(h.requests) == 3  # the refused call never reached the server
    assert exc.value.spent == 420
    assert exc.value.limit == 300
    assert h.client.budget.exceeded


async def test_spending_exactly_the_cap_is_still_allowed() -> None:
    h = Harness([ok(), ok(), ok()], max_run_cost_usd="280", **DOLLAR_PRICES)
    async with h.client as client:
        await client.chat(USER)
        await client.chat(USER)
        assert h.client.budget.spent == 280
        assert not h.client.budget.exceeded
        await client.chat(USER)  # allowed: 280 is not over 280
        with pytest.raises(BudgetExceededError):
            await client.chat(USER)
    assert len(h.requests) == 3


async def test_the_call_that_crosses_the_cap_still_returns_its_result() -> None:
    h = Harness([ok()], max_run_cost_usd="100", **DOLLAR_PRICES)
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.parsed == Receipt(vendor="ACME", total=Decimal("9.72"))
    assert result.cost_usd == 140
    assert h.client.budget.exceeded


async def test_a_repair_is_refused_when_the_first_call_already_exceeded_the_cap() -> None:
    h = Harness(
        [ok(completion(BAD_JSON)), ok(completion(GOOD))],
        max_run_cost_usd="100",
        llm_max_repairs=1,
        **DOLLAR_PRICES,
    )
    async with h.client as client:
        with pytest.raises(BudgetExceededError) as exc:
            await client.chat(USER, Receipt)
    assert len(h.requests) == 1  # no repair request was made
    # the parse error it was trying to fix is kept as the cause, with the raw output
    assert isinstance(exc.value.__cause__, ResponseParseError)
    assert exc.value.__cause__.raw_text == BAD_JSON


async def test_a_repair_within_the_cap_goes_ahead() -> None:
    h = Harness(
        [ok(completion(BAD_JSON)), ok(completion(GOOD))],
        max_run_cost_usd="1000",
        llm_max_repairs=1,
        **DOLLAR_PRICES,
    )
    async with h.client as client:
        result = await client.chat(USER, Receipt)
    assert result.repairs == 1
    assert h.client.budget.spent == 280
    assert h.client.budget.calls == 2  # each completed call is recorded on its own


async def test_unparseable_calls_still_count_toward_the_cap() -> None:
    h = Harness(
        [ok(completion(BAD_JSON))], max_run_cost_usd="1000", llm_max_repairs=0, **DOLLAR_PRICES
    )
    async with h.client as client:
        with pytest.raises(ResponseParseError):
            await client.chat(USER, Receipt)
    assert h.client.budget.spent == 140
    assert h.client.budget.calls == 1


async def test_calls_that_failed_before_any_usage_cost_nothing() -> None:
    h = Harness(failing(503, 1), max_run_cost_usd="100", llm_max_retries=0, **DOLLAR_PRICES)
    async with h.client as client:
        with pytest.raises(ProviderError):
            await client.chat(USER)
    assert h.client.budget.spent == 0
    assert h.client.budget.calls == 0


async def test_reasoning_tokens_count_toward_the_cap() -> None:
    # Google-style: 100 in, 100 out, 200 reasoning on top -> 100 + 300 x 2 = $700
    h = Harness(
        [usage_reply(100, 100, total=400, reasoning=200)], max_run_cost_usd="500", **DOLLAR_PRICES
    )
    async with h.client as client:
        await client.chat(USER, Receipt)
    assert h.client.budget.spent == 700
    assert h.client.budget.exceeded


async def test_a_budget_can_be_shared_by_several_clients() -> None:
    budget = RunBudget(Decimal(200))
    first = Harness([ok(), ok()], budget=budget, **DOLLAR_PRICES)
    second = Harness([ok()], budget=budget, **DOLLAR_PRICES)
    async with first.client as a, second.client as b:
        await a.chat(USER)  # $140
        await b.chat(USER)  # $280 in total: over
        with pytest.raises(BudgetExceededError):
            await a.chat(USER)
        with pytest.raises(BudgetExceededError):
            await b.chat(USER)
    assert budget.spent == 280
    assert len(first.requests) == 1
    assert len(second.requests) == 1


async def test_the_budget_is_built_from_settings_and_tracks_spend_even_without_a_cap() -> None:
    capped = Harness([ok()], max_run_cost_usd="5", **DOLLAR_PRICES)
    assert capped.client.budget.limit == Decimal(5)

    uncapped = Harness([ok(), ok()], **DOLLAR_PRICES)
    async with uncapped.client as client:
        await client.chat(USER)
        await client.chat(USER)
    assert uncapped.client.budget.limit is None
    assert uncapped.client.budget.spent == 280  # tracked for reporting, never enforced
    await capped.client.aclose()


async def test_responses_without_usage_are_counted_unmetered_and_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = completion("hi")
    del payload["usage"]
    h = Harness([ok(payload)], max_run_cost_usd="100", **DOLLAR_PRICES)
    with caplog.at_level(logging.WARNING, logger="docextract.providers.openai_compat"):
        async with h.client as client:
            await client.chat(USER)
    assert h.client.budget.unmetered_calls == 1
    assert h.client.budget.calls == 0
    assert h.client.budget.spent == 0
    assert "no usage block" in caplog.text


async def test_no_missing_usage_warning_when_there_is_no_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = completion("hi")
    del payload["usage"]
    h = Harness([ok(payload)])
    with caplog.at_level(logging.WARNING, logger="docextract.providers.openai_compat"):
        async with h.client as client:
            await client.chat(USER)
    assert "no usage block" not in caplog.text


async def test_a_loop_that_swallows_provider_errors_cannot_swallow_the_cap() -> None:
    h = Harness([ok(), ok()], max_run_cost_usd="100", **DOLLAR_PRICES)
    swallowed = 0
    async with h.client as client:
        await client.chat(USER)  # $140: over the cap
        with pytest.raises(BudgetExceededError):
            try:
                await client.chat(USER)
            except ProviderError:
                swallowed += 1
    assert swallowed == 0


# ---------------------------------------------------------------- response cache


def cached(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Settings overrides that turn the cache on in a throwaway directory (tests default off)."""
    return {"llm_cache": "on", "llm_cache_dir": tmp_path / "cache", **DOLLAR_PRICES, **overrides}


async def test_a_second_identical_call_is_served_from_the_cache(tmp_path: Path) -> None:
    h = Harness([ok(), ok()], **cached(tmp_path))
    async with h.client as client:
        first = await client.chat(USER, Receipt)
        second = await client.chat(USER, Receipt)

    assert len(h.requests) == 1  # the second call never reached the server
    assert first.cache_hit is False
    assert first.cost_usd == 140
    assert second.cache_hit is True
    assert second.cached_calls == 1
    assert second.attempts == 0  # attempts counts HTTP requests
    assert second.cost_usd == 0  # a replay costs nothing...
    assert second.avoided_cost_usd == 140  # ...and says what it would have cost
    assert second.usage == first.usage  # tokens are still on record
    assert second.parsed == first.parsed
    assert second.text == first.text
    assert second.structured_mode_used == first.structured_mode_used
    assert second.cached_latency_ms > 0 or first.latency_ms >= 0


async def test_cache_hits_are_counted_apart_from_real_spend_in_the_budget(tmp_path: Path) -> None:
    h = Harness([ok()], **cached(tmp_path))
    async with h.client as client:
        await client.chat(USER, Receipt)
        await client.chat(USER, Receipt)
    assert h.client.budget.spent == 140
    assert h.client.budget.calls == 1
    assert h.client.budget.cached_calls == 1


async def test_a_different_prompt_or_image_is_a_miss(tmp_path: Path) -> None:
    h = Harness([ok(), ok(), ok(), ok()], **cached(tmp_path))
    page = build_image_message("read", [PNG])
    other_page = build_image_message("read", [PNG + b"\x01"])
    async with h.client as client:
        await client.chat([page], Receipt)
        await client.chat([page], Receipt)  # same image: hit
        await client.chat([other_page], Receipt)  # different pixels: miss
        await client.chat([build_image_message("read it again", [PNG])], Receipt)  # prompt: miss
    assert len(h.requests) == 3


async def test_model_and_reasoning_effort_are_part_of_the_key(tmp_path: Path) -> None:
    first = Harness([ok()], **cached(tmp_path))
    async with first.client as client:
        await client.chat(USER, Receipt)
    for override in ({"llm_model": "another-model"}, {"llm_reasoning_effort": "low"}):
        other = Harness([ok()], **cached(tmp_path, **override))
        async with other.client as client:
            result = await client.chat(USER, Receipt)
        assert result.cache_hit is False
        assert len(other.requests) == 1


class ReceiptWithNote(Receipt):
    """A different schema that still accepts the canned reply."""

    note: str | None = None


async def test_the_response_schema_is_part_of_the_key(tmp_path: Path) -> None:
    h = Harness([ok(), ok()], **cached(tmp_path))
    async with h.client as client:
        await client.chat(USER, Receipt)
        await client.chat(USER, ReceiptWithNote)  # same prompt, different schema: miss
    assert len(h.requests) == 2


async def test_the_cache_is_shared_across_clients_and_runs(tmp_path: Path) -> None:
    first = Harness([ok()], **cached(tmp_path))
    async with first.client as client:
        await client.chat(USER, Receipt)
    second = Harness([], **cached(tmp_path))  # an empty script: any request would blow up
    async with second.client as client:
        result = await client.chat(USER, Receipt)
    assert result.cache_hit is True
    assert second.requests == []


async def test_per_call_cache_off_neither_reads_nor_writes(tmp_path: Path) -> None:
    h = Harness([ok(), ok(), ok()], **cached(tmp_path))
    async with h.client as client:
        await client.chat(USER, Receipt, cache="off")  # not written
        second = await client.chat(USER, Receipt)  # miss: nothing was stored
        third = await client.chat(USER, Receipt, cache="off")  # does not read the entry either
    assert second.cache_hit is False
    assert third.cache_hit is False
    assert len(h.requests) == 3


async def test_refresh_re_asks_the_model_and_overwrites_the_entry(tmp_path: Path) -> None:
    newer = '{"vendor": "NEWER", "total": "1.00"}'
    h = Harness([ok(), ok(completion(newer))], **cached(tmp_path))
    async with h.client as client:
        await client.chat(USER, Receipt)
        refreshed = await client.chat(USER, Receipt, cache="refresh")  # skips the read
        replay = await client.chat(USER, Receipt)  # now serves the refreshed answer
    assert refreshed.cache_hit is False
    assert refreshed.parsed is not None
    assert refreshed.parsed.vendor == "NEWER"
    assert replay.cache_hit is True
    assert replay.parsed is not None
    assert replay.parsed.vendor == "NEWER"
    assert len(h.requests) == 2


async def test_a_cache_setting_of_off_writes_nothing_but_a_call_can_opt_in(tmp_path: Path) -> None:
    h = Harness([ok(), ok()], **cached(tmp_path, llm_cache="off"))
    async with h.client as client:
        await client.chat(USER, Receipt)
        assert not (tmp_path / "cache").exists()  # not even the directory
        await client.chat(USER, Receipt, cache="on")  # opt in for one call: miss, then stored
        replay = await client.chat(USER, Receipt, cache="on")
    assert replay.cache_hit is True
    assert len(h.requests) == 2


async def test_refresh_as_a_setting_never_reads(tmp_path: Path) -> None:
    h = Harness([ok(), ok()], **cached(tmp_path, llm_cache="refresh"))
    async with h.client as client:
        await client.chat(USER, Receipt)
        second = await client.chat(USER, Receipt)
    assert second.cache_hit is False
    assert len(h.requests) == 2


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(400, text="bad request"),
        httpx.Response(503, text="overloaded"),
        httpx.Response(200, text="<html>gateway</html>"),
        httpx.Response(200, json={"choices": []}),
    ],
)
async def test_failed_exchanges_are_not_cached(tmp_path: Path, reply: httpx.Response) -> None:
    h = Harness([reply], **cached(tmp_path, llm_max_retries=0, llm_structured_mode="json_object"))
    async with h.client as client:
        with pytest.raises(ProviderError):
            await client.chat(USER, Receipt)
    assert list((tmp_path / "cache").rglob("*.json")) == []


async def test_unparseable_model_output_is_cached_so_repair_sequences_replay(
    tmp_path: Path,
) -> None:
    first = Harness(
        [ok(completion(BAD_JSON)), ok(completion(GOOD))], **cached(tmp_path, llm_max_repairs=1)
    )
    async with first.client as client:
        original = await client.chat(USER, Receipt)
    assert original.repairs == 1

    replay_harness = Harness([], **cached(tmp_path, llm_max_repairs=1))
    async with replay_harness.client as client:
        replay = await client.chat(USER, Receipt)
    assert replay_harness.requests == []  # both the bad answer and its repair came from the cache
    assert replay.cache_hit is True
    assert replay.cached_calls == 2
    assert replay.repairs == 1
    assert replay.parsed == original.parsed
    assert replay.cost_usd == 0
    assert replay.avoided_cost_usd == 280
    assert [r.raw_text for r in replay.repair_history] == [BAD_JSON]


async def test_a_response_that_is_part_cached_and_part_real_pays_only_for_the_real_part(
    tmp_path: Path,
) -> None:
    # run 1: no repairs allowed, so only the unusable first answer is produced (and cached)
    first = Harness([ok(completion(BAD_JSON))], **cached(tmp_path, llm_max_repairs=0))
    async with first.client as client:
        with pytest.raises(ResponseParseError):
            await client.chat(USER, Receipt)
    # run 2: repairs allowed: the first call is a hit, the repair is real
    second = Harness([ok(completion(GOOD))], **cached(tmp_path, llm_max_repairs=1))
    async with second.client as client:
        result = await client.chat(USER, Receipt)

    assert result.repairs == 1
    assert result.cached_calls == 1
    assert result.cache_hit is False  # not ALL of its calls were cached
    assert result.attempts == 1  # one HTTP request: the repair
    assert result.cost_usd == 140  # only the real call was paid for
    assert result.avoided_cost_usd == 140
    assert result.usage.input_tokens == 200  # tokens of both calls are recorded
    assert second.client.budget.spent == 140


async def test_errors_report_real_spend_and_cached_calls_separately(tmp_path: Path) -> None:
    first = Harness([ok(completion(BAD_JSON))], **cached(tmp_path, llm_max_repairs=0))
    async with first.client as client:
        with pytest.raises(ResponseParseError) as real:
            await client.chat(USER, Receipt)
    assert real.value.cost_usd == 140
    assert real.value.cached_calls == 0

    again = Harness([], **cached(tmp_path, llm_max_repairs=0))
    async with again.client as client:
        with pytest.raises(ResponseParseError) as replayed:
            await client.chat(USER, Receipt)
    assert replayed.value.cost_usd == 0
    assert replayed.value.cached_calls == 1
    assert replayed.value.usage is not None
    assert replayed.value.usage.input_tokens == 100  # the tokens are still on record


async def test_cache_hits_are_never_refused_by_an_exceeded_budget_but_misses_are(
    tmp_path: Path,
) -> None:
    h = Harness([ok(), ok()], **cached(tmp_path, max_run_cost_usd="100"))
    async with h.client as client:
        await client.chat(USER, Receipt)  # $140: over the $100 cap
        assert h.client.budget.exceeded
        replay = await client.chat(USER, Receipt)  # free, so allowed
        assert replay.cache_hit is True
        with pytest.raises(BudgetExceededError):  # a real call is refused
            await client.chat([{"role": "user", "content": "something new"}], Receipt)
    assert len(h.requests) == 1


async def test_a_replay_skips_structured_output_negotiation_and_keeps_the_recorded_tier(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["response_format"]["type"] == "json_schema":
            return httpx.Response(400, text="unsupported")
        return ok()

    first = Harness(handler=handler, **cached(tmp_path))
    async with first.client as client:
        original = await client.chat(USER, Receipt)
    assert original.structured_mode_used == "json_object"
    assert len(first.requests) == 2  # the rejected json_schema attempt, then json_object

    second = Harness([], **cached(tmp_path))  # a fresh client: it has not learned anything
    async with second.client as client:
        replay = await client.chat(USER, Receipt)
    assert second.requests == []  # the rejected attempt is not repeated either
    assert replay.structured_mode_used == "json_object"


async def test_a_corrupt_entry_is_a_miss_and_gets_rewritten(tmp_path: Path) -> None:
    h = Harness([ok(), ok()], **cached(tmp_path))
    async with h.client as client:
        await client.chat(USER, Receipt)
        for entry_file in (tmp_path / "cache").rglob("*.json"):
            entry_file.write_text("{truncated", encoding="utf-8")
        again = await client.chat(USER, Receipt)
        replay = await client.chat(USER, Receipt)
    assert again.cache_hit is False
    assert replay.cache_hit is True
    assert len(h.requests) == 2


async def test_entries_hold_no_api_key_and_no_page_pixels(tmp_path: Path) -> None:
    page = build_image_message("read", [PNG])
    h = Harness([ok()], **cached(tmp_path))
    async with h.client as client:
        await client.chat([page], Receipt)
    (entry_file,) = (tmp_path / "cache").rglob("*.json")
    text = entry_file.read_text(encoding="utf-8")
    assert API_KEY not in text
    assert "base64" not in text  # the image is identified by its hash, never stored
    document = json.loads(text)
    assert document["key_parts"]["model"] == "test-model"
    assert len(document["key_parts"]["image_hashes"]) == 1


async def test_plain_chat_is_cached_too(tmp_path: Path) -> None:
    h = Harness([ok(completion("hello"))], **cached(tmp_path))
    async with h.client as client:
        first = await client.chat(USER)
        second = await client.chat(USER)
    assert (first.text, second.text) == ("hello", "hello")
    assert second.cache_hit is True
    assert len(h.requests) == 1


# ---------------------------------------------------------------- secrets


async def test_api_key_never_appears_in_errors_or_reprs() -> None:
    h = Harness([httpx.Response(401, text="invalid key")])
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER)
        assert API_KEY not in str(exc.value)
        assert API_KEY not in repr(exc.value)
        assert API_KEY not in repr(client)


async def test_error_body_is_truncated() -> None:
    h = Harness([httpx.Response(400, text="x" * 10_000)], llm_structured_mode="json_object")
    async with h.client as client:
        with pytest.raises(ProviderError) as exc:
            await client.chat(USER, Receipt)
    assert len(str(exc.value)) < 1_000
