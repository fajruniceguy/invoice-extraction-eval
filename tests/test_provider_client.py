import asyncio
import json
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from docextract.config import Settings
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
