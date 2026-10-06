"""Async client for OpenAI-compatible chat-completions endpoints.

Features: multi-image (base64) messages, structured output (json_schema, falling back to
json_object plus a schema instruction when the server rejects it), timeouts, retry with
exponential backoff on 429/5xx/transport errors, token usage + latency on every call, and a
run-level cost cap (see budget.py).

The key never appears in errors or logs; endpoint, model and key come from `Settings`.
"""

import asyncio
import base64
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Generic, Literal, Self, TypeVar, overload

import httpx
from pydantic import BaseModel, ValidationError, computed_field

from docextract.config import Settings
from docextract.providers.budget import BudgetExceededError, RunBudget
from docextract.providers.schema_utils import simplify_schema

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)
Message = dict[str, Any]
Tier = Literal["json_schema", "json_object", "none"]

_CHAT_PATH = "chat/completions"
_FALLBACK_STATUSES = frozenset({400, 422})  # "this server does not like my response_format"
_ERROR_BODY_CHARS = 500
_MAX_JSON_CANDIDATES = 50
_SCHEMA_INSTRUCTION = (
    "Respond with a single JSON object that conforms to the following JSON Schema. "
    "Output JSON only: no prose and no code fences.\n"
)
_REPAIR_ERROR_CHARS = 1000
_REPAIR_PROMPT = (
    "Your previous reply could not be used: {problem}\n"
    "Reply again with the corrected JSON object only: no prose, no code fences. "
    "Keep every value that was already correct."
)


class Usage(BaseModel):
    """Token counts for one call (or the totals over a call and its repairs).

    `output_tokens` is `completion_tokens` exactly as the provider reported it. Providers
    disagree on whether that already contains reasoning ("thinking") tokens: OpenAI counts
    them inside it, while Google's OpenAI-compatible layer can leave them out and only count
    them in `total_tokens`. `parse_usage` settles it per response from `total_tokens` and
    records in `reasoning_outside_output` the reasoning tokens that must be billed on top, so
    cost neither double-counts them nor misses them.
    """

    input_tokens: int = 0
    output_tokens: int = 0  # completion_tokens as reported
    reasoning_tokens: int = 0  # reported by the provider, else inferred from total_tokens
    reasoning_outside_output: int = 0  # reasoning tokens NOT inside output_tokens (billed on top)
    total_tokens: int | None = None
    reasoning_inferred: bool = False  # reasoning_tokens came from the total_tokens gap
    output_includes_reasoning: bool | None = None  # None: could not be verified
    reported: bool = False  # False when the server returned no usage block

    @computed_field
    @property
    def billable_output_tokens(self) -> int:
        return self.output_tokens + self.reasoning_outside_output

    @property
    def reasoning_note(self) -> str:
        """One line saying what we know about reasoning tokens and how they were costed."""
        if not self.reported:
            return "no usage reported by the server"
        if self.reasoning_outside_output > 0:
            how = (
                "inferred from total_tokens - prompt - completion"
                if self.reasoning_inferred
                else "reported by the provider"
            )
            return (
                f"{self.reasoning_tokens} tokens, {how}; NOT inside completion_tokens, "
                f"so {self.reasoning_outside_output} added to billable output"
            )
        if self.output_includes_reasoning:
            return (
                f"{self.reasoning_tokens} tokens, reported by the provider and already "
                "inside completion_tokens (not added again)"
            )
        if self.total_tokens is None:
            return (
                f"{self.reasoning_tokens} tokens reported; no total_tokens to cross-check, "
                "assumed already inside completion_tokens (unverified)"
            )
        return "none: total_tokens equals prompt + completion"

    def cost_usd(self, settings: Settings) -> Decimal:
        return settings.cost_usd(self.input_tokens, self.billable_output_tokens)

    def __add__(self, other: "Usage") -> "Usage":
        both_totals = self.total_tokens is not None and other.total_tokens is not None
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            reasoning_outside_output=self.reasoning_outside_output + other.reasoning_outside_output,
            total_tokens=(
                (self.total_tokens or 0) + (other.total_tokens or 0) if both_totals else None
            ),
            reasoning_inferred=self.reasoning_inferred or other.reasoning_inferred,
            output_includes_reasoning=(
                self.output_includes_reasoning
                if self.output_includes_reasoning == other.output_includes_reasoning
                else None
            ),
            reported=self.reported or other.reported,
        )


class ProviderError(Exception):
    """A call failed. Carries what we know about the failed call, including its cost."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        attempts: int = 0,
        latency_ms: float = 0.0,
        usage: Usage | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts
        self.latency_ms = latency_ms
        self.usage = usage


class ResponseParseError(ProviderError):
    """The HTTP call succeeded but the model output was not usable JSON for the schema."""

    def __init__(self, message: str, *, raw_text: str, repairs: int = 0, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.raw_text = raw_text  # the last model output we saw
        self.repairs = repairs  # repair re-asks made before giving up


@dataclass(frozen=True)
class LLMResponse(Generic[M]):
    """Result of one `chat()`. When repairs happened, usage/cost/attempts/latency are the
    totals over the original call and every repair; text/raw/finish_reason are the last's."""

    text: str
    parsed: M | None
    usage: Usage
    latency_ms: float  # wall time of the whole call, including retries and backoff sleeps
    attempts: int  # HTTP requests made (retries, fallbacks and repairs included)
    finish_reason: str | None
    structured_mode_used: Tier
    cost_usd: Decimal
    model: str
    raw: dict[str, Any]
    repairs: int = 0  # times the model was re-asked to fix unusable output


@dataclass
class _Tally:
    """Running totals for one `chat()` call, so late errors still report earlier spend."""

    started: float
    attempts: int = 0
    repairs: int = 0
    usage: Usage | None = None  # None until some call has returned (nothing spent yet)

    @property
    def latency_ms(self) -> float:
        return _elapsed_ms(self.started)

    def add_usage(self, usage: Usage) -> None:
        self.usage = usage if self.usage is None else self.usage + usage

    def error_context(self) -> dict[str, Any]:
        return {"attempts": self.attempts, "latency_ms": self.latency_ms, "usage": self.usage}


@dataclass(frozen=True)
class _Completion:
    """One successful HTTP exchange, before the model output is parsed."""

    text: str
    finish_reason: str | None
    tier: Tier
    payload: dict[str, Any]


# ---------------------------------------------------------------- messages


def _sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported image format (expected PNG, JPEG or WebP)")


def build_image_message(
    prompt: str, images: Sequence[bytes | Path], *, role: str = "user"
) -> Message:
    """One message with the prompt text followed by the images, in order, as data URLs."""
    if not images:
        raise ValueError("at least one image is required")
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        data = image.read_bytes() if isinstance(image, Path) else image
        encoded = base64.b64encode(data).decode("ascii")
        url = f"data:{_sniff_mime(data)};base64,{encoded}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": role, "content": parts}


def _with_schema_instruction(messages: Sequence[Message], schema: dict[str, Any]) -> list[Message]:
    instruction = _SCHEMA_INSTRUCTION + json.dumps(schema)
    out = [dict(message) for message in messages]
    if out and out[0].get("role") == "system" and isinstance(out[0].get("content"), str):
        out[0]["content"] = f"{out[0]['content']}\n\n{instruction}"
    else:
        out.insert(0, {"role": "system", "content": instruction})
    return out


# ---------------------------------------------------------------- response parsing


def parse_json_payload(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of model text (tolerates code fences and prose).

    Non-integer numbers are parsed straight to Decimal, so money never passes through a float
    (which would drop trailing zeros and anything past ~17 significant digits).
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty response")
    decoder = json.JSONDecoder(parse_float=Decimal)
    position = -1
    for _ in range(_MAX_JSON_CANDIDATES):
        position = stripped.find("{", position + 1)
        if position == -1:
            break
        try:
            value, _end = decoder.raw_decode(stripped, position)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no valid JSON object found in response")


def parse_usage(raw: object) -> Usage:
    """Read a usage block (OpenAI or input/output_tokens spellings) and settle reasoning.

    Reasoning tokens come from `completion_tokens_details.reasoning_tokens` when present.
    Whether they are already inside `completion_tokens` is decided from `total_tokens`:

      total - prompt - completion > 0  -> the provider left reasoning out of completion_tokens,
                                          so the gap is billed on top (inferred when the
                                          provider did not report reasoning_tokens at all)
      gap == 0                         -> nothing is hidden; any reported reasoning is already
                                          inside completion_tokens, so it is NOT added again
      no total_tokens                  -> cannot cross-check; assume the OpenAI convention
                                          (reasoning inside completion_tokens), unverified
    """
    if not isinstance(raw, dict):
        return Usage()
    prompt = _first_int(raw, "prompt_tokens", "input_tokens")
    completion = _first_int(raw, "completion_tokens", "output_tokens")
    if prompt is None and completion is None:
        return Usage()
    prompt, completion = prompt or 0, completion or 0
    total = _first_int(raw, "total_tokens")
    details = raw.get("completion_tokens_details")
    reported = _first_int(details, "reasoning_tokens") if isinstance(details, dict) else None

    if total is None:
        return Usage(
            input_tokens=prompt,
            output_tokens=completion,
            reasoning_tokens=reported or 0,
            reported=True,
        )
    gap = total - prompt - completion  # <= 0 means nothing is hidden
    if gap > 0:
        return Usage(
            input_tokens=prompt,
            output_tokens=completion,
            reasoning_tokens=gap if reported is None else reported,
            reasoning_outside_output=gap,
            total_tokens=total,
            reasoning_inferred=reported is None,
            output_includes_reasoning=False,
            reported=True,
        )
    return Usage(
        input_tokens=prompt,
        output_tokens=completion,
        reasoning_tokens=reported or 0,
        total_tokens=total,
        output_includes_reasoning=True if reported else None,
        reported=True,
    )


def _first_int(source: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _extract_message(payload: Any) -> tuple[str, str | None]:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    content = message.get("content")
    if isinstance(content, list):  # some servers return a list of content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ValueError("response content has an unexpected type")
    return content, finish_reason if isinstance(finish_reason, str) else None


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None  # HTTP-date form is not honored; fall back to exponential backoff
    return seconds if seconds >= 0 else None


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _is_retryable(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


# ---------------------------------------------------------------- client


class OpenAICompatClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
        budget: RunBudget | None = None,
    ) -> None:
        self._settings = settings
        # one budget per client unless several clients are given the same one to share
        self.budget = budget if budget is not None else RunBudget.from_settings(settings)
        key = settings.llm_api_key.get_secret_value()
        self._http = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=httpx.Timeout(settings.llm_timeout_s),
            transport=transport,
        )
        self._sleep = sleep
        self._rng = rng
        self._json_schema_unsupported = False  # learned once json_object worked where it failed

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    @overload
    async def chat(
        self,
        messages: Sequence[Message],
        response_model: type[M],
        *,
        temperature: float = ...,
        max_tokens: int | None = ...,
    ) -> LLMResponse[M]: ...

    @overload
    async def chat(
        self,
        messages: Sequence[Message],
        response_model: None = ...,
        *,
        temperature: float = ...,
        max_tokens: int | None = ...,
    ) -> LLMResponse[BaseModel]: ...

    async def chat(
        self,
        messages: Sequence[Message],
        response_model: type[M] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse[Any]:
        tally = _Tally(started=time.perf_counter())
        max_repairs = self._settings.llm_max_repairs
        conversation = list(messages)
        last_error: ResponseParseError | None = None
        while True:
            try:
                self.budget.ensure_available()
            except BudgetExceededError as exceeded:
                # a refused repair keeps the parse error it was trying to fix as its cause
                raise exceeded from last_error
            completion = await self._complete(
                conversation, response_model, temperature, max_tokens, tally
            )
            if response_model is None:
                return self._response(completion, None, tally)
            try:
                parsed = self._parse_model(completion, response_model, tally)
            except ResponseParseError as error:
                # truncated output would just truncate again, so it is not worth a repair
                if tally.repairs >= max_repairs or completion.finish_reason == "length":
                    raise
                last_error = error
                tally.repairs += 1
                problem = _shorten(str(error), _REPAIR_ERROR_CHARS)
                logger.warning("unusable model output; repair %d/%d", tally.repairs, max_repairs)
                # always re-ask from the original messages plus the LAST bad output, so the
                # conversation does not grow with every repair
                conversation = [
                    *messages,
                    {"role": "assistant", "content": completion.text},
                    {"role": "user", "content": _REPAIR_PROMPT.format(problem=problem)},
                ]
                continue
            return self._response(completion, parsed, tally)

    async def _complete(
        self,
        messages: Sequence[Message],
        response_model: type[BaseModel] | None,
        temperature: float,
        max_tokens: int | None,
        tally: _Tally,
    ) -> _Completion:
        """One logical call: structured-output tiers (with fallback), retries, body parsing."""
        tiers = self._tiers(response_model)
        fell_back = False

        for index, tier in enumerate(tiers):
            body = self._build_body(messages, response_model, tier, temperature, max_tokens)
            response = await self._post(body, tally)
            if response.is_success:
                break
            if response.status_code in _FALLBACK_STATUSES and index < len(tiers) - 1:
                logger.warning("%s rejected (HTTP %s); falling back", tier, response.status_code)
                fell_back = True
                continue
            raise ProviderError(
                f"HTTP {response.status_code} from LLM endpoint: "
                f"{response.text[:_ERROR_BODY_CHARS]}",
                status_code=response.status_code,
                **tally.error_context(),
            )
        else:  # pragma: no cover - tiers is never empty
            raise AssertionError("no structured-output tier was attempted")

        if fell_back and self._settings.llm_structured_mode == "auto":
            self._json_schema_unsupported = True

        try:
            payload = response.json()
        except ValueError as exc:
            raise ResponseParseError(
                "LLM endpoint returned a non-JSON body",
                raw_text=response.text,
                repairs=tally.repairs,
                **tally.error_context(),
            ) from exc

        # usage is read before the message so a malformed response is still costed
        usage = parse_usage(payload.get("usage") if isinstance(payload, dict) else None)
        tally.add_usage(usage)
        self.budget.record(usage.cost_usd(self._settings), metered=usage.reported)
        if not usage.reported and self.budget.limit is not None:
            logger.warning("response had no usage block: its cost is unknown and not counted")
        try:
            text, finish_reason = _extract_message(payload)
        except ValueError as exc:
            raise ResponseParseError(
                f"unusable response from LLM endpoint: {exc}",
                raw_text=response.text,
                repairs=tally.repairs,
                **tally.error_context(),
            ) from exc
        return _Completion(text=text, finish_reason=finish_reason, tier=tier, payload=payload)

    def _response(
        self, completion: _Completion, parsed: BaseModel | None, tally: _Tally
    ) -> LLMResponse[Any]:
        usage = tally.usage or Usage()
        return LLMResponse(
            text=completion.text,
            parsed=parsed,
            usage=usage,
            latency_ms=tally.latency_ms,
            attempts=tally.attempts,
            finish_reason=completion.finish_reason,
            structured_mode_used=completion.tier,
            cost_usd=usage.cost_usd(self._settings),
            model=self._settings.llm_model,
            raw=completion.payload,
            repairs=tally.repairs,
        )

    # -- helpers

    def _tiers(self, response_model: type[BaseModel] | None) -> list[Tier]:
        if response_model is None:
            return ["none"]
        mode = self._settings.llm_structured_mode
        if mode == "auto":
            return (
                ["json_object"] if self._json_schema_unsupported else ["json_schema", "json_object"]
            )
        return [mode]

    def _build_body(
        self,
        messages: Sequence[Message],
        response_model: type[BaseModel] | None,
        tier: Tier,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if self._settings.llm_reasoning_effort:
            body["reasoning_effort"] = self._settings.llm_reasoning_effort
        if response_model is not None and tier != "none":
            schema = simplify_schema(response_model.model_json_schema())
            if tier == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": response_model.__name__, "schema": schema},
                }
            else:
                body["response_format"] = {"type": "json_object"}
                body["messages"] = _with_schema_instruction(messages, schema)
        return body

    async def _post(self, body: dict[str, Any], tally: _Tally) -> httpx.Response:
        """POST with retry on 429/5xx/transport errors. Returns any non-retryable response."""
        settings = self._settings
        for retry in range(settings.llm_max_retries + 1):
            tally.attempts += 1
            response: httpx.Response | None = None
            try:
                response = await self._http.post(_CHAT_PATH, json=body)
            except httpx.UnsupportedProtocol as exc:
                raise ProviderError(
                    f"invalid LLM_BASE_URL: {exc}", **tally.error_context()
                ) from exc
            except httpx.TransportError as exc:
                reason = f"{type(exc).__name__}: {exc}"
            else:
                if not _is_retryable(response.status_code):
                    return response
                reason = f"HTTP {response.status_code}"

            if retry == settings.llm_max_retries:
                raise ProviderError(
                    f"LLM call failed after {tally.attempts} attempt(s): {reason}",
                    status_code=response.status_code if response is not None else None,
                    **tally.error_context(),
                )
            delay = self._backoff_delay(retry, response)
            logger.warning("LLM call failed (%s); retry %d in %.2fs", reason, retry + 1, delay)
            await self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _backoff_delay(self, retry: int, response: httpx.Response | None) -> float:
        settings = self._settings
        retry_after = (
            _parse_retry_after(response.headers.get("retry-after"))
            if response is not None
            else None
        )
        if retry_after is not None:
            return min(retry_after, settings.llm_backoff_max_s)
        ceiling = min(settings.llm_backoff_max_s, settings.llm_backoff_base_s * 2**retry)
        return ceiling * (0.5 + self._rng() / 2)  # jitter: 50-100% of the ceiling

    @staticmethod
    def _parse_model(completion: _Completion, response_model: type[M], tally: _Tally) -> M:
        text, finish_reason = completion.text, completion.finish_reason
        try:
            data = parse_json_payload(text)
        except ValueError as exc:
            hint = (
                " (output was truncated: finish_reason=length)" if finish_reason == "length" else ""
            )
            raise ResponseParseError(
                f"model output is not valid JSON{hint}: {exc}",
                raw_text=text,
                repairs=tally.repairs,
                **tally.error_context(),
            ) from exc
        try:
            return response_model.model_validate(data)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()[:5]
            )
            raise ResponseParseError(
                f"model output does not match {response_model.__name__}: {problems}",
                raw_text=text,
                repairs=tally.repairs,
                **tally.error_context(),
            ) from exc
