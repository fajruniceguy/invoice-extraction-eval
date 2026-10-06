import copy
import json
from decimal import Decimal
from typing import Any

import httpx
import pytest

from docextract.config import Settings
from docextract.extract.preprocess import normalize_extraction_payload
from docextract.extract.schema import DocumentExtraction
from docextract.providers.openai_compat import (
    OpenAICompatClient,
    PreprocessResult,
    ResponseParseError,
)
from payloads import absent, extracted, full_payload, row

USER = [{"role": "user", "content": "extract"}]


def idr_payload(**overrides: Any) -> dict[str, Any]:
    payload = full_payload()
    payload["currency"] = extracted("IDR")
    payload.update(overrides)
    return payload


# ------------------------------------------------------------------ the normalizer itself


class TestNormalizeExtractionPayload:
    def test_indonesian_money_dates_and_cells_are_rewritten(self) -> None:
        payload = idr_payload(
            subtotal=extracted("1.126.126,13"),
            tax=extracted("123.873,87"),
            total=extracted("1.250.000,00"),
            document_date=extracted("5 Maret 2024"),
            line_items=[
                row(quantity="0,352", unit_price="25.900", amount="9.117"),
            ],
        )
        result = normalize_extraction_payload(payload)
        assert result.problems == ()
        assert result.data["subtotal"]["value"] == "1126126.13"
        assert result.data["tax"]["value"] == "123873.87"
        assert result.data["total"]["value"] == "1250000.00"
        assert result.data["document_date"]["value"] == "2024-03-05"
        cells = result.data["line_items"][0]
        # "25.900" and "9.117" are exactly the ambiguous shape; the IDR context decides them
        assert (cells["quantity"], cells["unit_price"], cells["amount"]) == (
            "0.352",
            "25900",
            "9117",
        )

    def test_every_rewrite_is_reported(self) -> None:
        result = normalize_extraction_payload(
            idr_payload(total=extracted("1.250.000,00"), document_date=extracted("5 Maret 2024"))
        )
        assert "total.value: '1.250.000,00' -> '1250000.00'" in result.changes
        assert "document_date.value: '5 Maret 2024' -> '2024-03-05'" in result.changes

    def test_the_input_is_never_mutated(self) -> None:
        payload = idr_payload(total=extracted("1.250.000,00"))
        snapshot = copy.deepcopy(payload)
        result = normalize_extraction_payload(payload)
        assert payload == snapshot
        assert result.data is not payload
        assert result.data["total"]["value"] == "1250000.00"

    def test_numbers_null_and_already_clean_strings_are_left_alone(self) -> None:
        payload = idr_payload(
            subtotal=extracted(16.0),  # a JSON number, not a string
            tax=extracted(1),
            discount=absent(),
            total=extracted("17.28"),  # already a plain decimal
            document_date=extracted("2024-03-05"),
        )
        result = normalize_extraction_payload(payload)
        assert result.changes == ()
        assert result.problems == ()
        assert result.data["subtotal"]["value"] == 16.0
        assert result.data["discount"]["value"] is None
        assert result.data["total"]["value"] == "17.28"

    def test_text_fields_are_never_touched(self) -> None:
        payload = idr_payload(
            vendor_name=extracted("1.250.000"),
            vendor_tax_id=extracted("01.234.567.8-901.000"),
            document_number=extracted("12/03/2024"),
        )
        result = normalize_extraction_payload(payload)
        assert result.data["vendor_name"]["value"] == "1.250.000"
        assert result.data["vendor_tax_id"]["value"] == "01.234.567.8-901.000"
        assert result.data["document_number"]["value"] == "12/03/2024"
        assert result.changes == ()

    @pytest.mark.parametrize(
        ("currency", "expected"),
        [("IDR", "1250"), ("idr", "1250"), (" IDR ", "1250"), ("USD", "1.250"), ("EUR", "1250")],
    )
    def test_the_payloads_own_currency_is_the_context(self, currency: str, expected: str) -> None:
        payload = full_payload()
        payload["currency"] = extracted(currency)
        payload["total"] = extracted("1.250")
        assert normalize_extraction_payload(payload).data["total"]["value"] == expected

    def test_an_ambiguous_value_without_a_currency_is_left_unchanged_and_reported(self) -> None:
        payload = full_payload()
        payload["currency"] = absent()
        payload["total"] = extracted("1.250")
        result = normalize_extraction_payload(payload)
        assert result.data["total"]["value"] == "1.250"  # not guessed
        assert len(result.problems) == 1
        assert result.problems[0].startswith("total.value: ")
        assert "no currency context" in result.problems[0]

    def test_an_abstained_currency_gives_no_context_either(self) -> None:
        payload = full_payload()
        payload["currency"] = {
            "value": None,
            "confidence": 0.2,
            "status": "abstained",
            "reasons": ["smudged"],
        }
        payload["total"] = extracted("1.250")
        assert len(normalize_extraction_payload(payload).problems) == 1

    def test_garbage_is_reported_not_rewritten(self) -> None:
        result = normalize_extraction_payload(idr_payload(total=extracted("about a million")))
        assert result.data["total"]["value"] == "about a million"
        assert any(problem.startswith("total.value: ") for problem in result.problems)

    def test_idr_documents_are_read_day_first(self) -> None:
        result = normalize_extraction_payload(idr_payload(document_date=extracted("05/03/2024")))
        assert result.data["document_date"]["value"] == "2024-03-05"
        assert result.problems == ()

    def test_other_currencies_do_not_assume_day_first(self) -> None:
        payload = full_payload()  # USD
        payload["document_date"] = extracted("05/03/2024")
        result = normalize_extraction_payload(payload)
        assert result.data["document_date"]["value"] == "05/03/2024"
        assert "day_first" in result.problems[0]

    def test_line_item_problems_name_the_row_and_cell(self) -> None:
        payload = full_payload()
        payload["currency"] = absent()
        payload["line_items"] = [row(), row(quantity="1.250")]
        result = normalize_extraction_payload(payload)
        assert any(p.startswith("line_items[2].quantity: ") for p in result.problems)

    def test_each_problem_is_reported_once_per_field(self) -> None:
        payload = full_payload()
        payload["currency"] = absent()
        payload["subtotal"] = extracted("1.250")
        payload["tax"] = extracted("2.500")
        result = normalize_extraction_payload(payload)
        assert len(result.problems) == 2

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"total": None},
            {"total": "17.28"},  # a bare value where a field object should be
            {"total": {"value": ["not", "a", "string"]}},
            {"line_items": None},
            {"line_items": "none"},
            {"line_items": [None, 5, "x"]},
            {"currency": "IDR", "total": {"value": "1.250"}},  # malformed currency field
        ],
    )
    def test_malformed_payloads_do_not_crash_it(self, payload: dict[str, Any]) -> None:
        result = normalize_extraction_payload(payload)
        assert isinstance(result, PreprocessResult)


# ------------------------------------------------------------------ with the client

GOOD = json.dumps(full_payload())


class Harness:
    """A client over httpx.MockTransport that replays scripted model replies."""

    def __init__(self, replies: list[str], **settings: Any) -> None:
        self.requests: list[dict[str, Any]] = []
        self._replies = list(replies)

        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            content = self._replies.pop(0)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                },
            )

        values: dict[str, Any] = {
            "llm_base_url": "https://llm.test/v1/",
            "llm_model": "m",
            "llm_api_key": "k",
            "llm_max_repairs": 1,
            **settings,
        }
        self.client = OpenAICompatClient(
            Settings(_env_file=None, **values), transport=httpx.MockTransport(respond)
        )


def reply(**overrides: Any) -> str:
    return json.dumps(idr_payload(**overrides))


async def test_indonesian_formatted_output_validates_without_a_repair_call() -> None:
    printed = reply(
        total=extracted("1.250.000,00"),
        subtotal=extracted("1.126.126,13"),
        document_date=extracted("5 Maret 2024"),
        line_items=[row(quantity="0,352", unit_price="25.900", amount="9.117")],
    )
    h = Harness([printed])
    async with h.client as client:
        result = await client.chat(
            USER, DocumentExtraction, preprocess=normalize_extraction_payload
        )

    assert len(h.requests) == 1  # no repair: normalization fixed it
    assert result.repairs == 0
    assert result.parsed is not None
    assert result.parsed.total.value == Decimal("1250000.00")
    assert result.parsed.subtotal.value == Decimal("1126126.13")
    assert str(result.parsed.document_date.value) == "2024-03-05"
    line = result.parsed.line_items[0]
    assert (line.quantity, line.unit_price, line.amount) == (Decimal("0.352"), 25900, 9117)
    assert "total.value: '1.250.000,00' -> '1250000.00'" in result.normalizations


async def test_without_the_hook_the_same_output_costs_a_repair_call() -> None:
    printed = reply(total=extracted("1.250.000,00"))
    h = Harness([printed, reply(total=extracted("1250000.00"))])
    async with h.client as client:
        result = await client.chat(USER, DocumentExtraction)  # no preprocess
    assert len(h.requests) == 2
    assert result.repairs == 1
    assert result.normalizations == ()


async def test_an_ambiguous_value_goes_to_repair_and_the_prompt_says_why() -> None:
    ambiguous = reply(currency=absent(), total=extracted("1.250"))
    fixed = reply(currency=absent(), total=extracted("1250.00"))
    h = Harness([ambiguous, fixed])
    async with h.client as client:
        result = await client.chat(
            USER, DocumentExtraction, preprocess=normalize_extraction_payload
        )

    assert result.repairs == 1
    assert len(h.requests) == 2
    repair_prompt = h.requests[1]["messages"][-1]["content"]
    assert "could not normalize" in repair_prompt
    assert "total.value" in repair_prompt
    assert "1250 (thousands) or 1.250 (decimal)" in repair_prompt
    assert result.parsed is not None
    assert result.parsed.total.value == Decimal("1250.00")


async def test_an_ambiguous_value_is_an_error_even_though_pydantic_would_accept_it() -> None:
    # Decimal("1.250") parses fine, so validation alone would silently accept the guess the
    # normalizer refused to make. With no repairs allowed the error must surface.
    h = Harness([reply(currency=absent(), total=extracted("1.250"))], llm_max_repairs=0)
    async with h.client as client:
        with pytest.raises(ResponseParseError, match="could not normalize") as exc:
            await client.chat(USER, DocumentExtraction, preprocess=normalize_extraction_payload)
    assert "total.value" in str(exc.value)
    # and without the hook that very same output is (wrongly) accepted as 1.250
    h = Harness([reply(currency=absent(), total=extracted("1.250"))], llm_max_repairs=0)
    async with h.client as client:
        unguarded = await client.chat(USER, DocumentExtraction)
    assert unguarded.parsed is not None
    assert unguarded.parsed.total.value == Decimal("1.250")


async def test_unparseable_text_in_a_money_field_goes_to_repair() -> None:
    h = Harness([reply(total=extracted("about a million")), reply()])
    async with h.client as client:
        result = await client.chat(
            USER, DocumentExtraction, preprocess=normalize_extraction_payload
        )
    assert result.repairs == 1
    assert "about a million" in result.repair_history[0].problem


async def test_normalization_does_not_hide_other_problems() -> None:
    broken = idr_payload(total=extracted("1.250.000,00"))
    broken["vendor_name"] = {"value": None, "confidence": 0.5, "status": "extracted"}  # invalid
    h = Harness([json.dumps(broken), reply()])
    async with h.client as client:
        result = await client.chat(
            USER, DocumentExtraction, preprocess=normalize_extraction_payload
        )
    assert result.repairs == 1
    assert "vendor_name" in result.repair_history[0].problem
    assert result.normalizations == ()  # the final, repaired output needed no rewriting


async def test_the_hook_runs_only_on_json_that_parsed() -> None:
    calls: list[dict[str, Any]] = []

    def spy(data: dict[str, Any]) -> PreprocessResult:
        calls.append(data)
        return PreprocessResult(data=data)

    h = Harness(["this is not json", GOOD])
    async with h.client as client:
        await client.chat(USER, DocumentExtraction, preprocess=spy)
    assert len(calls) == 1  # not called for the unusable first reply, once for the repair


async def test_a_custom_hook_may_rewrite_anything() -> None:
    def rename(data: dict[str, Any]) -> PreprocessResult:
        fixed = copy.deepcopy(data)
        fixed["vendor_name"]["value"] = "REWRITTEN"
        return PreprocessResult(data=fixed, changes=("vendor_name.value: rewritten",))

    h = Harness([GOOD])
    async with h.client as client:
        result = await client.chat(USER, DocumentExtraction, preprocess=rename)
    assert result.parsed is not None
    assert result.parsed.vendor_name.value == "REWRITTEN"
    assert result.normalizations == ("vendor_name.value: rewritten",)


async def test_normalizations_are_recomputed_on_a_cache_replay(tmp_path: Any) -> None:
    printed = reply(total=extracted("1.250.000,00"))
    cache = {"llm_cache": "on", "llm_cache_dir": tmp_path / "cache"}
    first = Harness([printed], **cache)
    async with first.client as client:
        original = await client.chat(
            USER, DocumentExtraction, preprocess=normalize_extraction_payload
        )
    again = Harness([], **cache)  # replays from the cache: the RAW completion is stored
    async with again.client as client:
        replay = await client.chat(
            USER, DocumentExtraction, preprocess=normalize_extraction_payload
        )
    assert replay.cache_hit is True
    assert replay.parsed == original.parsed
    assert replay.normalizations == original.normalizations
    assert replay.normalizations  # the cache holds raw output, so normalization really reran


async def test_the_realistic_model_output_numbers_pass_straight_through() -> None:
    # a well-behaved model returns plain numbers: the hook must not disturb them
    h = Harness([GOOD])
    async with h.client as client:
        result = await client.chat(
            USER, DocumentExtraction, preprocess=normalize_extraction_payload
        )
    assert result.normalizations == ()
    assert result.repairs == 0
