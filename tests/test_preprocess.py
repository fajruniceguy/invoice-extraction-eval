import copy
import json
from decimal import Decimal
from typing import Any

import httpx
import pytest

from docextract.config import Settings
from docextract.extract.preprocess import (
    change_label,
    check_contract,
    extraction_preprocess,
    normalize_extraction_payload,
)
from docextract.extract.schema import SCALAR_FIELD_NAMES, DocumentExtraction
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


# ------------------------------------------------------------------ the evidence contract


def cited(payload: dict[str, Any] | None = None, page: int = 1) -> dict[str, Any]:
    """`payload` (default: full_payload) with evidence and a page on every extracted field."""
    out = copy.deepcopy(full_payload() if payload is None else payload)
    for name in SCALAR_FIELD_NAMES:
        field = out[name]
        if field["status"] == "extracted":
            field.setdefault("evidence", str(field["value"]))
            field.setdefault("page", page)
    return out


class TestCheckContract:
    def test_a_fully_cited_payload_breaks_nothing(self) -> None:
        assert check_contract(cited(), page_count=1) == ()

    @pytest.mark.parametrize("evidence", [None, "", "   ", 42])
    def test_an_extracted_field_without_evidence_is_named(self, evidence: Any) -> None:
        payload = cited()
        payload["total"]["evidence"] = evidence
        assert check_contract(payload, 1) == (
            "extracted without evidence (quote the printed text): total",
        )

    def test_an_extracted_field_without_a_page_is_named(self) -> None:
        payload = cited()
        del payload["vendor_name"]["page"]
        payload["total"]["page"] = None
        assert check_contract(payload, 1) == ("extracted without a page: vendor_name, total",)

    def test_absent_and_abstained_fields_need_no_evidence_or_page(self) -> None:
        payload = cited()
        assert payload["vendor_tax_id"]["status"] == "absent"
        assert payload["payment_method"]["status"] == "abstained"
        assert "evidence" not in payload["vendor_tax_id"]
        assert check_contract(payload, 1) == ()

    @pytest.mark.parametrize("page", [0, 4, "4", Decimal("4.0"), 4.0, -1])
    def test_a_page_outside_the_document_is_named(self, page: Any) -> None:
        payload = cited()
        payload["total"]["page"] = page
        assert check_contract(payload, 3) == (
            "page outside 1..3 (the document has 3 pages): total",
        )

    @pytest.mark.parametrize("page", [1, 3, "2", Decimal("2"), Decimal("3.0"), 2.0])
    def test_a_page_inside_the_document_is_fine(self, page: Any) -> None:
        payload = cited()
        payload["total"]["page"] = page
        assert check_contract(payload, 3) == ()

    def test_even_an_absent_field_may_not_cite_a_page_that_was_not_sent(self) -> None:
        payload = cited()
        payload["discount"]["page"] = 2
        assert check_contract(payload, 1) == (
            "page outside 1..1 (the document has 1 page): discount",
        )

    # 0.5 and False would become page 0 (off the document) if they were truncated or counted
    @pytest.mark.parametrize(
        "page", ["two", Decimal("1.5"), Decimal("0.5"), 1.5, 0.5, True, False, [1]]
    )
    def test_a_page_that_is_not_a_whole_number_is_left_to_validation(self, page: Any) -> None:
        payload = cited()
        payload["total"]["page"] = page
        assert check_contract(payload, 3) == ()

    def test_uncited_rows_are_listed_as_ranges(self) -> None:
        payload = cited()
        payload["line_items"] = [row(evidence=None) for _ in range(3)] + [row(), row(evidence="")]
        payload["line_items"].append(row(page=None))
        assert check_contract(payload, 1) == (
            "extracted without evidence (quote the printed text): line_items[1-3,5]",
            "extracted without a page: line_items[6]",
        )

    def test_fields_come_before_rows_and_rows_off_the_document_are_named(self) -> None:
        payload = cited()
        payload["total"]["page"] = 9
        payload["line_items"] = [row(page=2), row(page=9)]
        assert check_contract(payload, 2) == (
            "page outside 1..2 (the document has 2 pages): total, line_items[2]",
        )

    def test_abstained_rows_need_no_evidence(self) -> None:
        payload = cited()
        payload["line_items"] = [
            {"status": "abstained", "confidence": 0.1, "reasons": ["torn"], "page": 1}
        ]
        assert check_contract(payload, 1) == ()

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"total": None},
            {"total": "17.28"},
            {"total": {"status": "extracted"}},
            {"line_items": None},
            {"line_items": [None, 5, "x"]},
        ],
    )
    def test_malformed_payloads_do_not_crash_it(self, payload: dict[str, Any]) -> None:
        assert isinstance(check_contract(payload, 1), tuple)

    def test_a_bare_extracted_field_breaks_both_rules(self) -> None:
        assert check_contract({"total": {"status": "extracted"}}, 1) == (
            "extracted without evidence (quote the printed text): total",
            "extracted without a page: total",
        )


class TestExtractionPreprocess:
    def test_it_normalizes_and_checks_the_contract(self) -> None:
        payload = cited(idr_payload(total=extracted("1.250.000,00", evidence="Rp 1.250.000,00")))
        payload["vendor_name"]["page"] = 3
        result = extraction_preprocess(page_count=2)(payload)
        assert result.data["total"]["value"] == "1250000.00"
        assert result.changes == ("total.value: '1.250.000,00' -> '1250000.00'",)
        assert result.problems == ()
        assert result.violations == ("page outside 1..2 (the document has 2 pages): vendor_name",)

    def test_normalizer_problems_are_kept(self) -> None:
        payload = cited(idr_payload(currency=absent(), total=extracted("1.250")))
        result = extraction_preprocess(page_count=1)(payload)
        assert len(result.problems) == 1 and "total.value" in result.problems[0]
        assert result.violations == ()

    def test_the_input_is_never_mutated(self) -> None:
        payload = cited(idr_payload(total=extracted("1.250.000,00")))
        before = copy.deepcopy(payload)
        extraction_preprocess(page_count=1)(payload)
        assert payload == before

    def test_a_document_has_at_least_one_page(self) -> None:
        with pytest.raises(ValueError):
            extraction_preprocess(page_count=0)


async def test_an_uncited_field_goes_to_repair_and_the_prompt_says_what_to_fix() -> None:
    uncited = cited()
    del uncited["total"]["evidence"]
    h = Harness([json.dumps(uncited), json.dumps(cited())])
    async with h.client as client:
        result = await client.chat(USER, DocumentExtraction, preprocess=extraction_preprocess(1))
    assert result.repairs == 1 and len(h.requests) == 2
    repair_prompt = h.requests[1]["messages"][-1]["content"]
    assert "extracted without evidence (quote the printed text): total" in repair_prompt
    assert "could not normalize" not in repair_prompt  # a violation is worded as itself
    assert result.repair_history[0].problem.endswith(
        "extracted without evidence (quote the printed text): total"
    )
    assert result.parsed is not None and result.parsed.total.evidence == "9.72"


async def test_a_page_that_was_not_sent_goes_to_repair() -> None:
    wrong = cited(page=1)
    wrong["line_items"][0]["page"] = 3
    h = Harness([json.dumps(wrong), json.dumps(cited())])
    async with h.client as client:
        result = await client.chat(USER, DocumentExtraction, preprocess=extraction_preprocess(2))
    assert result.repairs == 1
    assert "page outside 1..2 (the document has 2 pages): line_items[1]" in (
        result.repair_history[0].problem
    )


async def test_a_violation_is_an_error_even_though_validation_passes() -> None:
    uncited = json.dumps(full_payload())  # valid DocumentExtraction, but most fields uncited
    h = Harness([uncited], llm_max_repairs=0)
    async with h.client as client:
        with pytest.raises(ResponseParseError, match="extracted without evidence"):
            await client.chat(USER, DocumentExtraction, preprocess=extraction_preprocess(1))
    h = Harness([uncited], llm_max_repairs=0)
    async with h.client as client:  # the plain normalizer has no such rule
        assert (await client.chat(USER, DocumentExtraction)).parsed is not None


async def test_a_hook_with_only_violations_still_forces_the_error() -> None:
    def strict(data: dict[str, Any]) -> PreprocessResult:
        return PreprocessResult(data=data, violations=("rule X broken",))

    h = Harness([GOOD], llm_max_repairs=0)
    async with h.client as client:
        with pytest.raises(ResponseParseError, match=r"does not match DocumentExtraction: rule X"):
            await client.chat(USER, DocumentExtraction, preprocess=strict)


class TestChangeLabel:
    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            ("total.value: '1.250.000,00' -> '1250000.00'", "total.value"),
            ("document_date.value: '5 Maret 2024' -> '2024-03-05'", "document_date.value"),
            ("line_items[3].amount: '1.250' -> '1250'", "line_items[3].amount"),
            ("not a change", None),
            (": no label", None),
            ("", None),
            ("   ", None),
        ],
    )
    def test_the_target_a_change_names(self, change: str, expected: str | None) -> None:
        assert change_label(change) == expected

    def test_the_labels_of_a_real_normalization(self) -> None:
        payload = idr_payload(
            total=extracted("1.250.000,00"),
            document_date=extracted("5 Maret 2024"),
            line_items=[row(amount="1.250,00")],
        )
        result = normalize_extraction_payload(payload)
        assert [change_label(change) for change in result.changes] == [
            "total.value",
            "document_date.value",
            "line_items[1].amount",
        ]
