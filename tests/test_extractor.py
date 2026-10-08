import base64
import copy
import hashlib
import io
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from docextract.confidence import NORMALIZED_CAP, SHAPE_CAP
from docextract.config import Settings
from docextract.extract.extractor import (
    ExtractionRefused,
    build_messages,
    extract_pages,
)
from docextract.extract.prompt import (
    PROMPT_SHA256,
    SYSTEM_PROMPT,
    USER_PROMPT,
    prompt_sha256,
    user_text,
)
from docextract.ingest.stored import stored_pages
from docextract.providers.openai_compat import OpenAICompatClient, ResponseParseError
from docextract.validate.codes import (
    CONFIDENCE_BELOW_THRESHOLD,
    EVIDENCE_NOT_GROUNDED,
    TAX_ID_UNUSUAL_SHAPE,
)
from fakes.fake_llm import FakeLLMServer, canned_answer, chat_reply
from files import A4, make_sized_pdf
from ingest_support import Env, doc, settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a real LLM_* variable (or a real key) into these tests."""
    for name in list(os.environ):
        if name.startswith(("LLM_", "PRELABEL_", "MAX_RUN_COST")) and name not in (
            "LLM_CACHE",
            "LLM_CACHE_DIR",
        ):
            monkeypatch.delenv(name)


@pytest.fixture
def three_pages(env: Env) -> list[Path]:
    """A stored 3-page document whose pages differ (different sizes), so order is checkable."""
    make_sized_pdf(env.raw / "doc_001.pdf", [A4, (420.0, 595.0), (300.0, 300.0)])
    env.ingest(doc("doc_001.pdf"), settings=settings(dpi=40))
    pages = stored_pages("doc_001", env.pages)
    assert len({page.read_bytes() for page in pages}) == 3
    return pages


def make_settings(server: FakeLLMServer, **changes: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_base_url": server.base_url,
        "llm_model": "fake-extractor",
        "llm_api_key": "sk-test-key",
        "llm_max_repairs": 1,
        "llm_max_retries": 0,
    }
    values.update(changes)
    return Settings(_env_file=None, **values)


async def run(server: FakeLLMServer, pages: list[Path], **changes: Any) -> Any:
    config = make_settings(server, **changes)
    async with OpenAICompatClient(config) as client:
        return await extract_pages(pages, client=client, settings=config)


def answer(**changes: Any) -> dict[str, Any]:
    """The fake's valid, fully cited answer (every citation on page 1), with fields replaced."""
    data = canned_answer()
    data.update(copy.deepcopy(changes))
    return data


def sent_images(body: dict[str, Any]) -> list[bytes]:
    parts = body["messages"][1]["content"][1:]
    assert all(part["type"] == "image_url" for part in parts)
    return [base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) for part in parts]


# ------------------------------------------------------------------ the prompt


class TestPrompt:
    def test_it_asks_for_null_when_a_field_is_not_printed(self) -> None:
        assert 'its status is "absent" and its value is null' in SYSTEM_PROMPT
        assert "Never guess, infer or compute a value that is not printed" in SYSTEM_PROMPT

    def test_it_asks_for_a_verbatim_quote_and_the_page_of_every_extracted_value(self) -> None:
        assert "evidence is a verbatim quote of the printed text" in SYSTEM_PROMPT
        assert "page is the 1-based number of the page that quote is on" in SYSTEM_PROMPT
        assert "the first image is page 1" in SYSTEM_PROMPT
        assert "ONE status, confidence, evidence, page and reasons for the whole row" in (
            SYSTEM_PROMPT
        )

    def test_the_field_definitions_follow_the_labeling_rules(self) -> None:
        for phrase in ("the DPP", "the PPN", "nomor seri faktur", "seller's NPWP", "Harga Jual"):
            assert phrase in SYSTEM_PROMPT, phrase

    def test_an_item_code_column_is_kept_out_of_the_description(self) -> None:
        # dev runs differed on cons_003: the code ("BRG-001") was glued onto every description
        assert "description is the item name only" in SYSTEM_PROMPT
        assert "the code is not part of the description" in SYSTEM_PROMPT

    def test_the_prompt_hash_keeps_its_legacy_definition(self) -> None:
        legacy = hashlib.sha256(f"{SYSTEM_PROMPT}\x00{USER_PROMPT}".encode()).hexdigest()
        assert PROMPT_SHA256 == legacy == prompt_sha256(SYSTEM_PROMPT)

    def test_another_system_prompt_is_another_hash(self) -> None:
        assert prompt_sha256(SYSTEM_PROMPT + " ") != PROMPT_SHA256
        assert prompt_sha256("a") != prompt_sha256("b")

    def test_the_user_text_counts_the_pages(self) -> None:
        assert user_text(1) == "Extract the fields from this document (1 page, in order)."
        assert user_text(3) == "Extract the fields from this document (3 pages, in order)."


# ------------------------------------------------------------------ one multi-page call


class TestOneCall:
    async def test_every_page_goes_in_one_request_in_order(
        self, fake_llm: FakeLLMServer, three_pages: list[Path]
    ) -> None:
        result = await run(fake_llm, three_pages)
        assert result.pages == 3 and len(fake_llm.requests) == 1
        body = fake_llm.requests[0].body
        assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert body["messages"][1]["content"][0] == {"type": "text", "text": user_text(3)}
        assert sent_images(body) == [page.read_bytes() for page in three_pages]  # page order
        assert body["response_format"]["type"] == "json_schema"
        assert body["temperature"] == 0.0 and body["model"] == "fake-extractor"

    async def test_the_parsed_output_and_the_validated_copy_are_both_returned(
        self, fake_llm: FakeLLMServer, three_pages: list[Path]
    ) -> None:
        rows = canned_answer()["line_items"]
        rows[0]["amount"] = 99.0  # 2 x 4.50 is not 99: the arithmetic validator flags it
        fake_llm.enqueue(chat_reply(answer(line_items=rows)))
        result = await run(fake_llm, three_pages)
        assert result.parsed.line_items[0].flagged_cells == []  # what the model said
        assert result.validated.line_items[0].flagged_cells == ["quantity", "unit_price", "amount"]
        assert result.response.repairs == 0
        assert result.parsed.vendor_name.value == "CORNER CAFE DEMO"

    async def test_the_sent_page_size_follows_the_setting(
        self, fake_llm: FakeLLMServer, three_pages: list[Path]
    ) -> None:
        await run(fake_llm, three_pages, llm_image_max_long_edge=64)
        for data in sent_images(fake_llm.requests[0].body):
            assert max(Image.open(io.BytesIO(data)).size) <= 64

    def test_the_default_system_prompt_is_the_production_one(self, three_pages: list[Path]) -> None:
        config = Settings(_env_file=None, llm_base_url="http://x.test/v1", llm_model="m")
        assert build_messages(three_pages, config) == build_messages(
            three_pages, config, SYSTEM_PROMPT
        )

    async def test_another_system_prompt_replaces_only_the_system_message(
        self, fake_llm: FakeLLMServer, three_pages: list[Path]
    ) -> None:
        config = make_settings(fake_llm)
        async with OpenAICompatClient(config) as client:
            await extract_pages(
                three_pages, client=client, settings=config, system_prompt="Only the vendor."
            )
        body = fake_llm.requests[0].body
        assert body["messages"][0] == {"role": "system", "content": "Only the vendor."}
        assert body["messages"][1]["content"][0] == {"type": "text", "text": user_text(3)}
        assert sent_images(body) == [page.read_bytes() for page in three_pages]

    def test_build_messages_matches_what_is_sent(self, three_pages: list[Path]) -> None:
        config = Settings(_env_file=None, llm_base_url="http://x.test/v1", llm_model="m")
        messages = build_messages(three_pages, config)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert len(messages[1]["content"]) == 1 + 3


# ------------------------------------------------------------------ repairs


def _null_vendor() -> dict[str, Any]:
    return answer(vendor_name={"value": None, "confidence": 0.9, "status": "extracted"})


def _uncited_total() -> dict[str, Any]:
    total = canned_answer()["total"]
    total["evidence"] = None
    return answer(total=total)


def _uncited_row() -> dict[str, Any]:
    rows = canned_answer()["line_items"]
    del rows[1]["evidence"]
    return answer(line_items=rows)


def _page_four() -> dict[str, Any]:
    tax = canned_answer()["tax"]
    tax["page"] = 4
    return answer(tax=tax)


@pytest.mark.parametrize(
    ("bad", "problem"),
    [
        ("this is not json at all", "model output is not valid JSON"),
        (_null_vendor(), "vendor_name: Value error, an extracted field must have a non-null value"),
        (_uncited_total(), "extracted without evidence (quote the printed text): total"),
        (_uncited_row(), "extracted without evidence (quote the printed text): line_items[2]"),
        (_page_four(), "page outside 1..3 (the document has 3 pages): tax"),
    ],
    ids=["invalid-json", "extracted-null", "uncited-field", "uncited-row", "page-beyond"],
)
async def test_malformed_output_is_repaired(
    fake_llm: FakeLLMServer, three_pages: list[Path], bad: Any, problem: str
) -> None:
    fake_llm.enqueue(chat_reply(bad), chat_reply(answer()))
    result = await run(fake_llm, three_pages)
    assert result.response.repairs == 1 and len(fake_llm.requests) == 2
    assert problem in result.response.repair_history[0].problem
    raw = bad if isinstance(bad, str) else json.dumps(bad)
    assert result.response.repair_history[0].raw_text == raw
    # the re-ask names the problem and keeps the pages: same system prompt and images
    repair = fake_llm.requests[1].body["messages"]
    assert problem in repair[-1]["content"]
    assert sent_images(fake_llm.requests[1].body) == [p.read_bytes() for p in three_pages]
    assert result.parsed.total.value is not None


async def test_output_still_malformed_after_the_repairs_is_an_error(
    fake_llm: FakeLLMServer, three_pages: list[Path]
) -> None:
    fake_llm.enqueue(chat_reply(_uncited_total()), chat_reply(_page_four()))
    with pytest.raises(ResponseParseError, match="page outside 1..3") as caught:
        await run(fake_llm, three_pages)
    assert caught.value.repairs == 1 and len(fake_llm.requests) == 2
    assert "extracted without evidence" in caught.value.history[0].problem


async def test_with_no_repairs_allowed_the_first_failure_is_final(
    fake_llm: FakeLLMServer, three_pages: list[Path]
) -> None:
    fake_llm.enqueue(chat_reply(_uncited_total()))
    with pytest.raises(ResponseParseError, match="extracted without evidence"):
        await run(fake_llm, three_pages, llm_max_repairs=0)
    assert len(fake_llm.requests) == 1


# ------------------------------------------------------------------ refused before any request


async def test_more_pages_than_allowed_are_refused_without_a_request(
    fake_llm: FakeLLMServer, three_pages: list[Path]
) -> None:
    with pytest.raises(ExtractionRefused, match="3 pages, but LLM_MAX_PAGES is 2"):
        await run(fake_llm, three_pages, llm_max_pages=2)
    assert fake_llm.requests == []
    result = await run(fake_llm, three_pages, llm_max_pages=3)  # the limit itself is allowed
    assert result.pages == 3


async def test_no_pages_are_refused_without_a_request(fake_llm: FakeLLMServer) -> None:
    with pytest.raises(ExtractionRefused, match="no pages"):
        await run(fake_llm, [])
    assert fake_llm.requests == []


# ------------------------------------------------------------------ the M4 verdict


def _payload_field(value: Any, evidence: str, confidence: float = 0.95) -> dict[str, Any]:
    return {
        "value": value,
        "confidence": confidence,
        "status": "extracted",
        "evidence": evidence,
        "page": 1,
        "reasons": [],
    }


async def test_the_validated_copy_carries_the_confidence_caps(
    fake_llm: FakeLLMServer, three_pages: list[Path]
) -> None:
    # the fake's tax id ("12-3456789", 9 digits) is not NPWP-shaped: a shape cap on the copy
    fake_llm.enqueue(chat_reply(canned_answer()))
    result = await run(fake_llm, three_pages)
    assert result.parsed.vendor_tax_id.reasons == []
    assert result.parsed.vendor_tax_id.confidence == 0.95
    assert result.validated.vendor_tax_id.reasons == [TAX_ID_UNUSUAL_SHAPE]
    assert result.validated.vendor_tax_id.confidence == SHAPE_CAP


async def test_a_value_its_own_evidence_contradicts_is_abstained(
    fake_llm: FakeLLMServer, three_pages: list[Path]
) -> None:
    data = canned_answer()
    data["vendor_name"] = _payload_field("Other Shop", "CORNER CAFE DEMO")
    fake_llm.enqueue(chat_reply(data))
    result = await run(fake_llm, three_pages)
    assert result.parsed.vendor_name.value == "Other Shop"  # what the model said is kept
    assert result.validated.vendor_name.status == "abstained"
    assert result.validated.vendor_name.value is None
    assert result.validated.vendor_name.evidence == "CORNER CAFE DEMO"
    assert result.validated.vendor_name.reasons == [EVIDENCE_NOT_GROUNDED]


async def test_abstain_below_reaches_the_chain(
    fake_llm: FakeLLMServer, three_pages: list[Path]
) -> None:
    fake_llm.enqueue(chat_reply(canned_answer()))
    result = await run(fake_llm, three_pages, abstain_below=0.99)
    assert result.parsed.vendor_name.status == "extracted"
    assert result.validated.vendor_name.status == "abstained"
    assert result.validated.vendor_name.reasons == [CONFIDENCE_BELOW_THRESHOLD]


async def test_a_rewritten_value_is_capped_by_the_normalizations(
    fake_llm: FakeLLMServer, three_pages: list[Path]
) -> None:
    data = canned_answer()
    data["total"] = _payload_field("17,28", "TOTAL $17.28")
    fake_llm.enqueue(chat_reply(data))
    result = await run(fake_llm, three_pages)
    assert result.response.parsed.total.value == Decimal("17.28")
    assert list(result.response.normalizations) == ["total.value: '17,28' -> '17.28'"]
    assert result.validated.total.confidence == NORMALIZED_CAP
