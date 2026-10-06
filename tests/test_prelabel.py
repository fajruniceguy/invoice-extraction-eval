import base64
import io
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from docextract.config import Settings
from docextract.ingest.manifest import ManifestRow
from docextract.providers.openai_compat import OpenAICompatClient, ResponseParseError
from evalkit.groundtruth import UNLABELED, GroundTruth
from evalkit.gt_store import dumps_label, parse_label, read_label, write_label
from evalkit.prelabel import (
    MAX_IMAGE_BYTES,
    SYSTEM_PROMPT,
    BlindDocumentError,
    PrelabelDraft,
    PrelabelRefused,
    check_eligible,
    check_model_differs,
    draft_to_label,
    eligibility,
    prelabel_document,
    save_draft,
    stored_pages,
)
from fakes.fake_llm import FakeLLMServer, chat_reply
from files import A4, make_sized_pdf
from ingest_support import ROOT, Env, gray_png, settings, write_manifest_file


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a real LLM_* / PRELABEL_* variable (or a real key) into these tests."""
    for name in list(os.environ):
        if name.startswith(("LLM_", "PRELABEL_", "MAX_RUN_COST")) and name not in (
            "LLM_CACHE",
            "LLM_CACHE_DIR",
        ):
            monkeypatch.delenv(name)


GOOD: dict[str, Any] = {
    "vendor_name": "PT Contoh Makmur",
    "vendor_address": "Jl. Merdeka 12, Bandung",
    "vendor_tax_id": "01.234.567.8-901.000",
    "buyer_name": "CV Pembeli",
    "document_number": "INV/2024/0457",
    "document_date": "5 Maret 2024",
    "due_date": None,
    "currency": "Rp",
    "payment_method": "Transfer",
    "line_items": [
        {
            "description": "Kertas A4",
            "quantity": "10",
            "unit_price": "45.000,00",
            "amount": "450.000,00",
        }
    ],
    "subtotal": "450.000,00",
    "tax": "49.500,00",
    "discount": None,
    "total": "499.500,00",
    "notes": None,
}


def draft(**changes: Any) -> PrelabelDraft:
    return PrelabelDraft.model_validate({**GOOD, **changes})


def label_of(**changes: Any) -> GroundTruth:
    return draft_to_label(draft(**changes), doc_id="doc_001", model="claude-haiku-4-5")


def row(doc_id: str = "doc_001", **overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "doc_id": doc_id,
        "raw_files": f"{doc_id}.png",
        "content_group": doc_id,
        "doc_type": "faktur",
        "label_mode": "prelabel",
        "needs_redaction": "n",
    }
    values.update(overrides)
    if values.get("variant"):
        values.update(raw_files="", capture="synthetic_degraded")
    return ManifestRow.model_validate(values)


# ------------------------------------------------------------------ draft -> label


class TestDraftToLabel:
    def test_a_good_transcript_becomes_a_complete_draft(self) -> None:
        label = label_of()
        assert label.label_mode == "prelabel" and label.labeled_at is None
        assert label.vendor_name == "PT Contoh Makmur"
        assert label.vendor_tax_id == "01.234.567.8-901.000"
        assert str(label.document_date) == "2024-03-05"
        assert label.due_date is None  # not printed
        assert label.currency == "IDR"
        assert label.subtotal == Decimal("450000.00") and label.total == Decimal("499500.00")
        assert label.discount is None
        assert [item.model_dump() for item in label.line_items] == [
            {
                "description": "Kertas A4",
                "quantity": Decimal(10),
                "unit_price": Decimal("45000.00"),
                "amount": Decimal("450000.00"),
            }
        ]
        assert label.field_notes == {}

    def test_the_label_is_recorded_as_an_unverified_draft_by_the_model(self) -> None:
        assert label_of().notes == "pre-label draft by claude-haiku-4-5; unverified"
        assert label_of(notes="the stamp is unreadable").notes == (
            "pre-label draft by claude-haiku-4-5; unverified. the stamp is unreadable"
        )
        assert label_of(notes="  ").notes == "pre-label draft by claude-haiku-4-5; unverified"

    def test_null_means_not_printed_and_a_missing_key_means_not_answered(self) -> None:
        raw = {k: v for k, v in GOOD.items() if k != "buyer_name"}
        label = draft_to_label(PrelabelDraft.model_validate(raw), doc_id="d", model="m")
        assert label.buyer_name == UNLABELED
        assert label.field_notes == {"buyer_name": "the model did not answer this field"}
        assert label.due_date is None  # an explicit null

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_string_is_treated_as_not_printed(self, blank: str) -> None:
        assert label_of(vendor_tax_id=blank).vendor_tax_id is None

    def test_text_is_stripped_but_otherwise_left_as_printed(self) -> None:
        assert label_of(vendor_name="  PT  Contoh   Makmur ").vendor_name == "PT  Contoh   Makmur"

    @pytest.mark.parametrize(
        ("printed", "currency", "expected"),
        [
            ("Rp 1.250.000,00", "Rp", "1250000.00"),
            ("1.250.000,00", "IDR", "1250000.00"),
            ("9.116,8", "Rp", "9116.8"),
            ("1.250.000", "Rp", "1250000"),
            ("1.250", "Rp", "1250"),  # a rupiah context: '.' groups thousands
            ("1,250,000.00", "USD", "1250000.00"),
            ("$12.50", "USD", "12.50"),
            ("1.250", "USD", "1.250"),  # a dollar context: '.' is the decimal mark
            ("(1.500)", "Rp", "-1500"),
        ],
    )
    def test_money_goes_through_the_shared_normalizer(
        self, printed: str, currency: str, expected: str
    ) -> None:
        assert label_of(total=printed, currency=currency).total == Decimal(expected)

    def test_money_keeps_the_printed_scale(self) -> None:
        dumped = json.loads(dumps_label(label_of(total="1.250.000,00", subtotal="1.250.000")))
        assert (dumped["total"], dumped["subtotal"]) == ("1250000.00", "1250000")

    def test_an_ambiguous_number_is_left_for_the_human_not_guessed(self) -> None:
        label = label_of(total="1.250", currency=None)
        assert label.total == UNLABELED
        assert label.field_notes["total"].startswith("ambiguous number: printed '1.250'")
        # the same text with a currency is no longer ambiguous
        assert label_of(total="1.250", currency="Rp").total == Decimal(1250)

    def test_an_unreadable_number_is_left_for_the_human(self) -> None:
        label = label_of(tax="abc")
        assert label.tax == UNLABELED
        assert label.field_notes["tax"].startswith("unreadable number: printed 'abc'")

    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            ("5 Maret 2024", "2024-03-05"),
            ("2024-03-05", "2024-03-05"),
            ("05/03/2024", "2024-03-05"),
        ],
    )
    def test_dates_become_iso_and_an_idr_document_is_read_day_first(
        self, printed: str, expected: str
    ) -> None:
        assert str(label_of(document_date=printed, currency="Rp").document_date) == expected

    def test_an_ambiguous_date_without_a_currency_is_left_for_the_human(self) -> None:
        label = label_of(document_date="05/03/2024", currency=None)
        assert label.document_date == UNLABELED
        assert label.field_notes["document_date"].startswith("ambiguous date: printed '05/03/2024'")

    def test_an_impossible_date_is_unreadable(self) -> None:
        label = label_of(due_date="31/02/2024", currency="Rp")
        assert label.due_date == UNLABELED
        assert label.field_notes["due_date"].startswith("unreadable date")

    @pytest.mark.parametrize(
        ("printed", "code"),
        [
            ("Rp", "IDR"),
            ("Rp.", "IDR"),
            ("rupiah", "IDR"),
            ("IDR", "IDR"),
            ("$", "USD"),
            ("usd", "USD"),
            ("SGD", "SGD"),
            ("€", "EUR"),
        ],
    )
    def test_currencies(self, printed: str, code: str) -> None:
        assert label_of(currency=printed).currency == code

    @pytest.mark.parametrize(
        ("printed", "code"), [("CHF", "CHF"), ("aud", "AUD"), (" jpy ", "JPY")]
    )
    def test_any_other_three_letter_code_is_kept(self, printed: str, code: str) -> None:
        assert label_of(currency=printed).currency == code

    @pytest.mark.parametrize("printed", ["12$", "ab", "rupee-ish", "R.p"])
    def test_anything_else_is_not_a_currency(self, printed: str) -> None:
        assert label_of(currency=printed).currency == UNLABELED

    def test_an_unknown_currency_is_left_for_the_human(self) -> None:
        label = label_of(currency="rupee-ish")
        assert label.currency == UNLABELED
        assert label.field_notes["currency"] == "unknown currency: printed 'rupee-ish'"

    def test_decimals_and_floats_are_written_out_never_in_exponent_form(self) -> None:
        # str(Decimal("1E+6")) is "1E+6" and str(1e-07) is "1e-07": neither is a printed number
        label = label_of(total=Decimal("1E+6"), tax=1e-07, currency="USD")
        assert label.total == Decimal(1_000_000) and label.field_notes == {}
        assert label.tax == Decimal("0.0000001")

    def test_numbers_in_the_json_are_tolerated(self) -> None:
        label = label_of(total=499500, tax=Decimal("49500.5"), subtotal=450000.0)
        assert (label.total, label.tax, label.subtotal) == (
            Decimal(499500),
            Decimal("49500.5"),
            Decimal("450000.0"),
        )


class TestLineItems:
    def items(self, **changes: Any) -> Any:
        return label_of(**changes).line_items

    def test_cells_are_normalized_and_nulls_are_absent_cells(self) -> None:
        rows = self.items(
            line_items=[
                {"description": "A", "quantity": "2,5", "unit_price": "1.250,00", "amount": None},
                {"description": " B ", "quantity": None, "unit_price": None, "amount": "3.000"},
            ]
        )
        assert [r.model_dump() for r in rows] == [
            {
                "description": "A",
                "quantity": Decimal("2.5"),
                "unit_price": Decimal("1250.00"),
                "amount": None,
            },
            {"description": "B", "quantity": None, "unit_price": None, "amount": Decimal(3000)},
        ]

    def test_null_means_no_itemization_and_a_missing_key_means_not_answered(self) -> None:
        assert self.items(line_items=None) is None
        raw = {k: v for k, v in GOOD.items() if k != "line_items"}
        label = draft_to_label(PrelabelDraft.model_validate(raw), doc_id="d", model="m")
        assert label.line_items == UNLABELED
        assert label.field_notes["line_items"] == "the model did not answer this field"

    def test_a_cell_the_model_omitted_is_unlabeled_not_absent(self) -> None:
        rows = self.items(line_items=[{"description": "A", "amount": "5.000"}])
        assert rows[0].quantity == UNLABELED and rows[0].unit_price == UNLABELED
        assert rows[0].amount == Decimal(5000)

    def test_ambiguous_cells_are_left_and_listed_in_one_note(self) -> None:
        label = label_of(
            currency=None,
            line_items=[
                {"description": "A", "quantity": "1", "unit_price": "1.250", "amount": "1.250"},
                {"description": "B", "quantity": "x", "unit_price": "10", "amount": "10"},
            ],
        )
        rows = label.line_items
        assert rows[0].unit_price == UNLABELED and rows[0].amount == UNLABELED
        assert rows[1].quantity == UNLABELED and rows[1].unit_price == Decimal(10)
        note = label.field_notes["line_items"]
        assert "row 1 unit_price: ambiguous '1.250'" in note
        assert "row 1 amount: ambiguous '1.250'" in note
        assert "row 2 quantity: unreadable 'x'" in note

    def test_descriptions_are_never_parsed_as_numbers(self) -> None:
        rows = self.items(
            line_items=[{"description": "1.250", "quantity": "1", "unit_price": "5", "amount": "5"}]
        )
        assert rows[0].description == "1.250"

    def test_an_empty_table_is_an_empty_list_not_null(self) -> None:
        assert self.items(line_items=[]) == []


class TestTheDraftIsAValidLabelFile:
    def test_it_round_trips_through_the_label_store(self) -> None:
        label = label_of(total="1.250", currency=None)
        assert parse_label(dumps_label(label), source="x") == label

    def test_every_field_is_decided_explicitly_in_the_file(self) -> None:
        data = json.loads(dumps_label(label_of(due_date=None)))
        assert data["label_mode"] == "prelabel" and "labeled_at" not in data
        assert data["due_date"] is None and data["doc_id"] == "doc_001"


# ------------------------------------------------------------------ what may be drafted


class TestEligibility:
    def test_a_hand_labeled_prelabel_document_is_eligible(self) -> None:
        check_eligible(row())
        assert eligibility(row()) is None

    def test_a_blind_document_is_refused_with_the_blind_error(self) -> None:
        with pytest.raises(BlindDocumentError, match="doc_007 is a BLIND document"):
            check_eligible(row("doc_007", label_mode="blind"))
        assert isinstance(BlindDocumentError("x"), PrelabelRefused)

    def test_the_blind_rule_wins_over_every_other_reason(self) -> None:
        with pytest.raises(BlindDocumentError):
            check_eligible(row("doc_007", label_mode="blind", gt_from="doc_001"))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"label_mode": None}, "has no label_mode: run scripts/assign_split.py"),
            ({"gt_from": "doc_009"}, "takes its labels from doc_009"),
            ({"capture": "constructed"}, "is constructed: its label is generated"),
        ],
    )
    def test_other_documents_are_refused_with_a_reason(
        self, overrides: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(PrelabelRefused, match=message) as caught:
            check_eligible(row("doc_002", **overrides))
        assert not isinstance(caught.value, BlindDocumentError)
        assert message in (eligibility(row("doc_002", **overrides)) or "")

    def test_a_degraded_variant_is_not_drafted(self) -> None:
        variant = row("doc_001-deg-light", variant="light", gt_from="doc_001", label_mode=None)
        with pytest.raises(PrelabelRefused, match="takes its labels from doc_001"):
            check_eligible(variant)


class TestModelMustDiffer:
    def settings(self, **changes: Any) -> Settings:
        values: dict[str, Any] = {
            "llm_base_url": "https://api.anthropic.com/v1",
            "llm_model": "claude-sonnet-5-5",
            "prelabel_base_url": "https://api.anthropic.com/v1/",
            "prelabel_model": "claude-haiku-4-5",
        }
        values.update(changes)
        return Settings(_env_file=None, **values)

    def test_a_different_model_is_accepted(self) -> None:
        check_model_differs(self.settings())

    def test_the_same_model_on_the_same_endpoint_is_refused(self) -> None:
        with pytest.raises(PrelabelRefused, match="is the extractor's .* on the same endpoint"):
            check_model_differs(self.settings(llm_model="claude-haiku-4-5"))

    def test_unknown_extractor_settings_skip_the_check(self) -> None:
        check_model_differs(None)


# ------------------------------------------------------------------ the pages


class TestStoredPages:
    def ingested(self, env: Env, name: str = "doc_001") -> Any:
        gray_png(env, f"{name}.png")
        r = row(name)
        env.ingest(r)
        return r

    def test_it_returns_the_stored_pages_in_order(self, env: Env) -> None:
        make_sized_pdf(env.raw / "doc_001.png", [A4, A4, A4])  # a pdf, whatever its name says
        r = row()
        env.ingest(r, settings=settings(dpi=40))
        assert [p.name for p in stored_pages("doc_001", env.pages)] == [
            "p1.png",
            "p2.png",
            "p3.png",
        ]

    def test_it_never_needs_the_raw_files(self, env: Env) -> None:
        self.ingested(env)
        for raw in env.raw.iterdir():
            raw.unlink()
        env.raw.rmdir()
        assert [p.name for p in stored_pages("doc_001", env.pages)] == ["p1.png"]

    def test_a_document_that_was_not_ingested_is_refused(self, env: Env) -> None:
        with pytest.raises(PrelabelRefused, match="doc_001: not ingested yet"):
            stored_pages("doc_001", env.pages)

    def test_a_page_that_changed_since_ingestion_is_refused(self, env: Env) -> None:
        self.ingested(env)
        page = env.pages / "doc_001" / "p1.png"
        page.write_bytes(page.read_bytes() + b"x")
        with pytest.raises(PrelabelRefused, match="no longer matches ingest.json"):
            stored_pages("doc_001", env.pages)

    def test_a_missing_page_is_refused(self, env: Env) -> None:
        self.ingested(env)
        (env.pages / "doc_001" / "p1.png").unlink()
        with pytest.raises(PrelabelRefused, match="stored page p1.png is missing"):
            stored_pages("doc_001", env.pages)


# ------------------------------------------------------------------ calling the model


def prelabel_settings(**changes: Any) -> tuple[Settings, Settings]:
    """(the extractor's settings, the settings of the pre-labeling client)."""
    values: dict[str, Any] = {
        "llm_base_url": "https://extractor.test/v1/",
        "llm_model": "gemini-x",
        "llm_api_key": "sk-extractor",
        "llm_reasoning_effort": "low",
        "llm_structured_mode": "json_schema",
        "llm_max_repairs": 1,
        "llm_max_retries": 0,
        "prelabel_base_url": "https://api.anthropic.com/v1/",
        "prelabel_api_key": "sk-prelabel",
        "prelabel_model": "claude-haiku-4-5",
        "prelabel_price_input_per_mtok": "1",
        "prelabel_price_output_per_mtok": "5",
    }
    values.update(changes)
    extractor = Settings(_env_file=None, **values)
    return extractor, extractor.prelabel_settings()


class Recorder:
    def __init__(self, *replies: dict[str, Any] | str) -> None:
        self.requests: list[httpx.Request] = []
        self.replies = list(replies)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if self.replies else GOOD
        return httpx.Response(200, json=chat_reply(reply).body)

    def body(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def client_for(ps: Settings, recorder: Recorder) -> OpenAICompatClient:
    return OpenAICompatClient(ps, transport=httpx.MockTransport(recorder))


@pytest.fixture
def pages(env: Env) -> Path:
    gray_png(env, "doc_001.png", size=(64, 48))
    env.ingest(row())
    return env.pages


class TestPrelabelDocument:
    async def test_it_drafts_from_the_stored_pages_with_the_prelabel_model(
        self, pages: Path
    ) -> None:
        extractor, ps = prelabel_settings()
        recorder = Recorder(GOOD)
        async with client_for(ps, recorder) as client:
            result = await prelabel_document(
                row(), client=client, settings=ps, pages_dir=pages, extractor=extractor
            )
        assert result.doc_id == "doc_001" and result.pages == 1
        assert result.label.total == Decimal("499500.00")
        assert result.response.parsed is not None and result.response.repairs == 0
        assert len(recorder.requests) == 1

        request, body = recorder.requests[0], recorder.body()
        assert str(request.url) == "https://api.anthropic.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-prelabel"  # not the extractor's key
        assert body["model"] == "claude-haiku-4-5"
        assert body["temperature"] == 0.0 and body["max_tokens"] == ps.prelabel_max_tokens
        assert body["response_format"] == {"type": "json_object"}  # pinned, never json_schema
        assert "reasoning_effort" not in body  # the extractor's setting does not leak across

    async def test_the_schema_goes_into_the_prompt_because_the_endpoint_may_ignore_response_format(
        self, pages: Path
    ) -> None:
        _, ps = prelabel_settings()
        recorder = Recorder()
        async with client_for(ps, recorder) as client:
            await prelabel_document(row(), client=client, settings=ps, pages_dir=pages)
        system = recorder.body()["messages"][0]
        assert system["role"] == "system"
        assert system["content"].startswith(SYSTEM_PROMPT)
        assert '"vendor_name"' in system["content"] and '"line_items"' in system["content"]

    async def test_what_is_sent_is_exactly_the_stored_page(self, pages: Path) -> None:
        _, ps = prelabel_settings()
        recorder = Recorder()
        async with client_for(ps, recorder) as client:
            await prelabel_document(row(), client=client, settings=ps, pages_dir=pages)
        content = recorder.body()["messages"][1]["content"]
        assert content[0] == {
            "type": "text",
            "text": "Transcribe this document (1 page, in order).",
        }
        sent = content[1]["image_url"]["url"]
        assert sent.startswith("data:image/png;base64,")
        assert (
            base64.b64decode(sent.split(",", 1)[1]) == (pages / "doc_001" / "p1.png").read_bytes()
        )

    async def test_all_pages_go_in_one_request_in_order(self, env: Env) -> None:
        make_sized_pdf(env.raw / "doc_001.png", [A4, A4])
        env.ingest(row(), settings=settings(dpi=40))
        _, ps = prelabel_settings()
        recorder = Recorder()
        async with client_for(ps, recorder) as client:
            result = await prelabel_document(row(), client=client, settings=ps, pages_dir=env.pages)
        content = recorder.body()["messages"][1]["content"]
        assert result.pages == 2 and len(recorder.requests) == 1
        assert content[0]["text"] == "Transcribe this document (2 pages, in order)."
        assert [part["type"] for part in content[1:]] == ["image_url", "image_url"]

    async def test_a_page_larger_than_the_prelabel_size_is_downscaled_in_memory_only(
        self, env: Env
    ) -> None:
        Image.new("RGB", (3000, 2000), (128, 128, 128)).save(env.raw / "doc_001.png", "PNG")
        env.ingest(row(), settings=settings(max_long_edge=3000))
        stored = (env.pages / "doc_001" / "p1.png").read_bytes()
        _, ps = prelabel_settings(prelabel_image_max_long_edge=1000)
        recorder = Recorder()
        async with client_for(ps, recorder) as client:
            await prelabel_document(row(), client=client, settings=ps, pages_dir=env.pages)
        url = recorder.body()["messages"][1]["content"][1]["image_url"]["url"]
        sent = base64.b64decode(url.split(",", 1)[1])
        assert Image.open(io.BytesIO(sent)).size == (1000, 667)
        assert (
            env.pages / "doc_001" / "p1.png"
        ).read_bytes() == stored  # the stored page is untouched

    async def test_a_page_over_the_byte_limit_is_sent_as_a_smaller_derivative(
        self, env: Env
    ) -> None:
        import random

        noise = random.Random(1).randbytes(1600 * 1200 * 3)  # incompressible: a ~5.8 MB PNG
        Image.frombytes("RGB", (1600, 1200), noise).save(env.raw / "doc_001.png", "PNG")
        env.ingest(row())
        stored = (env.pages / "doc_001" / "p1.png").read_bytes()
        assert len(stored) > MAX_IMAGE_BYTES
        _, ps = prelabel_settings(
            prelabel_image_max_long_edge=2400
        )  # the size rule alone is a no-op
        recorder = Recorder()
        async with client_for(ps, recorder) as client:
            await prelabel_document(row(), client=client, settings=ps, pages_dir=env.pages)
        url = recorder.body()["messages"][1]["content"][1]["image_url"]["url"]
        sent = base64.b64decode(url.split(",", 1)[1])
        assert len(sent) <= MAX_IMAGE_BYTES
        assert (env.pages / "doc_001" / "p1.png").read_bytes() == stored

    async def test_unusable_output_is_re_asked_once(self, pages: Path) -> None:
        _, ps = prelabel_settings()
        recorder = Recorder("here is your JSON: oops", GOOD)
        async with client_for(ps, recorder) as client:
            result = await prelabel_document(row(), client=client, settings=ps, pages_dir=pages)
        assert result.response.repairs == 1 and len(recorder.requests) == 2
        assert result.label.total == Decimal("499500.00")

    async def test_output_that_stays_unusable_is_an_error(self, pages: Path) -> None:
        _, ps = prelabel_settings()
        recorder = Recorder("nope", "still nope")
        async with client_for(ps, recorder) as client:
            with pytest.raises(ResponseParseError):
                await prelabel_document(row(), client=client, settings=ps, pages_dir=pages)

    async def test_the_cost_is_recorded_from_the_prelabel_prices(self, pages: Path) -> None:
        _, ps = prelabel_settings()
        recorder = Recorder()
        async with client_for(ps, recorder) as client:
            result = await prelabel_document(row(), client=client, settings=ps, pages_dir=pages)
        # the canned usage is 1500 in / 700 out tokens: $1 and $5 per million
        assert result.response.cost_usd == Decimal("0.005000")
        assert result.unlabeled_fields == []


class TestNothingIsSentWhenRefused:
    """Every refusal happens BEFORE a request: a refused document costs and leaks nothing."""

    async def refuse(self, r: ManifestRow, pages: Path, error: type[Exception], **kw: Any) -> None:
        extractor, ps = prelabel_settings(**kw.pop("settings", {}))
        recorder = Recorder()
        async with client_for(ps, recorder) as client:
            with pytest.raises(error):
                await prelabel_document(
                    r,
                    client=client,
                    settings=ps,
                    pages_dir=pages,
                    extractor=kw.pop("extractor", extractor),
                )
        assert recorder.requests == []

    async def test_a_blind_document(self, pages: Path) -> None:
        await self.refuse(row(label_mode="blind"), pages, BlindDocumentError)

    async def test_a_blind_document_even_when_its_pages_exist_and_the_model_differs(
        self, pages: Path
    ) -> None:
        # the strongest form of the rule: everything else is in order, only the mode says blind
        await self.refuse(row("doc_001", label_mode="blind"), pages, BlindDocumentError)

    @pytest.mark.parametrize(
        "overrides", [{"label_mode": None}, {"gt_from": "doc_009"}, {"capture": "constructed"}]
    )
    async def test_documents_that_are_not_hand_labeled_prelabel_documents(
        self, pages: Path, overrides: dict[str, Any]
    ) -> None:
        await self.refuse(row(**overrides), pages, PrelabelRefused)

    async def test_the_extractors_own_model(self, pages: Path) -> None:
        await self.refuse(
            row(),
            pages,
            PrelabelRefused,
            settings={
                "llm_base_url": "https://api.anthropic.com/v1",
                "llm_model": "claude-haiku-4-5",
            },
        )

    async def test_a_document_that_was_not_ingested(self, env: Env) -> None:
        await self.refuse(row(), env.pages, PrelabelRefused)

    async def test_a_bundle_longer_than_the_page_limit(self, env: Env) -> None:
        make_sized_pdf(env.raw / "doc_001.png", [A4, A4, A4])
        env.ingest(row(), settings=settings(dpi=40))
        await self.refuse(row(), env.pages, PrelabelRefused, settings={"prelabel_max_pages": 2})


class TestSaveDraft:
    def test_the_draft_is_written_to_the_drafts_directory(self, tmp_path: Path) -> None:
        path = save_draft(row(), label_of(), tmp_path / "drafts")
        assert path == tmp_path / "drafts" / "doc_001.json"
        assert read_label(path) == label_of()

    def test_a_blind_document_cannot_get_a_draft_written_by_any_caller(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(BlindDocumentError):
            save_draft(row(label_mode="blind"), label_of(), tmp_path / "drafts")
        assert not (tmp_path / "drafts").exists()

    def test_a_draft_for_another_document_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(PrelabelRefused, match="the draft is for doc_001, not doc_002"):
            save_draft(row("doc_002"), label_of(), tmp_path / "drafts")
        assert not (tmp_path / "drafts").exists()

    def test_an_existing_draft_is_replaced_atomically(self, tmp_path: Path) -> None:
        drafts = tmp_path / "drafts"
        write_label(drafts, label_of(total="1.000"))
        save_draft(row(), label_of(total="2.000"), drafts)
        assert read_label(drafts / "doc_001.json").total == Decimal(2000)
        assert [p.name for p in drafts.iterdir()] == ["doc_001.json"]


# ------------------------------------------------------------------ the command line


def scrubbed_env(tmp_path: Path) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("LLM_", "PRELABEL_", "MAX_RUN_COST"))
    }
    env.update(LLM_CACHE="off", LLM_CACHE_DIR=str(tmp_path / "cache"))
    return env


@pytest.mark.slow
class TestCommandLine:
    @pytest.fixture
    def dataset(self, tmp_path: Path) -> dict[str, Any]:
        env = Env(tmp_path)
        for name in ("doc_001", "doc_002", "doc_008"):
            gray_png(env, f"{name}.png", size=(64, 48))
        rows = [
            row("doc_001"),
            row("doc_002"),
            row("doc_003", label_mode="blind"),
            row("doc_004", capture="constructed", label_mode=None),
            row("doc_005", gt_from="doc_001", content_group="doc_001", label_mode=None),
            row("doc_006", label_mode=None),
            row("doc_007", label_mode="blind"),
        ]
        for r in rows[:2]:
            env.ingest(r)
        return {
            "tmp": tmp_path,
            "env": env,
            "manifest": write_manifest_file(tmp_path, rows),
            "drafts": tmp_path / "gt" / "drafts",
            "gt": tmp_path / "gt",
        }

    def run(
        self, data: dict[str, Any], server: FakeLLMServer, *args: str, **env_values: str
    ) -> subprocess.CompletedProcess[str]:
        tmp: Path = data["tmp"]
        env_file = tmp / "test.env"
        values = {
            "LLM_BASE_URL": server.base_url,
            "LLM_MODEL": "extractor-model",
            "LLM_API_KEY": "sk-extractor-secret",
            "PRELABEL_BASE_URL": server.base_url,
            "PRELABEL_MODEL": "prelabel-model",
            "PRELABEL_API_KEY": "sk-prelabel-secret",
            "PRELABEL_PRICE_INPUT_PER_MTOK": "1",
            "PRELABEL_PRICE_OUTPUT_PER_MTOK": "5",
            "LLM_MAX_RETRIES": "0",
        }
        values.update(env_values)
        env_file.write_text("\n".join(f"{k}={v}" for k, v in values.items() if v), encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "prelabel.py"),
                "--manifest", str(data["manifest"]),
                "--pages-dir", str(data["env"].pages),
                "--drafts-dir", str(data["drafts"]),
                "--ground-truth-dir", str(data["gt"]),
                "--env-file", str(env_file),
                *args,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=scrubbed_env(tmp),
            cwd=ROOT,
            timeout=120,
        )  # fmt: skip

    def test_all_drafts_every_eligible_document_and_only_those(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        fake_llm.enqueue(chat_reply(GOOD), chat_reply(GOOD))
        done = self.run(dataset, fake_llm, "--all")
        assert done.returncode == 0, done.stdout + done.stderr
        assert sorted(p.name for p in dataset["drafts"].iterdir()) == [
            "doc_001.json",
            "doc_002.json",
        ]
        assert len(fake_llm.requests) == 2
        out = done.stdout
        assert "skipped: 2 blind, 1 borrows labels, 1 constructed, 1 no label_mode" in out
        assert "drafted   doc_001  1 page(s), 1500 in / 700 out tokens, $0.0050" in out
        assert (
            "done: 2 drafted, 0 failed, 3000 in / 1400 out tokens, $0.0100 (prelabel-model)" in out
        )
        draft_file = json.loads((dataset["drafts"] / "doc_001.json").read_text(encoding="utf-8"))
        assert draft_file["label_mode"] == "prelabel" and draft_file["total"] == "499500.00"
        for blind in ("doc_003", "doc_007"):
            assert not (dataset["drafts"] / f"{blind}.json").exists()
        assert all(r.body["model"] == "prelabel-model" for r in fake_llm.requests)

    def test_naming_a_blind_document_is_an_error_and_nothing_is_called(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        done = self.run(dataset, fake_llm, "--doc", "doc_001", "--doc", "doc_003")
        assert done.returncode == 1
        assert "REFUSED: doc_003 is a BLIND document" in done.stderr
        assert fake_llm.requests == []
        assert not dataset["drafts"].exists()  # not even the eligible one was drafted

    def test_every_blind_document_is_refused_when_named(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        done = self.run(dataset, fake_llm, "--doc", "doc_003", "--doc", "doc_007")
        assert done.returncode == 1
        assert done.stderr.count("is a BLIND document") == 2 and fake_llm.requests == []

    @pytest.mark.parametrize(
        ("doc_id", "message"),
        [
            ("doc_004", "is constructed"),
            ("doc_005", "takes its labels from doc_001"),
            ("doc_006", "has no label_mode"),
        ],
    )
    def test_naming_other_ineligible_documents_is_an_error_too(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer, doc_id: str, message: str
    ) -> None:
        done = self.run(dataset, fake_llm, "--doc", doc_id)
        assert done.returncode == 1 and message in done.stderr and fake_llm.requests == []

    def test_dry_run_calls_nothing(self, dataset: dict[str, Any], fake_llm: FakeLLMServer) -> None:
        done = self.run(dataset, fake_llm, "--all", "--dry-run")
        assert done.returncode == 0
        assert "dry run: would draft 2 document(s): doc_001, doc_002" in done.stdout
        assert fake_llm.requests == [] and not dataset["drafts"].exists()

    def test_the_same_model_as_the_extractor_is_refused_before_any_call(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        done = self.run(dataset, fake_llm, "--all", PRELABEL_MODEL="extractor-model")
        assert done.returncode == 2
        assert "REFUSED: the pre-labeling model (extractor-model) is the extractor's" in done.stderr
        assert fake_llm.requests == []

    def test_allow_same_model_is_the_explicit_escape_hatch(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        fake_llm.enqueue(chat_reply(GOOD), chat_reply(GOOD))
        done = self.run(
            dataset, fake_llm, "--all", "--allow-same-model", PRELABEL_MODEL="extractor-model"
        )
        assert done.returncode == 0 and len(fake_llm.requests) == 2

    @pytest.mark.parametrize("missing", ["PRELABEL_MODEL", "PRELABEL_BASE_URL"])
    def test_unconfigured_pre_labeling_exits_2(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer, missing: str
    ) -> None:
        done = self.run(dataset, fake_llm, "--all", **{missing: ""})
        assert done.returncode == 2 and missing in done.stderr and fake_llm.requests == []

    def test_existing_drafts_and_finished_labels_are_kept_unless_forced(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        write_label(dataset["drafts"], label_of(vendor_name="Earlier draft"))
        write_label(dataset["gt"], draft_to_label(draft(), doc_id="doc_002", model="x"))
        done = self.run(dataset, fake_llm, "--all")
        assert done.returncode == 0 and "nothing to draft" in done.stdout
        assert "1 already drafted" in done.stdout and "1 already labeled" in done.stdout
        assert fake_llm.requests == []
        fake_llm.enqueue(chat_reply(GOOD))
        done = self.run(dataset, fake_llm, "--all", "--force")
        assert done.returncode == 0 and len(fake_llm.requests) == 1  # doc_002 stays labeled
        assert read_label(dataset["drafts"] / "doc_001.json").vendor_name == "PT Contoh Makmur"

    def test_a_failing_document_does_not_stop_the_others(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        fake_llm.enqueue(chat_reply("nope"), chat_reply("still nope"), chat_reply(GOOD))
        done = self.run(dataset, fake_llm, "--all")
        assert done.returncode == 1
        assert "FAILED" in done.stdout and "drafted   doc_002" in done.stdout
        assert "1 drafted, 1 failed" in done.stdout
        assert not (dataset["drafts"] / "doc_001.json").exists()
        assert (dataset["drafts"] / "doc_002.json").exists()

    def test_a_document_that_was_not_ingested_fails_alone(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        for sidecar in (dataset["env"].pages / "doc_001").glob("ingest.json"):
            sidecar.unlink()
        fake_llm.enqueue(chat_reply(GOOD))
        done = self.run(dataset, fake_llm, "--all")
        assert done.returncode == 1 and "doc_001: not ingested yet" in done.stdout
        assert len(fake_llm.requests) == 1 and (dataset["drafts"] / "doc_002.json").exists()

    def test_the_cost_cap_stops_the_run_and_keeps_what_was_done(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        fake_llm.enqueue(chat_reply(GOOD), chat_reply(GOOD))
        # each call costs $0.005, so the cap is crossed by the first and the second is refused
        done = self.run(dataset, fake_llm, "--all", PRELABEL_MAX_RUN_COST_USD="0.001")
        assert done.returncode == 3
        assert len(fake_llm.requests) == 1
        assert "STOPPED   doc_002" in done.stdout and "1 drafted" in done.stdout
        assert (dataset["drafts"] / "doc_001.json").exists()
        assert not (dataset["drafts"] / "doc_002.json").exists()

    def test_documents_are_drafted_one_at_a_time_by_default(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        slow = chat_reply(GOOD)
        slow.delay_s = 0.8
        fake_llm.enqueue(slow, chat_reply(GOOD))
        assert self.run(dataset, fake_llm, "--all").returncode == 0
        first, second = fake_llm.requests
        assert second.received_at - first.received_at >= 0.7  # the second waited for the first

    def test_concurrency_lets_requests_overlap(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        slow = chat_reply(GOOD)
        slow.delay_s = 0.8
        slow_too = chat_reply(GOOD)
        slow_too.delay_s = 0.8
        fake_llm.enqueue(slow, slow_too)
        done = self.run(dataset, fake_llm, "--all", "--concurrency", "2")
        assert done.returncode == 0
        first, second = fake_llm.requests
        assert abs(second.received_at - first.received_at) < 0.5  # in flight at the same time

    def test_concurrency(self, dataset: dict[str, Any], fake_llm: FakeLLMServer) -> None:
        fake_llm.enqueue(chat_reply(GOOD), chat_reply(GOOD))
        done = self.run(dataset, fake_llm, "--all", "--concurrency", "2")
        assert done.returncode == 0 and len(list(dataset["drafts"].iterdir())) == 2

    def test_no_key_is_ever_printed(self, dataset: dict[str, Any], fake_llm: FakeLLMServer) -> None:
        fake_llm.enqueue(chat_reply(GOOD), chat_reply(GOOD))
        done = self.run(dataset, fake_llm, "--all")
        assert "sk-prelabel-secret" not in done.stdout + done.stderr
        assert "sk-extractor-secret" not in done.stdout + done.stderr
        sent = [
            {k.lower(): v for k, v in r.headers.items()}["authorization"] for r in fake_llm.requests
        ]
        assert (
            sent == ["Bearer sk-prelabel-secret"] * 2
        )  # the pre-labeling key, not the extractor's

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            ([], "Choose the documents"),
            (["--all", "--doc", "doc_001"], "Choose the documents"),
            (["--doc", "doc_999"], "Not in the manifest: doc_999"),
            (["--all", "--concurrency", "0"], "--concurrency must be at least 1"),
        ],
    )
    def test_bad_arguments_exit_2(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer, args: list[str], message: str
    ) -> None:
        done = self.run(dataset, fake_llm, *args)
        assert done.returncode == 2 and message in done.stderr and fake_llm.requests == []

    def test_an_unreadable_manifest_exits_2(
        self, dataset: dict[str, Any], fake_llm: FakeLLMServer
    ) -> None:
        dataset["manifest"].write_text("nonsense\n", encoding="utf-8")
        done = self.run(dataset, fake_llm, "--all")
        assert done.returncode == 2 and "cannot be read" in done.stderr


def test_the_prompt_asks_for_the_printed_value_and_names_the_faktur_rules() -> None:
    for phrase in (
        "Report what is PRINTED",
        "never compute a value that is not printed",
        "A value hidden under a black box is null",
        "DPP",
        "PPN",
        "nomor seri faktur",
        "a faktur without a printed grand total has total null",
        "service charge",
        "one JSON object",
    ):
        assert phrase in SYSTEM_PROMPT, phrase
