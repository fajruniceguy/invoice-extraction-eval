import asyncio
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from docextract.config import Settings
from docextract.extract.prompt import PROMPT_SHA256, SYSTEM_PROMPT, prompt_sha256
from docextract.extract.runner import (
    DocRecord,
    RunRecord,
    SelectionError,
    blind_refusal,
    exit_code,
    run_extraction,
    select_documents,
)
from docextract.extract.schema import DocumentExtraction
from docextract.ingest.degrade import degrade_document, variant_row
from docextract.ingest.manifest import ManifestRow
from docextract.providers.cache import schema_hash
from docextract.providers.openai_compat import OpenAICompatClient
from fakes.fake_llm import FakeLLMServer, Reply, canned_answer, chat_reply
from files import A4, make_sized_pdf
from ingest_support import Env, gray_png, settings

KEY = "sk-runner-secret-key"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a real LLM_* variable (or a real key) into these tests."""
    for name in list(os.environ):
        if name.startswith(("LLM_", "PRELABEL_", "MAX_RUN_COST")) and name not in (
            "LLM_CACHE",
            "LLM_CACHE_DIR",
        ):
            monkeypatch.delenv(name)


def manifest_row(doc_id: str, **overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "doc_id": doc_id,
        "raw_files": f"{doc_id}.png",
        "content_group": doc_id,
        "doc_type": "invoice",
        "capture": "born_digital",
        "split": "dev",
        "needs_redaction": "n",
    }
    values.update(overrides)
    return ManifestRow.model_validate(values)


class Dataset:
    """Synthetic documents ingested under tmp_path, a fake model, and a run folder."""

    def __init__(self, env: Env, server: FakeLLMServer, tmp_path: Path) -> None:
        self.env = env
        self.server = server
        self.labels = tmp_path / "ground_truth"
        self.labels.mkdir()
        self.runs = tmp_path / "runs"
        self.rows: list[ManifestRow] = []

    def png(self, doc_id: str, *, ingest: bool = True, **overrides: Any) -> ManifestRow:
        row = manifest_row(doc_id, **overrides)
        gray_png(self.env, f"{doc_id}.png")
        if ingest:
            self.env.ingest(row)
        self.rows.append(row)
        return row

    def pdf(self, doc_id: str, pages: int, **overrides: Any) -> ManifestRow:
        row = manifest_row(doc_id, raw_files=f"{doc_id}.pdf", **overrides)
        make_sized_pdf(self.env.raw / f"{doc_id}.pdf", [A4, (420.0, 595.0), (300.0, 300.0)][:pages])
        self.env.ingest(row, settings=settings(dpi=40))
        self.rows.append(row)
        return row

    def variant(self, source: ManifestRow) -> ManifestRow:
        degrade_document(source, "light", pages_dir=self.env.pages, backend="opencv")
        row = variant_row(source, "light")
        self.rows.append(row)
        return row

    def settings(self, **changes: Any) -> Settings:
        values: dict[str, Any] = {
            "llm_base_url": self.server.base_url,
            "llm_model": "fake-extractor",
            "llm_api_key": KEY,
            "llm_max_repairs": 1,
            "llm_max_retries": 0,
        }
        values.update(changes)
        return Settings(_env_file=None, **values)

    async def run(
        self, selected: list[ManifestRow] | None = None, run_id: str = "r1", **changes: Any
    ) -> RunRecord:
        cache = changes.pop("cache", None)
        config = self.settings(**changes)
        async with OpenAICompatClient(config) as client:
            return await run_extraction(
                self.rows if selected is None else selected,
                rows=self.rows,
                client=client,
                settings=config,
                pages_dir=self.env.pages,
                labels_dir=self.labels,
                run_dir=self.runs / run_id,
                selection={"split": "dev"},
                cache=cache,
            )

    def record(self, doc_id: str, run_id: str = "r1") -> DocRecord:
        path = self.runs / run_id / "docs" / f"{doc_id}.json"
        return DocRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def run_json(self, run_id: str = "r1") -> dict[str, Any]:
        return json.loads((self.runs / run_id / "run.json").read_text(encoding="utf-8"))


@pytest.fixture
def data(env: Env, fake_llm: FakeLLMServer, tmp_path: Path) -> Dataset:
    return Dataset(env, fake_llm, tmp_path)


def uncited() -> dict[str, Any]:
    answer = canned_answer()
    answer["total"]["evidence"] = None
    return answer


def off_page() -> dict[str, Any]:
    answer = canned_answer()
    answer["tax"]["page"] = 7
    return answer


# ------------------------------------------------------------------ a synthetic dev set


class TestDevSet:
    """A 3-page document, its degraded variant, a document with no pages, one that gets HTTP 500,
    one whose output stays malformed after the repair, and a plain good one."""

    @pytest.fixture
    def outcome(self, data: Dataset) -> tuple[Dataset, RunRecord]:
        three = data.pdf("doc_a", 3)
        data.variant(three)
        data.png("doc_b", ingest=False)
        data.png("doc_c")
        data.png("doc_d")
        data.png("doc_e")
        data.server.enqueue(
            chat_reply(canned_answer()),  # doc_a
            chat_reply(canned_answer()),  # doc_a-deg-light
            Reply(500, {"error": {"message": "upstream exploded"}}),  # doc_c (no retries)
            chat_reply(uncited()),  # doc_d, then its repair is also unusable
            chat_reply(off_page()),
        )  # doc_e: the queue is empty, so the fake's default valid answer
        return data, _run(data)

    def test_every_document_gets_a_record_and_failures_do_not_stop_the_run(
        self, outcome: tuple[Dataset, RunRecord]
    ) -> None:
        data, run = outcome
        assert run.statuses == {
            "doc_a": "ok",
            "doc_a-deg-light": "ok",
            "doc_b": "refused",
            "doc_c": "failed",
            "doc_d": "failed",
            "doc_e": "ok",
        }
        assert sorted(p.stem for p in (data.runs / "r1" / "docs").iterdir()) == sorted(run.statuses)
        assert exit_code(run) == 1 and run.stopped is None
        assert len(data.server.requests) == 6  # a 1, variant 1, c 1, d 2 (a repair), e 1

    def test_a_multi_page_document_is_one_call_with_its_stored_pages(
        self, outcome: tuple[Dataset, RunRecord]
    ) -> None:
        data, _ = outcome
        record = data.record("doc_a")
        folder = data.env.pages / "doc_a"
        assert [(p.file, p.sha256) for p in record.pages] == [
            (name, _sha(folder / name)) for name in ("p1.png", "p2.png", "p3.png")
        ]
        assert record.parsed is not None and record.validated is not None
        assert record.parsed.vendor_name.value == "CORNER CAFE DEMO"
        assert record.raw_text == json.dumps(canned_answer())
        assert record.attempts == 1 and record.repairs == 0
        assert record.structured_mode_used == "json_schema" and record.finish_reason == "stop"
        assert record.usage is not None and record.usage["input_tokens"] == 1500

    def test_a_variant_sends_its_own_degraded_pages(
        self, outcome: tuple[Dataset, RunRecord]
    ) -> None:
        data, _ = outcome
        record = data.record("doc_a-deg-light")
        folder = data.env.pages / "doc_a-deg-light"
        assert [(p.file, p.sha256) for p in record.pages] == [
            (name, _sha(folder / name)) for name in ("p1.png", "p2.png", "p3.png")
        ]
        assert record.pages != data.record("doc_a").pages

    def test_failures_are_recorded_with_what_is_known(
        self, outcome: tuple[Dataset, RunRecord]
    ) -> None:
        data, _ = outcome
        missing = data.record("doc_b")
        assert missing.error is not None and missing.error.startswith("doc_b: not ingested yet")
        assert missing.pages == [] and missing.attempts == 0

        http = data.record("doc_c")
        assert http.error is not None and "HTTP 500" in http.error
        assert "upstream exploded" in http.error and http.parsed is None

        malformed = data.record("doc_d")
        assert malformed.status == "failed" and malformed.repairs == 1
        assert malformed.error is not None and "page outside 1..1" in malformed.error
        assert malformed.raw_text == json.dumps(off_page())  # the last output seen
        assert len(malformed.repair_history) == 1
        assert "extracted without evidence" in malformed.repair_history[0].problem
        assert malformed.repair_history[0].raw_text == json.dumps(uncited())
        assert malformed.attempts == 2 and malformed.usage is not None
        assert malformed.parsed is None and malformed.validated is None

    def test_run_json_identifies_the_run_and_adds_it_up(
        self, outcome: tuple[Dataset, RunRecord]
    ) -> None:
        data, run = outcome
        saved = RunRecord.model_validate(data.run_json())
        assert saved == run
        assert saved.run_id == "r1" and saved.model == "fake-extractor"
        assert saved.endpoint == data.server.base_url
        assert saved.prompt_sha256 == PROMPT_SHA256
        assert saved.schema_sha256 == schema_hash(DocumentExtraction)
        assert saved.selection == {"split": "dev"}
        assert saved.settings["max_pages"] == 15 and saved.settings["max_repairs"] == 1
        totals = saved.totals
        assert (totals.documents, totals.ok, totals.failed, totals.refused, totals.not_run) == (
            6,
            3,
            2,
            1,
            0,
        )
        assert totals.repairs == 1
        assert totals.input_tokens == 1500 * 5  # 3 ok calls + 2 calls of the malformed one
        assert totals.output_tokens == 700 * 5
        assert totals.cost_usd == Decimal(0) and saved.budget_spent_usd == Decimal(0)

    def test_the_api_key_is_never_written(self, outcome: tuple[Dataset, RunRecord]) -> None:
        data, _ = outcome
        written = list((data.runs / "r1").rglob("*.json"))
        assert len(written) == 7
        for path in written:
            assert KEY not in path.read_text(encoding="utf-8"), path


def _run(data: Dataset, **changes: Any) -> RunRecord:
    return asyncio.run(data.run(**changes))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def test_the_page_limit_refuses_a_document_without_a_request(data: Dataset) -> None:
    data.pdf("doc_a", 3)
    data.png("doc_e")
    run = await data.run(llm_max_pages=2)
    assert run.statuses == {"doc_a": "refused", "doc_e": "ok"}
    assert exit_code(run) == 1  # a refusal alone is enough for a non-zero exit
    assert "3 pages, but LLM_MAX_PAGES is 2" in str(data.record("doc_a").error)
    assert len(data.record("doc_a").pages) == 3
    assert len(data.server.requests) == 1


async def test_a_changed_stored_page_is_refused(data: Dataset) -> None:
    data.png("doc_a")
    page = data.env.pages / "doc_a" / "p1.png"
    page.write_bytes(page.read_bytes() + b"x")
    run = await data.run()
    assert run.statuses == {"doc_a": "refused"}
    assert "no longer matches ingest.json" in str(data.record("doc_a").error)
    assert data.server.requests == []


async def test_the_record_keeps_what_the_model_said_and_what_the_validators_made_of_it(
    data: Dataset,
) -> None:
    data.png("doc_a")
    answer = canned_answer()
    answer["line_items"][0]["amount"] = 99.0  # 2 x 4.50 is not 99
    data.server.enqueue(chat_reply(answer))
    await data.run()
    record = data.record("doc_a")
    assert record.parsed is not None and record.validated is not None
    assert record.parsed.line_items[0].flagged_cells == []
    assert record.validated.line_items[0].flagged_cells == ["quantity", "unit_price", "amount"]
    assert record.parsed.line_items[0].amount == Decimal("99.0")


async def test_a_repaired_document_is_ok_and_its_repair_is_recorded(data: Dataset) -> None:
    data.png("doc_a")
    data.server.enqueue(chat_reply(uncited()), chat_reply(canned_answer()))
    run = await data.run()
    record = data.record("doc_a")
    assert run.statuses == {"doc_a": "ok"} and exit_code(run) == 0
    assert record.repairs == 1 and run.totals.repairs == 1
    assert "extracted without evidence" in record.repair_history[0].problem


# ------------------------------------------------------------------ the cost cap


def capped(**changes: Any) -> dict[str, Any]:
    # 1500 prompt tokens at $1000 per 1M = $1.50 per call; the cap is $1.00
    return {
        "llm_price_input_per_mtok": "1000",
        "llm_price_output_per_mtok": "0",
        "max_run_cost_usd": "1.00",
        **changes,
    }


async def test_the_cost_cap_stops_the_run_and_keeps_what_was_done(data: Dataset) -> None:
    for doc_id in ("doc_a", "doc_b", "doc_c"):
        data.png(doc_id)
    run = await data.run(**capped())
    assert run.statuses == {"doc_a": "ok", "doc_b": "not_run", "doc_c": "not_run"}
    assert run.stopped == "cost_cap" and exit_code(run) == 3
    assert len(data.server.requests) == 1  # the call that crossed the cap completed, no more
    assert data.record("doc_a").cost_usd == Decimal("1.5")
    assert "run cost cap exceeded" in str(data.record("doc_b").error)
    assert data.record("doc_b").pages != []  # it got as far as its pages, not to a call
    assert data.record("doc_c").error == "the cost cap was reached"
    saved = data.run_json()
    assert saved["stopped"] == "cost_cap" and saved["totals"]["not_run"] == 2
    assert Decimal(saved["budget_spent_usd"]) == Decimal("1.5")


async def test_a_repair_refused_by_the_cap_keeps_the_unusable_output(data: Dataset) -> None:
    data.png("doc_a")
    data.png("doc_b")
    data.server.enqueue(chat_reply(uncited()))
    run = await data.run(**capped())
    assert run.statuses == {"doc_a": "failed", "doc_b": "not_run"}
    assert run.stopped == "cost_cap" and exit_code(run) == 3
    record = data.record("doc_a")
    assert record.error is not None and "run cost cap exceeded" in record.error
    assert "refused a repair of" in record.error and "extracted without evidence" in record.error
    assert record.raw_text == json.dumps(uncited())
    assert record.cost_usd == Decimal("1.5")
    assert len(data.server.requests) == 1


async def test_cached_answers_are_free_and_reported_apart(data: Dataset, tmp_path: Path) -> None:
    data.png("doc_a")
    cache = {"llm_cache_dir": tmp_path / "cache", **capped(max_run_cost_usd="10")}
    first = await data.run(run_id="first", cache="on", **cache)
    second = await data.run(run_id="second", cache="on", **cache)
    assert first.totals.cost_usd == Decimal("1.5") and first.totals.cached_calls == 0
    assert second.totals.cost_usd == Decimal(0) and second.totals.cached_calls == 1
    assert second.totals.avoided_cost_usd == Decimal("1.5")
    assert data.record("doc_a", "second").cache_hit is True
    assert second.settings["cache"] == "on"
    assert len(data.server.requests) == 1  # the second run never reached the model


# ------------------------------------------------------------------ the blind rule


class TestBlindDocuments:
    def make(self, data: Dataset) -> tuple[ManifestRow, ManifestRow]:
        blind = data.png("doc_x", label_mode="blind", split="test")
        return blind, data.variant(blind)

    async def test_a_blind_document_and_its_variant_are_refused_with_no_request(
        self, data: Dataset
    ) -> None:
        blind, variant = self.make(data)
        assert variant.label_mode is None and variant.gt_from == "doc_x"
        run = await data.run()
        assert run.statuses == {"doc_x": "refused", "doc_x-deg-light": "refused"}
        assert data.server.requests == []
        assert "doc_x is BLIND and doc_x has no finished label" in str(data.record("doc_x").error)
        assert "takes its label from the BLIND document doc_x" in str(
            data.record("doc_x-deg-light").error
        )
        assert data.record("doc_x").pages == []  # refused before its pages were even read

    async def test_once_the_label_is_finished_they_are_extracted(self, data: Dataset) -> None:
        self.make(data)
        (data.labels / "doc_x.json").write_text("{}", encoding="utf-8")
        run = await data.run()
        assert run.statuses == {"doc_x": "ok", "doc_x-deg-light": "ok"}
        assert len(data.server.requests) == 2

    def test_a_draft_is_not_a_finished_label(self, data: Dataset) -> None:
        blind, _ = self.make(data)
        (data.labels / "drafts").mkdir()
        (data.labels / "drafts" / "doc_x.json").write_text("{}", encoding="utf-8")
        assert blind_refusal(blind, data.rows, data.labels) is not None

    def test_other_documents_are_not_affected(self, data: Dataset) -> None:
        for mode in ("prelabel", None):
            row = manifest_row("doc_p", label_mode=mode)
            assert blind_refusal(row, [row], data.labels) is None


# ------------------------------------------------------------------ selection


class TestSelectDocuments:
    rows = [
        manifest_row("doc_1", split="test"),
        manifest_row("doc_2"),
        manifest_row("doc_3", split="test"),
        manifest_row("doc_4"),
    ]

    def test_a_split_in_manifest_order(self) -> None:
        assert [r.doc_id for r in select_documents(self.rows, split="test")] == ["doc_1", "doc_3"]

    def test_named_documents_in_the_order_given(self) -> None:
        chosen = select_documents(self.rows, doc_ids=["doc_4", "doc_1"])
        assert [r.doc_id for r in chosen] == ["doc_4", "doc_1"]

    def test_an_empty_split_is_an_error(self) -> None:
        with pytest.raises(SelectionError, match="^no documents in split 'dev'$"):
            select_documents([manifest_row("doc_1", split="test")], split="dev")

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"doc_ids": ["doc_9", "doc_2", "doc_8"]}, "not in the manifest: doc_9, doc_8"),
            ({"doc_ids": ["doc_2", "doc_2"]}, "named twice"),
            ({}, "either a split or doc ids"),
            ({"split": "dev", "doc_ids": ["doc_2"]}, "either a split or doc ids"),
        ],
    )
    def test_bad_selections_are_errors(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(SelectionError, match=message):
            select_documents(self.rows, **kwargs)


# ------------------------------------------------------------------ the system prompt


class TestSystemPrompt:
    async def test_a_run_without_an_override_records_the_production_prompt(
        self, data: Dataset
    ) -> None:
        data.png("doc_a")
        run = await data.run()
        assert run.prompt_sha256 == PROMPT_SHA256
        assert data.server.requests[0].body["messages"][0]["content"] == SYSTEM_PROMPT

    async def test_an_override_is_sent_and_its_hash_is_recorded(self, data: Dataset) -> None:
        data.png("doc_a")
        data.png("doc_b")
        config = data.settings()
        async with OpenAICompatClient(config) as client:
            run = await run_extraction(
                data.rows,
                rows=data.rows,
                client=client,
                settings=config,
                pages_dir=data.env.pages,
                labels_dir=data.labels,
                run_dir=data.runs / "custom",
                selection={"docs": ["doc_a", "doc_b"]},
                system_prompt="Extract nothing clever.",
            )
        assert [r.body["messages"][0]["content"] for r in data.server.requests] == [
            "Extract nothing clever."
        ] * 2
        assert run.prompt_sha256 == prompt_sha256("Extract nothing clever.") != PROMPT_SHA256
        assert data.run_json("custom")["prompt_sha256"] == run.prompt_sha256
