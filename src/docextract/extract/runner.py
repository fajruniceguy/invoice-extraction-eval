"""Extract a set of documents and record every outcome under outputs/runs/<run_id>/.

  docs/<doc_id>.json   one record per document, written the moment that document is done
  run.json             the run: model, endpoint, prompt and schema hashes, settings, selection,
                       each document's status, totals (real and cached cost apart), why it stopped

One document at a time. A document that fails (a provider error, output still unusable after the
repairs, stored pages missing or changed, too many pages) is recorded and the run goes on. The
cost cap is different: `BudgetExceededError` stops the run, what is done is kept, and the rest is
recorded as `not_run` (it is deliberately not a ProviderError, so it is never swallowed here).

Blind documents: a document that is blind, or that borrows the label of a blind document
(`gt_from`), is refused with no request while that label is not finished. A human labels it from
the page alone, so no model output for it may exist before then. This module must not import
evalkit, so "finished" is checked as evalkit.gt_store does: ground_truth/<doc_id>.json exists.

Records hold the model's raw text, its parsed output and the validated copy, so what the model
said and what the validators concluded can always be compared. They never hold the API key.
"""

import os
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from docextract.config import CacheMode, Settings
from docextract.extract.concurrent import run_bounded
from docextract.extract.extractor import ExtractionRefused, extract_pages
from docextract.extract.prompt import SYSTEM_PROMPT, prompt_sha256
from docextract.extract.schema import DocumentExtraction
from docextract.ingest.manifest import ManifestRow
from docextract.ingest.pipeline import sha256_hex
from docextract.ingest.stored import StoredPagesError, stored_pages
from docextract.providers.budget import BudgetExceededError
from docextract.providers.cache import schema_hash
from docextract.providers.openai_compat import (
    OpenAICompatClient,
    ProviderError,
    RepairRecord,
    ResponseParseError,
)

DocStatus = Literal["ok", "failed", "refused", "not_run"]


class SelectionError(Exception):
    """The documents asked for cannot be selected (and why)."""


class PageRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    sha256: str


class Repair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: str
    raw_text: str


class DocRecord(BaseModel):
    """What happened to one document. `parsed` is what the model said (after normalization),
    `validated` what the validators made of it; both are None unless status is ok."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    status: DocStatus
    error: str | None = None
    pages: list[PageRef] = []
    raw_text: str | None = None  # the last model output seen
    parsed: DocumentExtraction | None = None
    validated: DocumentExtraction | None = None
    repairs: int = 0
    repair_history: list[Repair] = []
    normalizations: list[str] = []
    usage: dict[str, Any] | None = None
    cost_usd: Decimal = Decimal(0)  # what was actually paid
    avoided_cost_usd: Decimal = Decimal(0)  # what the cached calls would have cost
    cached_calls: int = 0
    cache_hit: bool = False
    latency_ms: float = 0.0
    attempts: int = 0
    structured_mode_used: str | None = None
    finish_reason: str | None = None


class RunTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: int = 0
    ok: int = 0
    failed: int = 0
    refused: int = 0
    not_run: int = 0
    repairs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0  # billable: reasoning the provider left out of completion included
    cost_usd: Decimal = Decimal(0)
    avoided_cost_usd: Decimal = Decimal(0)
    cached_calls: int = 0


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    started_at: str
    finished_at: str
    model: str
    endpoint: str
    prompt_sha256: str
    schema_sha256: str
    settings: dict[str, Any]
    selection: dict[str, Any]
    statuses: dict[str, DocStatus]
    totals: RunTotals
    budget_spent_usd: Decimal  # the client's own tally: includes calls of documents cut short
    stopped: Literal["cost_cap"] | None = None
    concurrency: int = 1  # documents in flight at once (not a result-changing setting)


# ---------------------------------------------------------------- what to run


def select_documents(
    rows: Sequence[ManifestRow], *, split: str | None = None, doc_ids: Sequence[str] = ()
) -> list[ManifestRow]:
    """The rows of `split` in manifest order, or the named documents in the order given."""
    if (split is None) == (not doc_ids):
        raise SelectionError("choose either a split or doc ids")
    if split is not None:
        chosen = [row for row in rows if row.split == split]
        if not chosen:
            raise SelectionError(f"no documents in split {split!r}")
        return chosen
    by_id = {row.doc_id: row for row in rows}
    unknown = [doc_id for doc_id in doc_ids if doc_id not in by_id]
    if unknown:
        raise SelectionError(f"not in the manifest: {', '.join(unknown)}")
    if len(set(doc_ids)) != len(doc_ids):
        raise SelectionError("a doc id is named twice")
    return [by_id[doc_id] for doc_id in doc_ids]


def blind_refusal(row: ManifestRow, rows: Sequence[ManifestRow], labels_dir: Path) -> str | None:
    """Why `row` may not be extracted yet because of the blind rule, or None."""
    owner = next((r for r in rows if r.doc_id == row.gt_from), row) if row.gt_from else row
    if "blind" not in (row.label_mode, owner.label_mode):
        return None
    if (labels_dir / f"{owner.doc_id}.json").is_file():
        return None
    whose = "" if owner is row else f" (it takes its label from the BLIND document {owner.doc_id})"
    return (
        f"{row.doc_id} is BLIND{whose} and {owner.doc_id} has no finished label yet: a human "
        "labels it from the page alone first, so no model output may exist for it before then"
    )


# ---------------------------------------------------------------- running


def write_json_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _history(history: Sequence[RepairRecord]) -> list[Repair]:
    return [Repair(problem=record.problem, raw_text=record.raw_text) for record in history]


def _page_refs(pages: Sequence[Path]) -> list[PageRef]:
    return [PageRef(file=page.name, sha256=sha256_hex(page.read_bytes())) for page in pages]


def _failure(doc_id: str, exc: ProviderError, pages: list[PageRef]) -> DocRecord:
    record = DocRecord(
        doc_id=doc_id,
        status="failed",
        error=str(exc),
        pages=pages,
        usage=exc.usage.model_dump(mode="json") if exc.usage is not None else None,
        cost_usd=exc.cost_usd,
        cached_calls=exc.cached_calls,
        latency_ms=exc.latency_ms,
        attempts=exc.attempts,
    )
    if isinstance(exc, ResponseParseError):
        record.raw_text = exc.raw_text
        record.repairs = exc.repairs
        record.repair_history = _history(exc.history)
    return record


async def extract_record(
    doc_id: str,
    pages: Sequence[Path],
    *,
    client: OpenAICompatClient,
    settings: Settings,
    cache: CacheMode | None,
    system_prompt: str,
) -> tuple[DocRecord, bool]:
    """One document's record from its page files, and whether the cost cap stopped the run on it.
    The runner (stored pages) and the API (uploaded pages) both come through here, so a record
    never depends on which of them produced it."""
    refs = _page_refs(pages)
    try:
        result = await extract_pages(
            pages, client=client, settings=settings, cache=cache, system_prompt=system_prompt
        )
    except ExtractionRefused as exc:
        return DocRecord(doc_id=doc_id, status="refused", error=str(exc), pages=refs), False
    except BudgetExceededError as exc:
        cause = exc.__cause__
        if isinstance(cause, ResponseParseError):  # the cap refused a repair of this document
            record = _failure(doc_id, cause, refs)
            record.error = f"{exc} (refused a repair of: {cause})"
            return record, True
        return DocRecord(doc_id=doc_id, status="not_run", error=str(exc), pages=refs), True
    except ProviderError as exc:
        return _failure(doc_id, exc, refs), False
    response = result.response
    record = DocRecord(
        doc_id=doc_id,
        status="ok",
        pages=refs,
        raw_text=response.text,
        parsed=result.parsed,
        validated=result.validated,
        repairs=response.repairs,
        repair_history=_history(response.repair_history),
        normalizations=list(response.normalizations),
        usage=response.usage.model_dump(mode="json"),
        cost_usd=response.cost_usd,
        avoided_cost_usd=response.avoided_cost_usd,
        cached_calls=response.cached_calls,
        cache_hit=response.cache_hit,
        latency_ms=response.latency_ms,
        attempts=response.attempts,
        structured_mode_used=response.structured_mode_used,
        finish_reason=response.finish_reason,
    )
    return record, False


async def _extract_one(
    row: ManifestRow,
    rows: Sequence[ManifestRow],
    *,
    client: OpenAICompatClient,
    settings: Settings,
    pages_dir: Path,
    labels_dir: Path,
    cache: CacheMode | None,
    system_prompt: str,
) -> tuple[DocRecord, bool]:
    """One document's record, and whether the cost cap stopped the run on it."""
    reason = blind_refusal(row, rows, labels_dir)
    if reason is not None:
        return DocRecord(doc_id=row.doc_id, status="refused", error=reason), False
    try:
        pages = stored_pages(row.doc_id, pages_dir, derived=row.is_derived)
    except StoredPagesError as exc:
        return DocRecord(doc_id=row.doc_id, status="refused", error=str(exc)), False
    return await extract_record(
        row.doc_id,
        pages,
        client=client,
        settings=settings,
        cache=cache,
        system_prompt=system_prompt,
    )


def _totals(records: Sequence[DocRecord]) -> RunTotals:
    totals = RunTotals(documents=len(records))
    for record in records:
        setattr(totals, record.status, getattr(totals, record.status) + 1)
        totals.repairs += record.repairs
        totals.cost_usd += record.cost_usd
        totals.avoided_cost_usd += record.avoided_cost_usd
        totals.cached_calls += record.cached_calls
        if record.usage is not None:
            totals.input_tokens += int(record.usage.get("input_tokens", 0))
            totals.output_tokens += int(record.usage.get("billable_output_tokens", 0))
    return totals


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


async def run_extraction(
    selected: Sequence[ManifestRow],
    *,
    rows: Sequence[ManifestRow],
    client: OpenAICompatClient,
    settings: Settings,
    pages_dir: Path,
    labels_dir: Path,
    run_dir: Path,
    selection: dict[str, Any],
    cache: CacheMode | None = None,
    on_record: Callable[[DocRecord], None] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    concurrency: int = 1,
) -> RunRecord:
    """Extract `selected` into `run_dir`, `concurrency` documents at a time (1 = one after the
    other). `rows` is the whole manifest (for the blind rule); `settings` are the client's;
    `selection` is recorded as given; `system_prompt` is the prompt sent, and the run records its
    hash. Once the cost cap stops a document, no document that has not started is called: calls
    already in flight finish, so the spend can pass the cap by up to `concurrency` calls."""
    started = datetime.now(UTC).isoformat(timespec="seconds")
    stopped = False

    def finish(record: DocRecord) -> DocRecord:
        write_json_atomic(
            run_dir / "docs" / f"{record.doc_id}.json", record.model_dump_json(indent=2) + "\n"
        )
        if on_record is not None:
            on_record(record)
        return record

    async def work(row: ManifestRow) -> DocRecord:
        nonlocal stopped
        record, hit_cap = await _extract_one(
            row,
            rows,
            client=client,
            settings=settings,
            pages_dir=pages_dir,
            labels_dir=labels_dir,
            cache=cache,
            system_prompt=system_prompt,
        )
        if hit_cap:
            stopped = True
        return finish(record)

    def not_run(row: ManifestRow) -> DocRecord:
        return finish(
            DocRecord(doc_id=row.doc_id, status="not_run", error="the cost cap was reached")
        )

    records = await run_bounded(
        list(selected), work, concurrency=concurrency, skip=lambda: stopped, skipped=not_run
    )

    run = RunRecord(
        run_id=run_dir.name,
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        model=settings.llm_model,
        endpoint=settings.llm_base_url,
        prompt_sha256=prompt_sha256(system_prompt),
        schema_sha256=schema_hash(DocumentExtraction),
        settings={
            "structured_mode": settings.llm_structured_mode,
            "reasoning_effort": settings.llm_reasoning_effort,
            "image_max_long_edge": settings.llm_image_max_long_edge,
            "max_pages": settings.llm_max_pages,
            "max_repairs": settings.llm_max_repairs,
            "cache": cache or settings.llm_cache,
            "max_run_cost_usd": (
                str(settings.max_run_cost_usd) if settings.max_run_cost_usd is not None else None
            ),
            "temperature": 0.0,
        },
        selection=selection,
        statuses={record.doc_id: record.status for record in records},
        totals=_totals(records),
        budget_spent_usd=client.budget.spent,
        stopped="cost_cap" if stopped else None,
        concurrency=concurrency,
    )
    write_json_atomic(run_dir / "run.json", run.model_dump_json(indent=2) + "\n")
    return run


def exit_code(run: RunRecord) -> int:
    """0 every document ok, 1 some failed or were refused, 3 the cost cap stopped the run."""
    if run.stopped == "cost_cap":
        return 3
    return 1 if run.totals.failed or run.totals.refused else 0
