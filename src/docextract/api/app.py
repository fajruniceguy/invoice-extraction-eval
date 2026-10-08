"""HTTP API: upload documents, get back the same records the extraction runner writes.

  GET  /health          status, model, concurrency, budget (never a secret)
  POST /extract         ONE document from the uploaded `files` (its pages or photos, in order)
  POST /extract/batch   MANY documents: every uploaded file is one document, run concurrently

Both POST routes answer with `DocRecord`s (the run-record type), so an API answer is shaped like
a run folder entry and can be scored. `/extract` maps failures to HTTP codes (400 unreadable file,
401 bad token, 413 too large, 422 too many pages, 502 provider failure, 503 cost cap);
`/extract/batch` always answers 200 and marks each document: a bad file is `refused`, a provider
failure `failed`, and once the cost cap is hit the documents that had not started are `not_run`.

One process-wide semaphore (`API_CONCURRENCY`) bounds documents in flight across ALL requests.
The cost cap (`MAX_RUN_COST_USD`) is a server-lifetime budget, and calls already in flight finish
after it is crossed, so spend can pass it by up to that many calls. Uploads are written only
inside a temp directory that is removed after the request; nothing is ever written under data/.

Run it with:  python -m uv run uvicorn --factory docextract.api.app:create_app
"""

import asyncio
import hmac
import re
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from docextract.api.uploads import UploadError, pages_from_uploads
from docextract.calllog import CallLog
from docextract.config import Settings
from docextract.extract.concurrent import run_bounded
from docextract.extract.prompt import SYSTEM_PROMPT, prompt_sha256
from docextract.extract.runner import DocRecord, extract_record, run_totals
from docextract.extract.schema import DocumentExtraction
from docextract.providers.cache import schema_hash
from docextract.providers.openai_compat import OpenAICompatClient

# error code -> HTTP status for /extract (the batch answers 200 whatever happens to a document)
STATUS_OF_CODE: dict[str, int] = {
    "unreadable_file": 400,
    "file_too_large": 413,
    "too_many_pages": 422,
    "refused": 422,
    "provider_error": 502,
    "budget_exceeded": 503,
}
_DOC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_COMPACT_DROPS = {"raw_text", "parsed", "repair_history"}


class ApiError(Exception):
    """An error answer: HTTP status, a stable code, a message, and the record when there is one."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        record: DocRecord | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.record = record
        self.headers = headers or {}


@dataclass
class ApiState:
    settings: Settings
    client: OpenAICompatClient
    semaphore: asyncio.Semaphore
    calllog: CallLog


@dataclass(frozen=True)
class Outcome:
    record: DocRecord
    code: str | None  # None = ok
    hit_cap: bool = False


def result_settings(cfg: Settings) -> dict[str, Any]:
    """The settings that shape an answer, as `run.json` records them (the runner's own dict), so an
    eval client can write a run record that describes what THIS server used. No secret in it."""
    return {
        "structured_mode": cfg.llm_structured_mode,
        "reasoning_effort": cfg.llm_reasoning_effort,
        "image_max_long_edge": cfg.llm_image_max_long_edge,
        "max_pages": cfg.llm_max_pages,
        "max_repairs": cfg.llm_max_repairs,
        "cache": cfg.llm_cache,
        "max_run_cost_usd": (
            str(cfg.max_run_cost_usd) if cfg.max_run_cost_usd is not None else None
        ),
        "temperature": 0.0,
    }


def _state(request: Request) -> ApiState:
    state: ApiState = request.app.state.api
    return state


async def require_token(request: Request) -> None:
    token = _state(request).settings.api_token
    if token is None:
        return
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    expected = token.get_secret_value().encode()
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.strip().encode(), expected):
        raise ApiError(
            401,
            "unauthorized",
            "send the API token as 'Authorization: Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _dump(record: DocRecord, compact: bool) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude=_COMPACT_DROPS if compact else None)


async def _read(upload: UploadFile, limit: int) -> bytes | None:
    """The upload's bytes, or None when it is over `limit` (reads at most limit + 1 bytes)."""
    try:
        data = await upload.read(limit + 1)
    finally:
        await upload.close()
    return data if len(data) <= limit else None


def _doc_id_from_name(name: str | None, index: int) -> str:
    """A safe id from the client's file name. Only an echo in the answer: never a path."""
    stem = Path(name or "").stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-._")[:60]
    return cleaned or f"doc-{index + 1}"


async def _process(
    state: ApiState,
    doc_id: str,
    files: list[bytes],
    *,
    page_range: str | None,
    request_id: str,
    route: str,
    index: int,
) -> Outcome:
    """One document, start to finish. Never raises for a bad document: it comes back as a record.
    The caller holds the concurrency semaphore."""
    cfg = state.settings
    if state.client.budget.exceeded:
        outcome = Outcome(
            DocRecord(doc_id=doc_id, status="not_run", error="the cost cap was reached"),
            "budget_exceeded",
            True,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="docextract-") as workdir:
            try:
                pages = await asyncio.to_thread(
                    pages_from_uploads,
                    files,
                    Path(workdir),
                    max_pages=cfg.llm_max_pages,
                    page_range=page_range,
                    max_bytes=cfg.api_max_upload_mb * 1024 * 1024,
                )
            except UploadError as exc:
                outcome = Outcome(
                    DocRecord(doc_id=doc_id, status="refused", error=str(exc)), exc.code
                )
            else:  # the pages are read (hashed, sent) inside this block: the folder goes after it
                record, hit_cap = await extract_record(
                    doc_id,
                    pages,
                    client=state.client,
                    settings=cfg,
                    cache=None,
                    system_prompt=SYSTEM_PROMPT,
                )
                code: str | None = None
                if hit_cap:
                    code = "budget_exceeded"
                elif record.status == "failed":
                    code = "provider_error"
                elif record.status == "refused":
                    code = "refused"
                outcome = Outcome(record, code, hit_cap)
    state.calllog.log(outcome.record, request_id=request_id, route=route, index=index)
    return outcome


def create_app(
    settings: Settings | None = None, *, client: OpenAICompatClient | None = None
) -> FastAPI:
    """Build the app. With no arguments it reads the environment (uvicorn --factory); tests pass a
    `Settings` and, if they like, a client (which they then close themselves)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings if settings is not None else Settings()
        llm = client if client is not None else OpenAICompatClient(cfg)
        app.state.api = ApiState(
            settings=cfg,
            client=llm,
            semaphore=asyncio.Semaphore(cfg.api_concurrency),
            calllog=CallLog(cfg.call_log_path, model=cfg.llm_model),
        )
        try:
            yield
        finally:
            if client is None:
                await llm.aclose()

    app = FastAPI(title="docextract", lifespan=lifespan)

    @app.middleware("http")
    async def request_id(request: Request, call_next: Any) -> Any:
        request.state.request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        body: dict[str, Any] = {
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": getattr(request.state, "request_id", None),
            }
        }
        if exc.record is not None:
            body["record"] = _dump(exc.record, compact=True)
        return JSONResponse(body, status_code=exc.status, headers=exc.headers)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        state = _state(request)
        budget = state.client.budget
        return {
            "status": "ok",
            "model": state.settings.llm_model,
            "endpoint": state.settings.llm_base_url,
            "prompt_sha256": prompt_sha256(SYSTEM_PROMPT),
            "schema_sha256": schema_hash(DocumentExtraction),
            "settings": result_settings(state.settings),
            "concurrency": state.settings.api_concurrency,
            "max_batch": state.settings.api_max_batch,
            "budget": {
                "spent_usd": str(budget.spent),
                "limit_usd": None if budget.limit is None else str(budget.limit),
                "exceeded": budget.exceeded,
            },
            "calls": budget.calls,
            "cached_calls": budget.cached_calls,
        }

    @app.post("/extract", dependencies=[Depends(require_token)])
    async def extract(
        request: Request,
        files: Annotated[list[UploadFile] | None, File()] = None,
        page_range: Annotated[str | None, Form()] = None,
        doc_id: Annotated[str | None, Form()] = None,
        compact: Annotated[bool, Query()] = False,
    ) -> JSONResponse:
        state = _state(request)
        limit = state.settings.api_max_upload_mb * 1024 * 1024
        uploads = files or []
        if not uploads:
            raise ApiError(400, "no_files", "send at least one file in the 'files' field")
        if doc_id is not None and not _DOC_ID.fullmatch(doc_id):
            raise ApiError(
                400, "invalid_doc_id", "doc_id: letters, digits, '_', '-' or '.', at most 64"
            )
        contents: list[bytes] = []
        for number, upload in enumerate(uploads, start=1):
            data = await _read(upload, limit)
            if data is None:
                raise ApiError(
                    413,
                    "file_too_large",
                    f"file {number} is larger than {state.settings.api_max_upload_mb} MB",
                )
            contents.append(data)
        rid: str = request.state.request_id
        async with state.semaphore:
            outcome = await _process(
                state,
                doc_id or f"upload-{rid}",
                contents,
                page_range=page_range or None,
                request_id=rid,
                route="/extract",
                index=0,
            )
        if outcome.code is not None:
            raise ApiError(
                STATUS_OF_CODE[outcome.code],
                outcome.code,
                outcome.record.error or outcome.code,
                record=outcome.record,
            )
        return JSONResponse(_dump(outcome.record, compact))

    @app.post("/extract/batch", dependencies=[Depends(require_token)])
    async def extract_batch(
        request: Request,
        files: Annotated[list[UploadFile] | None, File()] = None,
        compact: Annotated[bool, Query()] = False,
    ) -> JSONResponse:
        state = _state(request)
        cfg = state.settings
        uploads = files or []
        if not uploads:
            raise ApiError(400, "no_files", "send at least one file in the 'files' field")
        if len(uploads) > cfg.api_max_batch:
            raise ApiError(
                413,
                "batch_too_large",
                f"{len(uploads)} files, but at most {cfg.api_max_batch} per batch",
            )
        rid: str = request.state.request_id
        limit = cfg.api_max_upload_mb * 1024 * 1024
        items: list[tuple[int, str, bytes | None]] = []
        used: set[str] = set()
        for index, upload in enumerate(uploads):
            doc_id = _doc_id_from_name(upload.filename, index)
            if doc_id in used:
                doc_id = f"{doc_id}-{index + 1}"
            used.add(doc_id)
            items.append((index, doc_id, await _read(upload, limit)))

        def not_started(item: tuple[int, str, bytes | None]) -> Outcome:
            index, doc_id, _ = item
            record = DocRecord(doc_id=doc_id, status="not_run", error="the cost cap was reached")
            state.calllog.log(record, request_id=rid, route="/extract/batch", index=index)
            return Outcome(record, "budget_exceeded", True)

        async def work(item: tuple[int, str, bytes | None]) -> Outcome:
            index, doc_id, data = item
            if data is None:
                record = DocRecord(
                    doc_id=doc_id,
                    status="refused",
                    error=f"larger than {cfg.api_max_upload_mb} MB",
                )
                state.calllog.log(record, request_id=rid, route="/extract/batch", index=index)
                return Outcome(record, "file_too_large")
            return await _process(
                state,
                doc_id,
                [data],
                page_range=None,
                request_id=rid,
                route="/extract/batch",
                index=index,
            )

        started = time.monotonic()
        outcomes = await run_bounded(
            items,
            work,
            semaphore=state.semaphore,
            skip=lambda: state.client.budget.exceeded,
            skipped=not_started,
        )
        records = [outcome.record for outcome in outcomes]
        run = run_totals(records).model_dump(mode="json")
        run["stopped"] = "cost_cap" if any(outcome.hit_cap for outcome in outcomes) else None
        run["wall_ms"] = round((time.monotonic() - started) * 1000, 1)
        run["concurrency"] = cfg.api_concurrency
        return JSONResponse({"run": run, "documents": [_dump(r, compact) for r in records]})

    return app
