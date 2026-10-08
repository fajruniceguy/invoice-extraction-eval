"""Evaluate through the HTTP API: the same run folder `extract.py` writes, but every document goes
through POST /extract of a running server, then it is scored.

Usage: python -m uv run python scripts/eval_api.py [--split dev|test | --doc DOC_ID ...]
           [--api-url URL] [--token T] [--run-id ID] [--concurrency N] [--timeout S]
           [--no-score] [--dry-run]
           [--manifest FILE] [--pages-dir DIR] [--ground-truth-dir DIR] [--runs-dir DIR]

This is what `docker compose run --rm eval` runs (no selection = the whole test split). The client
holds NO model key and never reads .env: the server (the `api` service) calls the model. It sends
each document's stored (redacted)
pages as the `files` of one request, in page order, so the server's answer is a `DocRecord` like
the ones `extract.py` writes. A PNG page is read back by the server byte for byte; a JPEG page is
decoded and encoded again at the same quality (upload goes through ingestion's own steps), so its
pixels differ slightly from the stored page: the pages hashed in each record are the SERVER's.

The blind rule is enforced here (the server knows no labels): a BLIND document, or a variant of
one, whose label is not finished is refused with zero requests, as in `extract.py`.

`run.json` records what the SERVER reported on GET /health (model, endpoint, prompt and schema
hashes, result-affecting settings) and the spend the server's budget moved by during the run.
That budget lives as long as the server process, so a run that starts on a server that has
already spent most of MAX_RUN_COST_USD can be stopped early: `docker compose restart api`.

Exit codes: 0 every document ok, 1 a document failed or was refused, 2 bad arguments, an API
that cannot be reached, a refused token or an empty selection, 3 the server's cost cap stopped
the run (documents not started are "not_run").
"""

import argparse
import asyncio
import mimetypes
import os
import re
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from docextract.extract.concurrent import run_bounded
from docextract.extract.runner import (
    DocRecord,
    RunRecord,
    SelectionError,
    blind_refusal,
    exit_code,
    new_run_id,
    run_totals,
    select_documents,
    write_json_atomic,
)
from docextract.ingest.manifest import ManifestError, ManifestRow, read_manifest
from docextract.ingest.stored import StoredPagesError, stored_pages
from docextract.paths import DatasetPaths, find_repo_root

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_HEALTH_KEYS = ("model", "endpoint", "prompt_sha256", "schema_sha256", "settings", "budget")


class ApiUnavailable(Exception):
    """The API cannot be used at all (unreachable, wrong token, not our server): exit 2."""


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def print_record(record: DocRecord) -> None:
    if record.status == "ok":
        tokens = record.usage or {}
        print(
            f"  ok        {record.doc_id}  {len(record.pages)} page(s), "
            f"{tokens.get('input_tokens', 0)} in / {tokens.get('billable_output_tokens', 0)} out "
            f"tokens, ${record.cost_usd:.4f}, {record.latency_ms / 1000:.1f}s, "
            f"{record.repairs} repair(s){' [cached]' if record.cache_hit else ''}"
        )
    else:
        print(f"  {record.status.upper():<9} {record.doc_id}: {record.error}")


def _auth(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


async def server_facts(http: httpx.AsyncClient) -> dict[str, Any]:
    """GET /health, checked to be a docextract server that reports what a run record needs."""
    try:
        response = await http.get("/health")
    except httpx.HTTPError as exc:
        raise ApiUnavailable(f"the API cannot be reached: {exc!r}") from exc
    try:
        facts = response.json()
    except ValueError:
        facts = None
    if response.status_code != 200 or not isinstance(facts, dict):
        raise ApiUnavailable(
            f"GET /health answered HTTP {response.status_code}, not a docextract API"
        )
    missing = [key for key in _HEALTH_KEYS if key not in facts]
    if missing:
        raise ApiUnavailable(f"GET /health lacks {', '.join(missing)}: is it an older server?")
    return facts


def _spent(facts: dict[str, Any]) -> Decimal:
    try:
        return Decimal(str(facts["budget"]["spent_usd"]))
    except (KeyError, TypeError, InvalidOperation):
        raise ApiUnavailable("GET /health: budget.spent_usd is not a number") from None


def _failed(doc_id: str, error: str) -> DocRecord:
    return DocRecord(doc_id=doc_id, status="failed", error=error)


async def extract_via_api(
    http: httpx.AsyncClient, doc_id: str, pages: Sequence[Path]
) -> tuple[DocRecord, bool]:
    """One document through POST /extract: its record and whether the server's cost cap is hit."""
    files = [
        ("files", (page.name, page.read_bytes(), mimetypes.guess_type(page.name)[0] or "image/png"))
        for page in pages
    ]
    try:
        response = await http.post("/extract", files=files, data={"doc_id": doc_id})
    except httpx.HTTPError as exc:
        return _failed(doc_id, f"the API request failed: {exc!r}"), False
    if response.status_code == 401:
        raise ApiUnavailable("the API refused the token (HTTP 401): pass --token or API_TOKEN")
    try:
        body = response.json()
    except ValueError:
        return _failed(doc_id, f"the API answered HTTP {response.status_code}, not JSON"), False
    if response.status_code == 200:
        try:
            return DocRecord.model_validate(body), False
        except ValidationError as exc:
            return _failed(doc_id, f"the API's answer is not a document record: {exc}"), False
    error = body.get("error", {}) if isinstance(body, dict) else {}
    code = error.get("code")
    message = f"HTTP {response.status_code} {code}: {error.get('message', 'no message')}"
    try:  # the server sends the (compact) record, error included, when it can attribute one
        record = DocRecord.model_validate(body["record"])
    except (KeyError, TypeError, ValidationError):
        record = _failed(doc_id, message)
    return record, code == "budget_exceeded"


async def evaluate(
    selected: Sequence[ManifestRow],
    rows: Sequence[ManifestRow],
    *,
    http: httpx.AsyncClient,
    pages_dir: Path,
    labels_dir: Path,
    run_dir: Path,
    selection: dict[str, Any],
    concurrency: int = 1,
    on_record: Callable[[DocRecord], None] | None = None,
) -> RunRecord:
    """Send `selected` to the API, `concurrency` at a time, and record the run in `run_dir`."""
    facts = await server_facts(http)
    spent_before = _spent(facts)
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
        reason = blind_refusal(row, rows, labels_dir)
        if reason is not None:
            return finish(DocRecord(doc_id=row.doc_id, status="refused", error=reason))
        try:
            pages = stored_pages(row.doc_id, pages_dir, derived=row.is_derived)
        except StoredPagesError as exc:
            return finish(DocRecord(doc_id=row.doc_id, status="refused", error=str(exc)))
        record, hit_cap = await extract_via_api(http, row.doc_id, pages)
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
    spent_after = _spent(await server_facts(http))
    run = RunRecord(
        run_id=run_dir.name,
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        model=facts["model"],
        endpoint=facts["endpoint"],
        prompt_sha256=facts["prompt_sha256"],
        schema_sha256=facts["schema_sha256"],
        settings=facts["settings"],
        selection=selection,
        statuses={record.doc_id: record.status for record in records},
        totals=run_totals(records),
        budget_spent_usd=spent_after - spent_before,
        stopped="cost_cap" if stopped else None,
        concurrency=concurrency,
    )
    write_json_atomic(run_dir / "run.json", run.model_dump_json(indent=2) + "\n")
    return run


def dry_run(
    selected: Sequence[ManifestRow], rows: Sequence[ManifestRow], pages_dir: Path, labels_dir: Path
) -> int:
    for row in selected:
        reason = blind_refusal(row, rows, labels_dir)
        if reason is None:
            try:
                pages = stored_pages(row.doc_id, pages_dir, derived=row.is_derived)
            except StoredPagesError as exc:
                reason = str(exc)
        if reason is None:
            print(f"  would send   {row.doc_id} ({len(pages)} page(s))")
        else:
            print(f"  would refuse {row.doc_id}: {reason}")
    print(f"dry run: {len(selected)} document(s), nothing was sent")
    return 0


def score(run_dir: Path, manifest: Path, pages_dir: Path, labels_dir: Path) -> int:
    """Score the finished run with scripts/score.py (exit 0 scored, 1 not scorable)."""
    import score as score_script  # next to this file: the script's own directory is on sys.path

    return int(
        score_script.main(
            [
                str(run_dir),
                "--manifest",
                str(manifest),
                "--pages-dir",
                str(pages_dir),
                "--gt-dir",
                str(labels_dir),
            ]
        )
    )


async def run(
    args: argparse.Namespace, selected: list[ManifestRow], rows: list[ManifestRow]
) -> int:
    run_dir: Path = args.runs_dir / args.run_id
    selection = {"split": args.split} if args.split else {"docs": list(args.doc)}
    async with httpx.AsyncClient(
        base_url=args.api_url.rstrip("/"),
        headers=_auth(args.token),
        timeout=httpx.Timeout(args.timeout, connect=10.0),
    ) as http:
        try:
            record = await evaluate(
                selected,
                rows,
                http=http,
                pages_dir=args.pages_dir,
                labels_dir=args.ground_truth_dir,
                run_dir=run_dir,
                selection=selection,
                concurrency=args.concurrency,
                on_record=print_record,
            )
        except ApiUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 2
    totals = record.totals
    print(
        f"done: {totals.ok} ok, {totals.failed} failed, {totals.refused} refused, "
        f"{totals.not_run} not run; {totals.repairs} repair(s); {totals.input_tokens} in / "
        f"{totals.output_tokens} out tokens; ${totals.cost_usd:.4f} paid "
        f"(+${totals.avoided_cost_usd:.4f} answered from the cache)"
    )
    if record.stopped == "cost_cap":
        print(
            "STOPPED: the server's cost cap (MAX_RUN_COST_USD) was exceeded: "
            f"${record.budget_spent_usd:.4f} spent during this run"
        )
    print(f"run: {run_dir}")
    code = exit_code(record)
    if not args.no_score:
        scored = score(run_dir, args.manifest, args.pages_dir, args.ground_truth_dir)
        code = code or scored
    return code


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    paths = DatasetPaths.default()
    root = find_repo_root()
    parser = argparse.ArgumentParser(description="Evaluate documents through the HTTP API.")
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--split", choices=["dev", "test"], help="every document of a split (default: test)"
    )
    which.add_argument("--doc", action="append", default=[], help="a doc_id (repeatable)")
    parser.add_argument(
        "--api-url", default=os.environ.get("API_URL", "http://127.0.0.1:8000"), help="the server"
    )
    parser.add_argument("--token", default=os.environ.get("API_TOKEN"), help="bearer token")
    parser.add_argument("--run-id", help="name of the run folder (default: the UTC time)")
    parser.add_argument(
        "--concurrency", type=_positive_int, default=2, help="documents sent at a time (default 2)"
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="seconds to wait for one document"
    )
    parser.add_argument("--no-score", action="store_true", help="write the run, do not score it")
    parser.add_argument("--dry-run", action="store_true", help="plan only, send nothing")
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--pages-dir", type=Path, default=paths.pages_dir)
    parser.add_argument("--ground-truth-dir", type=Path, default=paths.ground_truth_dir)
    parser.add_argument("--runs-dir", type=Path, default=root / "outputs" / "runs")
    args = parser.parse_args(argv)
    if args.split is None and not args.doc:
        args.split = "test"  # `docker compose run --rm eval` with no arguments

    try:
        rows = read_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        print(f"The manifest cannot be read: {exc}", file=sys.stderr)
        return 2
    try:
        selected = select_documents(rows, split=args.split, doc_ids=args.doc)
    except SelectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.dry_run:
        return dry_run(selected, rows, args.pages_dir, args.ground_truth_dir)

    args.run_id = args.run_id or new_run_id()
    if not _RUN_ID.fullmatch(args.run_id):
        print(f"--run-id {args.run_id!r}: letters, digits, '_', '-' or '.' only", file=sys.stderr)
        return 2
    if (args.runs_dir / args.run_id).exists():
        print(
            f"{args.runs_dir / args.run_id} already exists: choose another --run-id",
            file=sys.stderr,
        )
        return 2
    print(
        f"eval: {len(selected)} document(s) -> {args.api_url} -> {args.runs_dir / args.run_id}, "
        f"concurrency {args.concurrency}"
    )
    return asyncio.run(run(args, selected, rows))


if __name__ == "__main__":
    sys.exit(main())
