"""Extract documents with the configured model into outputs/runs/<run_id>/.

Usage: python -m uv run python scripts/extract.py (--split dev|test | --doc DOC_ID ...)
           [--run-id ID] [--cache on|off|refresh] [--system-prompt-file PATH] [--dry-run]
           [--env-file PATH]
           [--manifest FILE] [--pages-dir DIR] [--ground-truth-dir DIR] [--runs-dir DIR]

Calls LLM_BASE_URL / LLM_MODEL (see .env.example), so it costs money unless the answers come from
the response cache (on by default; --cache refresh asks the model again and overwrites it). Every
page of a document goes in ONE call, and only the stored (redacted) pages are sent, checked against
their ingest.json / degrade.json. Documents run one at a time. Per document it writes
docs/<doc_id>.json (raw output, parsed, validated, repairs, usage, cost) as soon as that document
is done, and run.json at the end (see docextract.extract.runner).

Refused before any call (status "refused", the run goes on):
  * a BLIND document, or a variant of one, whose label is not finished yet;
  * a document that is not ingested, or whose stored pages changed since;
  * a document with more pages than LLM_MAX_PAGES.
A provider error, or output still unusable after the repairs (LLM_MAX_REPAIRS), is "failed" and the
run goes on. The run cost cap (MAX_RUN_COST_USD) stops the run: the rest is "not_run".

--dry-run lists what would be extracted or refused (blind rule and stored pages), and calls nothing.
Exit codes: 0 every document ok, 1 a document failed or was refused, 2 bad arguments,
configuration or an empty selection, 3 the run cost cap was exceeded.
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from docextract.config import Settings
from docextract.extract.prompt import SYSTEM_PROMPT, prompt_sha256
from docextract.extract.runner import (
    DocRecord,
    SelectionError,
    blind_refusal,
    exit_code,
    new_run_id,
    run_extraction,
    select_documents,
)
from docextract.ingest.manifest import ManifestError, ManifestRow, read_manifest
from docextract.ingest.stored import StoredPagesError, stored_pages
from docextract.paths import DatasetPaths, find_repo_root
from docextract.providers.openai_compat import OpenAICompatClient

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


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


def dry_run(
    selected: list[ManifestRow], rows: list[ManifestRow], pages_dir: Path, labels_dir: Path
) -> int:
    for row in selected:
        reason = blind_refusal(row, rows, labels_dir)
        if reason is None:
            try:
                pages = stored_pages(row.doc_id, pages_dir, derived=row.is_derived)
            except StoredPagesError as exc:
                reason = str(exc)
        if reason is None:
            print(f"  would extract {row.doc_id} ({len(pages)} page(s))")
        else:
            print(f"  would refuse  {row.doc_id}: {reason}")
    print(f"dry run: {len(selected)} document(s), nothing was called")
    return 0


async def run(
    selected: list[ManifestRow],
    rows: list[ManifestRow],
    settings: Settings,
    args: argparse.Namespace,
    run_dir: Path,
    system_prompt: str,
) -> int:
    selection = {"split": args.split} if args.split else {"docs": list(args.doc)}
    async with OpenAICompatClient(settings) as client:
        record = await run_extraction(
            selected,
            rows=rows,
            client=client,
            settings=settings,
            pages_dir=args.pages_dir,
            labels_dir=args.ground_truth_dir,
            run_dir=run_dir,
            selection=selection,
            cache=args.cache,
            on_record=print_record,
            system_prompt=system_prompt,
        )
    totals = record.totals
    print(
        f"done: {totals.ok} ok, {totals.failed} failed, {totals.refused} refused, "
        f"{totals.not_run} not run; {totals.repairs} repair(s); {totals.input_tokens} in / "
        f"{totals.output_tokens} out tokens; ${totals.cost_usd:.4f} paid "
        f"(+${totals.avoided_cost_usd:.4f} answered from the cache)"
    )
    if record.stopped == "cost_cap":
        print(
            f"STOPPED: the run cost cap (MAX_RUN_COST_USD) was exceeded: "
            f"${record.budget_spent_usd:.4f} spent"
        )
    print(f"run: {run_dir}")
    return exit_code(record)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    paths = DatasetPaths.default()
    root = find_repo_root()
    parser = argparse.ArgumentParser(description="Extract documents with the configured model.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--split", choices=["dev", "test"], help="every document of a split")
    which.add_argument("--doc", action="append", default=[], help="a doc_id (repeatable)")
    parser.add_argument("--run-id", help="name of the run folder (default: the UTC time)")
    parser.add_argument(
        "--cache",
        choices=["on", "off", "refresh"],
        help="override LLM_CACHE for this run (refresh: ask again and overwrite)",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="send this text as the system prompt instead of the production one (the run "
        "records its hash)",
    )
    parser.add_argument("--dry-run", action="store_true", help="plan only, call nothing")
    parser.add_argument("--env-file", type=Path, default=root / ".env")
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--pages-dir", type=Path, default=paths.pages_dir)
    parser.add_argument("--ground-truth-dir", type=Path, default=paths.ground_truth_dir)
    parser.add_argument("--runs-dir", type=Path, default=root / "outputs" / "runs")
    args = parser.parse_args()

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
    system_prompt = SYSTEM_PROMPT
    if args.system_prompt_file is not None:
        try:
            system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"--system-prompt-file {args.system_prompt_file}: {exc}", file=sys.stderr)
            return 2
        if not system_prompt.strip():
            print(
                f"--system-prompt-file {args.system_prompt_file}: the file is blank",
                file=sys.stderr,
            )
            return 2
    if args.dry_run:
        return dry_run(selected, rows, args.pages_dir, args.ground_truth_dir)

    run_id = args.run_id or new_run_id()
    if not _RUN_ID.fullmatch(run_id):
        print(f"--run-id {run_id!r}: letters, digits, '_', '-' or '.' only", file=sys.stderr)
        return 2
    run_dir = args.runs_dir / run_id
    if run_dir.exists():
        print(f"{run_dir} already exists: choose another --run-id", file=sys.stderr)
        return 2
    try:
        settings = Settings(_env_file=args.env_file)
    except ValidationError as exc:
        print(f"Configuration problem (see .env.example): {exc}", file=sys.stderr)
        return 2
    source = f"from {args.system_prompt_file}" if args.system_prompt_file else "production"
    print(
        f"extract: {settings.llm_model} at {settings.llm_base_url}, "
        f"{len(selected)} document(s) -> {run_dir}\n"
        f"  prompt: {prompt_sha256(system_prompt)[:12]} ({source})"
    )
    return asyncio.run(run(selected, rows, settings, args, run_dir, system_prompt))


if __name__ == "__main__":
    sys.exit(main())
