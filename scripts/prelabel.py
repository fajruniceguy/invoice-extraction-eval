"""Draft ground truth with a model, for a human to verify: ground_truth/drafts/<doc_id>.json.

Usage: python -m uv run python scripts/prelabel.py (--doc DOC_ID ... | --all)
           [--dry-run] [--force] [--concurrency N] [--allow-same-model] [--env-file PATH]
           [--manifest FILE] [--pages-dir DIR] [--drafts-dir DIR] [--ground-truth-dir DIR]

Calls the model in PRELABEL_BASE_URL / PRELABEL_MODEL / PRELABEL_API_KEY (see .env.example),
through the same client as the extractor, and costs money. A draft is a ground-truth file with
label_mode=prelabel and no labeled_at: it becomes a label only when a human has verified every
field in the labeling app.

What it refuses, before any call is made:
  * BLIND documents (label_mode=blind): a model draft must never exist for them. Naming one with
    --doc is an error; --all simply leaves them out;
  * documents that are not label_mode=prelabel, that borrow labels (gt_from), or whose labels
    are generated (constructed);
  * a pre-labeling model that is the extractor's (same endpoint and model), unless
    --allow-same-model, which you should not need;
  * documents that are not ingested, or whose stored pages changed since ingestion.
Only the stored, REDACTED pages are sent. Existing drafts and finished labels are kept unless
--force (a finished label is never drafted over).

--dry-run prints what would be drafted and why anything is skipped, and calls nothing.
Exit codes: 0 done, 1 a document was refused or failed, 2 bad arguments or configuration,
3 the run cost cap (PRELABEL_MAX_RUN_COST_USD) was exceeded.
"""

import argparse
import asyncio
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from docextract.config import Settings
from docextract.ingest.manifest import ManifestError, ManifestRow, read_manifest
from docextract.paths import DatasetPaths, find_repo_root
from docextract.providers.budget import BudgetExceededError
from docextract.providers.openai_compat import OpenAICompatClient, ProviderError
from evalkit.gt_store import label_path
from evalkit.prelabel import (
    PrelabelRefused,
    check_eligible,
    eligibility,
    prelabel_document,
    save_draft,
)


def plan(
    rows: list[ManifestRow], wanted: list[str], drafts: Path, labels: Path, force: bool
) -> tuple[list[ManifestRow], list[tuple[str, str]]]:
    """(rows to draft, [(doc_id, why not)]) for the selection."""
    chosen = [row for row in rows if (row.doc_id in wanted if wanted else True)]
    todo: list[ManifestRow] = []
    skipped: list[tuple[str, str]] = []
    for row in chosen:
        reason = eligibility(row)
        if reason is None and label_path(labels, row.doc_id).exists():
            reason = "already has a finished label"
        elif reason is None and not force and label_path(drafts, row.doc_id).exists():
            reason = "already has a draft (use --force to redo it)"
        if reason is None:
            todo.append(row)
        else:
            skipped.append((row.doc_id, reason))
    return todo, skipped


def category(reason: str) -> str:
    for needle, name in (
        ("BLIND", "blind"),
        ("no label_mode", "no label_mode"),
        ("takes its labels", "borrows labels"),
        ("constructed", "constructed"),
        ("finished label", "already labeled"),
        ("already has a draft", "already drafted"),
    ):
        if needle in reason:
            return name
    return "other"


async def run(
    todo: list[ManifestRow],
    settings: Settings,
    pages_dir: Path,
    drafts_dir: Path,
    concurrency: int,
    allow_same_model: bool,
) -> int:
    ps = settings.prelabel_settings()
    extractor = None if allow_same_model else settings
    failures = 0
    totals = {"cost": Decimal(0), "input": 0, "output": 0, "drafted": 0}
    stop = asyncio.Event()
    semaphore = asyncio.Semaphore(concurrency)
    lines: dict[str, str] = {}

    async with OpenAICompatClient(ps) as client:

        async def one(row: ManifestRow) -> None:
            nonlocal failures
            async with semaphore:
                if stop.is_set():
                    lines[row.doc_id] = "  not run   (the cost cap was reached)"
                    return
                try:
                    result = await prelabel_document(
                        row, client=client, settings=ps, pages_dir=pages_dir, extractor=extractor
                    )
                    save_draft(row, result.label, drafts_dir)
                except BudgetExceededError as exc:
                    stop.set()
                    lines[row.doc_id] = f"  STOPPED   {row.doc_id}: {exc}"
                    return
                except (PrelabelRefused, ProviderError) as exc:
                    failures += 1
                    lines[row.doc_id] = f"  FAILED    {exc}"
                    return
            response = result.response
            totals["cost"] += response.cost_usd
            totals["input"] += response.usage.input_tokens
            totals["output"] += response.usage.output_tokens
            totals["drafted"] += 1
            unlabeled = result.unlabeled_fields
            lines[row.doc_id] = (
                f"  drafted   {row.doc_id}  {result.pages} page(s), "
                f"{response.usage.input_tokens} in / {response.usage.output_tokens} out tokens, "
                f"${response.cost_usd:.4f}, {response.latency_ms / 1000:.1f}s, "
                f"{response.repairs} repair(s), {len(unlabeled)} field(s) left for the human"
                f"{' [cached]' if response.cache_hit else ''}"
            )

        await asyncio.gather(*(one(row) for row in todo))

    for row in todo:
        print(lines[row.doc_id])
    print(
        f"done: {totals['drafted']} drafted, {failures} failed, "
        f"{totals['input']} in / {totals['output']} out tokens, ${totals['cost']:.4f} "
        f"({ps.llm_model})"
    )
    if stop.is_set():
        return 3
    return 1 if failures else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(description="Draft ground truth with a model.")
    parser.add_argument("--doc", action="append", default=[], help="a doc_id (repeatable)")
    parser.add_argument("--all", action="store_true", help="every eligible document")
    parser.add_argument("--dry-run", action="store_true", help="plan only, call nothing")
    parser.add_argument("--force", action="store_true", help="redo existing drafts")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--allow-same-model", action="store_true")
    parser.add_argument("--env-file", type=Path, default=find_repo_root() / ".env")
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--pages-dir", type=Path, default=paths.pages_dir)
    parser.add_argument("--drafts-dir", type=Path, default=paths.drafts_dir)
    parser.add_argument("--ground-truth-dir", type=Path, default=paths.ground_truth_dir)
    args = parser.parse_args()

    if bool(args.doc) == args.all:
        print("Choose the documents: --doc DOC_ID (repeatable) or --all", file=sys.stderr)
        return 2
    if args.concurrency < 1:
        print("--concurrency must be at least 1", file=sys.stderr)
        return 2
    try:
        rows = read_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        print(f"The manifest cannot be read: {exc}", file=sys.stderr)
        return 2
    known = {row.doc_id for row in rows}
    unknown = [doc for doc in args.doc if doc not in known]
    if unknown:
        print(f"Not in the manifest: {', '.join(unknown)}", file=sys.stderr)
        return 2

    if args.doc:  # naming a document that cannot be drafted is an error, not a quiet skip
        by_id = {row.doc_id: row for row in rows}
        refused = []
        for doc in args.doc:
            try:
                check_eligible(by_id[doc])
            except PrelabelRefused as exc:
                refused.append(str(exc))
        if refused:
            for message in refused:
                print(f"REFUSED: {message}", file=sys.stderr)
            return 1

    todo, skipped = plan(rows, args.doc, args.drafts_dir, args.ground_truth_dir, args.force)
    if args.all and skipped:
        counts = Counter(category(reason) for _, reason in skipped)
        print("skipped: " + ", ".join(f"{n} {name}" for name, n in sorted(counts.items())))
    elif skipped:
        for doc, reason in skipped:
            print(f"skipped {doc}: {reason}")
    if args.dry_run:
        print(f"dry run: would draft {len(todo)} document(s): {', '.join(r.doc_id for r in todo)}")
        return 0
    if not todo:
        print("nothing to draft")
        return 0

    try:
        settings = Settings(_env_file=args.env_file)
        settings.prelabel_settings()  # validates that PRELABEL_BASE_URL / PRELABEL_MODEL are set
    except (ValidationError, ValueError) as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2
    if settings.same_model_as_prelabel() and not args.allow_same_model:
        print(
            f"REFUSED: the pre-labeling model ({settings.prelabel_model}) is the extractor's "
            f"({settings.llm_model}) on the same endpoint. Set PRELABEL_* to another model.",
            file=sys.stderr,
        )
        return 2
    print(
        f"prelabel: {settings.prelabel_model} at {settings.prelabel_base_url}, {len(todo)} doc(s)"
    )
    return asyncio.run(
        run(
            todo,
            settings,
            args.pages_dir,
            args.drafts_dir,
            args.concurrency,
            args.allow_same_model,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
