"""Send one image to the configured LLM and print the parsed extraction, tokens and latency.

Usage: python -m uv run python scripts/smoke.py [IMAGE] [--raw] [--cache] [--env-file PATH]
       [--output PATH]

IMAGE defaults to the first image in data/raw, then to tests/fixtures/sample_receipt.png.
Configuration (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, ...) comes from the environment, then
from the env file (default: .env at the repo root).

The response cache is BYPASSED here (a smoke check must hit the real API); --cache allows it.
With --raw the model's raw text is printed before the parsed result. Whatever the outcome, a
JSON artifact (raw text, parsed result, validator output, repaired failures, usage) is written
to --output (default outputs/smoke-last.json, which is gitignored). It never contains the key.

Exit codes: 0 success, 1 the call or the model output failed, 2 configuration/input problem,
3 the run cost cap (MAX_RUN_COST_USD) was exceeded.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from docextract.config import Settings
from docextract.extract.schema import DocumentExtraction
from docextract.providers.budget import BudgetExceededError, RunBudget
from docextract.providers.openai_compat import (
    OpenAICompatClient,
    ProviderError,
    RepairRecord,
    ResponseParseError,
    Usage,
    build_image_message,
)
from docextract.validate.arithmetic import validate_line_items

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_receipt.png"
DEFAULT_OUTPUT = ROOT / "outputs" / "smoke-last.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
RAW_EXCERPT_CHARS = 600

SYSTEM_PROMPT = """\
You extract structured data from an invoice or receipt image.

For every field return: value, confidence (0 to 1), status, evidence, page and reasons.
status is exactly one of:
- "extracted": the document states the field. value is non-null, evidence is a verbatim \
quote from the document, and confidence is how sure you are the value is right.
- "absent": the document does not state this field at all (for example a receipt with no \
due date). value MUST be null. confidence is how sure you are that it is absent, and can be \
high. reasons is optional.
- "abstained": the field may be on the document but you cannot read or decide it confidently \
(blurry, ambiguous, conflicting). value MUST be null and reasons MUST say why.
Never guess or infer a value that is not printed.

- Dates as ISO YYYY-MM-DD. Money as plain decimal numbers: no currency symbols, no thousands \
separators. currency is the ISO 4217 code.
- page is the 1-based number of the image the evidence is on.
- line_items: one entry per printed line. Each row has ONE status, confidence, evidence, page \
and reasons for the whole row. An "extracted" row has at least one non-null cell; use null \
for cells the row does not show.
"""

USER_PROMPT = "Extract the fields from this document."


def pick_image(argument: str | None) -> Path:
    if argument:
        return Path(argument)
    raw = ROOT / "data" / "raw"
    candidates = sorted(p for p in raw.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return candidates[0] if candidates else FIXTURE


def describe_config_error(exc: ValidationError) -> str:
    lines = [
        f"  {'.'.join(str(part) for part in error['loc']).upper()}: {error['msg']}"
        for error in exc.errors()
    ]
    return "Configuration problem (see .env.example):\n" + "\n".join(lines)


async def run(image: Path, env_file: Path, output: Path, show_raw: bool, use_cache: bool) -> int:
    if not image.is_file():
        hint = (
            "  generate it with: python -m uv run python scripts/make_sample_receipt.py"
            if image == FIXTURE
            else ""
        )
        print(f"Image not found: {image}\n{hint}".rstrip(), file=sys.stderr)
        return 2
    try:
        settings = Settings(_env_file=env_file)
    except ValidationError as exc:
        print(describe_config_error(exc), file=sys.stderr)
        return 2

    if not use_cache:
        settings = settings.model_copy(update={"llm_cache": "off"})

    try:
        message = build_image_message(USER_PROMPT, [image])
    except ValueError as exc:
        print(f"Cannot use {image}: {exc}", file=sys.stderr)
        return 2

    print(f"model={settings.llm_model} endpoint={settings.llm_base_url} image={image.name}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, message]
    # Only fields that are safe to share ever go in the artifact: never the key.
    artifact: dict[str, Any] = {
        "model": settings.llm_model,
        "endpoint": settings.llm_base_url,
        "image": image.name,
    }
    async with OpenAICompatClient(settings) as client:
        try:
            result = await client.chat(messages, DocumentExtraction, temperature=0.0)
        except ResponseParseError as exc:
            print(f"\nModel output was not usable: {exc}", file=sys.stderr)
            print(f"--- raw output (first 2000 chars) ---\n{exc.raw_text[:2000]}", file=sys.stderr)
            _print_repaired(exc.history, file=sys.stderr)
            _print_stats(
                exc.usage,
                exc.cost_usd,
                exc.latency_ms,
                exc.attempts,
                exc.repairs,
                None,
                None,
                exc.cached_calls,
            )
            _print_budget(client.budget)
            artifact |= {
                "ok": False,
                "error": str(exc),
                "raw_text": exc.raw_text,
                "repair_history": _history_json(exc.history),
                "attempts": exc.attempts,
                "repairs": exc.repairs,
                "latency_ms": round(exc.latency_ms),
                "usage": (exc.usage or Usage()).model_dump(mode="json"),
            }
            _save(output, artifact)
            return 1
        except BudgetExceededError as exc:
            print(f"\nStopped: {exc}", file=sys.stderr)
            cause = exc.__cause__
            if isinstance(cause, ResponseParseError):
                # a refused repair: show the output we were not allowed to fix
                print(f"The last model output was not usable: {cause}", file=sys.stderr)
                print(
                    f"--- raw output (first 2000 chars) ---\n{cause.raw_text[:2000]}",
                    file=sys.stderr,
                )
                _print_repaired(cause.history, file=sys.stderr)
            _print_budget(client.budget)
            artifact |= {
                "ok": False,
                "error": str(exc),
                "stopped": "cost_cap",
                "raw_text": cause.raw_text if isinstance(cause, ResponseParseError) else None,
                "spent_usd": str(client.budget.spent),
            }
            _save(output, artifact)
            return 3
        except ProviderError as exc:
            print(f"\nLLM call failed: {exc}", file=sys.stderr)
            print(f"attempts={exc.attempts} latency_ms={exc.latency_ms:.0f}", file=sys.stderr)
            artifact |= {
                "ok": False,
                "error": str(exc),
                "status_code": exc.status_code,
                "attempts": exc.attempts,
                "latency_ms": round(exc.latency_ms),
            }
            _save(output, artifact)
            return 1

    parsed = result.parsed
    assert parsed is not None
    if show_raw:
        print("\n--- raw model output ---")
        print(result.text)
        print("--- parsed (what the schema made of it) ---")
    print(parsed.model_dump_json(indent=2))
    validated = validate_line_items(parsed)
    _print_repaired(result.repair_history, file=sys.stdout)
    _print_validators(parsed, validated)
    _print_stats(
        result.usage,
        result.cost_usd,
        result.latency_ms,
        result.attempts,
        result.repairs,
        result.structured_mode_used,
        result.finish_reason,
        result.cached_calls,
        result.avoided_cost_usd,
    )
    _print_budget(client.budget)
    artifact |= {
        "ok": True,
        "structured_mode_used": result.structured_mode_used,
        "finish_reason": result.finish_reason,
        "attempts": result.attempts,
        "repairs": result.repairs,
        "latency_ms": round(result.latency_ms),
        "usage": result.usage.model_dump(mode="json"),
        "provider_usage": result.raw.get("usage"),  # exactly what the server reported
        "cost_usd": str(result.cost_usd),
        "cached_calls": result.cached_calls,
        "avoided_cost_usd": str(result.avoided_cost_usd),
        "raw_text": result.text,
        "parsed": parsed.model_dump(mode="json"),
        "validated": validated.model_dump(mode="json"),
        "repair_history": _history_json(result.repair_history),
    }
    _save(output, artifact)
    return 0


def _history_json(history: Sequence[RepairRecord]) -> list[dict[str, str]]:
    return [{"problem": record.problem, "raw_text": record.raw_text} for record in history]


def _save(path: Path, artifact: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"(could not write {path}: {exc})", file=sys.stderr)
        return
    print(f"artifact      : {path}")


def _print_repaired(history: Sequence[RepairRecord], *, file: TextIO) -> None:
    if not history:
        return
    print(
        f"\n--- repaired failures ({len(history)}): the model's first output was unusable ---",
        file=file,
    )
    for number, record in enumerate(history, start=1):
        print(f"attempt {number}: {record.problem}", file=file)
        excerpt = record.raw_text[:RAW_EXCERPT_CHARS]
        more = " ..." if len(record.raw_text) > RAW_EXCERPT_CHARS else ""
        print(f"  raw output: {excerpt}{more}", file=file)


def _print_validators(raw: DocumentExtraction, validated: DocumentExtraction) -> None:
    flagged = [
        (number, before, after)
        for number, (before, after) in enumerate(
            zip(raw.line_items, validated.line_items, strict=True), start=1
        )
        if after.flagged_cells
    ]
    print("\n--- validators ---")
    print(f"line-item arithmetic: {len(raw.line_items)} row(s) checked, {len(flagged)} flagged")
    for number, before, after in flagged:
        label = repr(before.description) if before.description else "(no description)"
        print(
            f"  row {number} {label}: {', '.join(after.reasons)} -> flagged {after.flagged_cells};"
            f" confidence {before.confidence} -> {after.confidence}"
        )


def _print_stats(
    usage: Usage | None,
    cost_usd: Decimal,
    latency_ms: float,
    attempts: int,
    repairs: int,
    mode: str | None,
    finish_reason: str | None,
    cached_calls: int = 0,
    avoided_cost_usd: Decimal = Decimal(0),
) -> None:
    tokens = usage or Usage()
    reported = "" if tokens.reported else " (server reported no usage)"
    print("\n--- stats ---")
    print(f"tokens in/out : {tokens.input_tokens} / {tokens.billable_output_tokens}{reported}")
    if tokens.reasoning_outside_output:
        print(
            f"                (output = {tokens.output_tokens} completion"
            f" + {tokens.reasoning_outside_output} reasoning the provider left out of it)"
        )
    print(f"reasoning     : {tokens.reasoning_note}")
    print(f"latency       : {latency_ms:.0f} ms over {attempts} attempt(s), {repairs} repair(s)")
    print(f"cost          : ${cost_usd:.6f}")
    if cached_calls:
        print(
            f"cache         : {cached_calls} call(s) answered from the cache"
            f" (would have cost ${avoided_cost_usd:.6f})"
        )
    if mode is not None:
        print(f"structured    : {mode}   finish_reason={finish_reason}")


def _print_budget(budget: RunBudget) -> None:
    if budget.limit is None:
        return
    unmetered = (
        f"   ({budget.unmetered_calls} call(s) returned no usage and are not counted)"
        if budget.unmetered_calls
        else ""
    )
    print(f"budget        : ${budget.spent:.6f} spent of the ${budget.limit:.6f} cap{unmetered}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Send one image to the configured LLM; print the extraction, tokens, latency."
    )
    parser.add_argument("image", nargs="?", help="image to send (default: data/raw, then fixture)")
    parser.add_argument(
        "--env-file", type=Path, default=ROOT / ".env", help="env file (default: repo .env)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON artifact: raw text, parsed result, usage (default: outputs/smoke-last.json)",
    )
    parser.add_argument("--raw", action="store_true", help="print the model's raw text first")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="allow the response cache (default: bypassed, a smoke check must hit the real API)",
    )
    args = parser.parse_args()
    return asyncio.run(
        run(pick_image(args.image), args.env_file, args.output, args.raw, args.cache)
    )


if __name__ == "__main__":
    sys.exit(main())
