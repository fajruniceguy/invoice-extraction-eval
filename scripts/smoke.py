"""Send one image to the configured LLM and print the parsed extraction, tokens and latency.

Usage: python -m uv run python scripts/smoke.py [IMAGE] [--env-file PATH]

IMAGE defaults to the first image in data/raw, then to tests/fixtures/sample_receipt.png.
Configuration (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, ...) comes from the environment, then
from the env file (default: .env at the repo root).

Exit codes: 0 success, 1 the call or the model output failed, 2 configuration/input problem.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError

from docextract.config import Settings
from docextract.extract.schema import DocumentExtraction
from docextract.providers.openai_compat import (
    OpenAICompatClient,
    ProviderError,
    ResponseParseError,
    Usage,
    build_image_message,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_receipt.png"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

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


async def run(image: Path, env_file: Path) -> int:
    if not image.is_file():
        hint = (
            "  generate it with: uv run python scripts/make_sample_receipt.py"
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

    try:
        message = build_image_message(USER_PROMPT, [image])
    except ValueError as exc:
        print(f"Cannot use {image}: {exc}", file=sys.stderr)
        return 2

    print(f"model={settings.llm_model} endpoint={settings.llm_base_url} image={image.name}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, message]
    async with OpenAICompatClient(settings) as client:
        try:
            result = await client.chat(messages, DocumentExtraction, temperature=0.0)
        except ResponseParseError as exc:
            print(f"\nModel output was not usable: {exc}", file=sys.stderr)
            print(f"--- raw output (first 2000 chars) ---\n{exc.raw_text[:2000]}", file=sys.stderr)
            _print_stats(settings, exc.usage, exc.latency_ms, exc.attempts, None, None)
            return 1
        except ProviderError as exc:
            print(f"\nLLM call failed: {exc}", file=sys.stderr)
            print(f"attempts={exc.attempts} latency_ms={exc.latency_ms:.0f}", file=sys.stderr)
            return 1

    assert result.parsed is not None
    print(result.parsed.model_dump_json(indent=2))
    _print_stats(
        settings,
        result.usage,
        result.latency_ms,
        result.attempts,
        result.structured_mode_used,
        result.finish_reason,
    )
    return 0


def _print_stats(
    settings: Settings,
    usage: Usage | None,
    latency_ms: float,
    attempts: int,
    mode: str | None,
    finish_reason: str | None,
) -> None:
    tokens = usage or Usage()
    cost = settings.cost_usd(tokens.input_tokens, tokens.output_tokens)
    reported = "" if tokens.reported else " (server reported no usage)"
    print("\n--- stats ---")
    print(f"tokens in/out : {tokens.input_tokens} / {tokens.output_tokens}{reported}")
    print(f"latency       : {latency_ms:.0f} ms over {attempts} attempt(s)")
    print(f"cost          : ${cost:.6f}")
    if mode is not None:
        print(f"structured    : {mode}   finish_reason={finish_reason}")


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
    args = parser.parse_args()
    return asyncio.run(run(pick_image(args.image), args.env_file))


if __name__ == "__main__":
    sys.exit(main())
