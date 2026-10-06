# CLAUDE.md

Project rules for Claude Code. Read this before making changes.

A document extraction system for invoices/receipts plus a rigorous evaluation harness.

## Stack and tooling
- Python 3.11, `uv` for dependencies, `ruff` + `pytest`.
- Type hints everywhere. Pydantic v2 for all schemas.

## Layout
```
src/docextract/{config.py, providers/, ingest/, extract/, validate/, confidence/, api/}
src/evalkit/{comparators.py, matching.py, metrics.py, taxonomy.py, scorecard.py, compare.py, groundtruth.py}
data/{raw, pages}
ground_truth/
outputs/
tests/
docs/
```
`groundtruth.py` (the ground-truth schema) is the only file in `evalkit` not in the original layout brief.

## Provider
- The provider is OpenAI-compatible and fully configured via env: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, plus per-token prices for cost (`LLM_PRICE_INPUT_PER_MTOK`, `LLM_PRICE_OUTPUT_PER_MTOK`, USD per 1M tokens).
- Never hardcode model names or secrets. `.env` is gitignored; `.env.example` is the only committed example.

## Core logic
- Extraction logic and the eval harness are hand-written. No LangChain / LlamaIndex for core logic.

## Ground truth
- Never edit `ground_truth/*.json` silently. Every change gets a line in `ground_truth/CHANGELOG.md`.

## Workflow
- Small conventional commits after each working step.
- Append a short entry to `docs/DEVLOG.md` per milestone (what you did, decisions, problems).
- Run `pytest -q` and `ruff check` before every commit.

## Commands
```
uv sync                              # install deps
uv run pytest -q                     # tests
uv run ruff check                    # lint
uv run ruff format .                 # format
uv run python scripts/smoke.py       # one live call (needs LLM_* in .env)
```
