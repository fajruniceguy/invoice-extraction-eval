# CLAUDE.md

Project rules for Claude Code. Read this before making changes.

A document extraction system for invoices/receipts plus a rigorous evaluation harness.

## Stack and tooling
- Python 3.11 (project venv in `.venv`, created by uv), `uv` for dependencies, `ruff` + `pytest`.
- Type hints everywhere (ruff `ANN` enforces it in `src/` and `scripts/`). Pydantic v2 for all schemas.

## Layout
```
src/docextract/{config.py, providers/, ingest/, extract/, validate/, confidence/, api/}
src/evalkit/{comparators.py, matching.py, metrics.py, taxonomy.py, scorecard.py, compare.py, groundtruth.py}
data/{manifest.csv, redactions.yaml, pages/, private/{raw,preview}}
ground_truth/{, drafts/}
outputs/
scripts/
tests/{fakes/, fixtures/}
docs/
```
`groundtruth.py` (the ground-truth schema) is the only file in `evalkit` not in the original layout brief.

## Provider
- The provider is OpenAI-compatible and fully configured via env: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, plus per-token prices for cost (`LLM_PRICE_INPUT_PER_MTOK`, `LLM_PRICE_OUTPUT_PER_MTOK`, USD per 1M tokens).
- Optional env: `LLM_MAX_REPAIRS` (default 1: re-asks when output is not valid JSON / does not match the schema), `LLM_REASONING_EFFORT` (passed through as `reasoning_effort` when set), `LLM_STRUCTURED_MODE`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `LLM_BACKOFF_BASE_S`, `LLM_BACKOFF_MAX_S`, and `MAX_RUN_COST_USD` (below). `.env.example` lists them all.
- Never hardcode model names or secrets. `.env` is gitignored and holds a real key: never print it, never edit it, never commit it. `.env.example` is the only committed example.
- Tests and fakes must never read `.env` or call the real API (`scripts/smoke.py --env-file <empty file>` plus a scrubbed environment; see `tests/test_provider_integration.py`).
- Reasoning ("thinking") tokens: providers disagree on whether `completion_tokens` already includes them. `Usage` settles it per response from `total_tokens` (see `parse_usage`), so cost never double-counts. Always cost a call with `Usage.cost_usd(settings)`, never from `output_tokens` alone.

## Page images: two different sizes (`docextract/imaging.py`)
- The STORED size is a ceiling applied once at ingestion (`INGEST_MAX_LONG_EDGE`, 2400 px). The SENT size is `LLM_IMAGE_MAX_LONG_EDGE`, derived in memory from the stored page per request (`build_image_message(max_long_edge=...)`), so a resolution sweep never needs a re-ingest and the stored page is never modified. Both use the same rule.
- Rule: normal pages cap the long edge; elongated pages (long/short > 2.5, thermal receipts) cap the AREA (~6 MP at E=2400, scaling with E squared) so the width stays legible; no upscaling; no side over 8000 px.
- Encoding: PDF renders and PNG sources are stored as PNG, JPEG/HEIC photos as JPEG q92. Every encode strips ALL metadata, including what Pillow would otherwise re-emit from `.info` (comment, ICC, DPI): never save a page without going through `encode_page`.

## Response cache (`LLM_CACHE=on|off|refresh`)
- Content-hash cache of RAW completions (`providers/cache.py`), keyed per HTTP exchange on page-image hashes, prompt, model, endpoint, reasoning effort, temperature, max_tokens, structured mode and schema hash. Parsing, normalization and validation rerun on every replay, so a validator change never serves a stale result and repair sequences replay deterministically.
- On by default (eval); `smoke.py` bypasses it unless `--cache`, because a smoke check must hit the real API. `chat(cache="off"|"refresh")` overrides per call. Tests force it off and into a tmp dir (autouse fixture in `tests/conftest.py`).
- A hit costs 0: `LLMResponse.cost_usd` is what was actually paid, `avoided_cost_usd` what the hits would have cost, `cached_calls`/`cache_hit` say how much came from the cache, and the budget counts cached calls apart. Scorecards must report real and cached cost separately. Hits are never refused by the cost cap.
- It replays the FIRST answer ever seen for a key: use `refresh` (or `off`) when you actually want a new sample.

## Cost cap (`MAX_RUN_COST_USD`)
- The client records the cost of every completed call (repairs included) in a `RunBudget` and refuses to start a new call or repair once spend exceeds the cap, raising `BudgetExceededError`. The call that crosses the cap completes; concurrent in-flight calls can overshoot by up to concurrency x one call's cost.
- Settings refuse a cap while both prices are 0 (it could never trigger).
- **Batch and eval runners must** catch `BudgetExceededError` (it is deliberately not a `ProviderError`), stop dispatching, flush partial results, and exit non-zero. Do not let a per-document `except` swallow it.

## Schema and validation
- `extract/schema.py` only defines shape. Field and line-item status is `extracted | absent | abstained`: absent = the document does not state it (value null, confidence = certainty of absence); abstained = may be present but not confident (value null, reasons required).
- Business rules live in `validate/` (first rule: line-item arithmetic). They return copies and never mutate the model's raw output, so what the model said vs what the validators concluded is always inspectable. `flagged_cells` is derived there, never asked of the model.
- Money is `Decimal` end to end: model JSON is parsed with `parse_float=Decimal`.
- **Normalization (`docextract/normalize.py`) never guesses.** `parse_number` / `parse_date` return `ok | ambiguous | invalid`; `1.250` is thousands in an IDR context, a decimal in a USD context, and **ambiguous** (with both candidates) with no context; `05/03/2024` is ambiguous unless a day is above 12 or `day_first` is known (IDR documents are read day-first). A currency marker in the text beats the caller's currency. Shared by the extraction pipeline, `prelabel` and the label app.
- **Order in the pipeline: parse JSON -> normalize -> validate -> repair only if that still fails** (`chat(preprocess=normalize_extraction_payload)`). A value the normalizer declined to resolve is an error even if pydantic could parse it (`Decimal("1.250")` parses fine, and accepting it would make the guess the normalizer refused to make). Every rewrite is reported in `LLMResponse.normalizations`.

## Core logic
- Extraction logic and the eval harness are hand-written. No LangChain / LlamaIndex for core logic.

## Dataset and ingestion (`src/docextract/ingest/`)
- **Privacy by location.** Real originals live in `data/private/raw/` and unredacted previews in `data/private/preview/`: that whole tree is gitignored and must never be committed. Only normalized, REDACTED pages (`data/pages/<doc_id>/p<N>.png|jpg`), `data/manifest.csv`, `data/redactions.yaml` and ground truth are committed. Do not read page content of real documents while building tooling: use synthetic fixtures (`tests/files.py`), and for real files only metadata, counts and sizes.
- `data/manifest.csv` has one row per LOGICAL document (14 columns + optional `gt_from`, `variant`). `scripts/manifest_init.py` scans the raw folder and appends rows for unreferenced files only (merge-safe, neutral `doc_NNN` ids, only strong-evidence inference); the rest is filled by hand. A row is never overwritten.
- `scripts/ingest.py` makes pages: EXIF upright -> PDFs at `INGEST_DPI` -> size cap (`INGEST_MAX_LONG_EDGE`, elongated pages by area) -> redaction -> encode (PDF/PNG sources -> PNG, JPEG/HEIC photos -> JPEG q92, ALL metadata stripped). Idempotent via a content hash recorded in `ingest.json` (which holds sizes, hashes and page mapping but never a raw file name). Never encode a page except through `imaging.encode_page`.
- **Redaction is a hard gate:** blank `needs_redaction`, or `y` without boxes, is refused (nothing is written); `n` with boxes is refused; boxes must fit the stored page. Burn BEFORE encoding. Boxes are in stored-page pixels.
- Windows: a long raw file name (200+ chars) makes plain `Path.is_file()` return False and the file silently vanishes: every listing and read of raw files goes through `fsutil.long_path`.

## Ground truth
- Never edit `ground_truth/*.json` silently. Every change gets a line in `ground_truth/CHANGELOG.md`.

## Workflow
- Small conventional commits after each working step.
- Append a short entry to `docs/DEVLOG.md` per milestone (what you did, decisions, problems).
- Run `pytest -q` and `ruff check` before every commit.
- Commit with the existing global git identity; do not change git config or rewrite authorship.
- Stage explicit paths, not `git add -A`: the working tree can hold edits that are not yours (e.g. a moved `.env.example`).

## Commands (this machine: Windows 10, PowerShell)
`uv` is a per-user `pip install` whose Scripts directory is not on PATH, and bare `python` is the 3.10 Microsoft Store stub. So call uv as `python -m uv ...`; it runs the project's 3.11 venv for you. (Where `uv` is on PATH, plain `uv ...` is equivalent.)
```
python -m uv sync                                   # create/update .venv (3.11) and install deps
python -m uv run pytest -q                          # default suite: offline, ~3 s (tests marked slow are excluded)
python -m uv run pytest -q -m slow                  # ONLY the slow tests (they spawn subprocesses), ~13 s
python -m uv run pytest -q -m "slow or not slow"    # everything (run this before a milestone commit / report)
python -m uv run pytest -q tests/test_budget.py     # one file
python -m uv run ruff check                         # lint
python -m uv run ruff format .                      # format
python -m uv run python scripts/manifest_init.py [--dry-run]   # scan data/private/raw, append manifest rows
python -m uv run python scripts/ingest.py [--doc ID] [--check] [--force]   # raw -> data/pages (idempotent)
python -m uv run python scripts/make_sample_receipt.py [OUT.png]   # regenerate the synthetic fixture (deterministic)
python -m uv run python scripts/smoke.py [IMAGE] [--raw]   # ONE LIVE CALL using .env: costs money; writes outputs/smoke-last.json
python -m uv run python tests/fakes/fake_llm.py 8765 [--reject-schema]   # fake OpenAI-compatible server, no key needed
```
To exercise `smoke.py` without a key: start the fake server in one terminal, then in another set `LLM_BASE_URL=http://127.0.0.1:8765/v1`, `LLM_MODEL=fake`, `LLM_API_KEY=x` (and prices if you want a cost) and pass `--env-file` pointing at an empty file so the real `.env` is not read.

Shell notes (these bit us):
- Commit messages: `git commit -m @'...'@` with the closing `'@` at column 0. `git commit -F -` does not take a PowerShell here-string.
- Do not put `2>&1` on native commands in Windows PowerShell 5.1: uv's normal stderr progress lines get reported as `NativeCommandError`.
- In Claude Code's Bash tool a backslash is un-escaped once more inside heredocs, so a script containing `\n` and friends gets silently corrupted. Write such scripts with the Write tool, not `cat <<EOF`.
- Mutation-test new logic (temporarily break it, confirm a test fails, restore): it has found vacuous tests here more than once.
