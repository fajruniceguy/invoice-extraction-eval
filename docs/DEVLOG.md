# DEVLOG

One short entry per milestone: what was done, decisions, problems.

## M1 - Scaffolding (2026-10-06)

**Done.** Repo skeleton and tooling (uv, Python 3.11, ruff with `ANN`, pytest-asyncio);
env-driven `Settings`; `OpenAICompatClient` (multi-image base64 messages, json_schema with
json_object fallback, retry/backoff on 429/5xx/transport errors, usage + latency + cost on
every call); extraction schema (`Field[T]`, `LineItem`, `DocumentExtraction`); ground-truth
schema with the value / null / `"__unlabeled__"` tri-state; smoke script and a synthetic
receipt fixture. 193 tests, all offline.

**Decisions.**
- Raw `httpx` instead of the `openai` SDK: full control of retries, latency and usage, and
  tests use `httpx.MockTransport`.
- Line items carry one confidence/evidence/page per row, not per cell (cheaper, rows are what
  we match and score).
- `Field[T]` enforces `abstained <=> value is None`. Strict on purpose; repair belongs to a
  later milestone.
- Ground truth defaults every field to unlabeled and rejects unknown keys, so a typo like
  `vendor_nme` fails loudly instead of silently skipping a field. Lives in
  `evalkit/groundtruth.py`, a file the original layout did not list.
- Prices are USD per 1M tokens (`LLM_PRICE_INPUT_PER_MTOK` / `LLM_PRICE_OUTPUT_PER_MTOK`).
- `latency_ms` is the whole call including retries and backoff; `attempts` says how many HTTP
  requests that was. A failed call still reports attempts, latency and usage.
- The json_schema -> json_object downgrade is remembered per client, and only once json_object
  actually worked, so one bad-image 400 cannot permanently downgrade a client.
- `build_image_message` sniffs PNG/JPEG/WebP and refuses zero images: an empty page list must
  not silently reach the model.

**Problems.**
- `uv` was not installed and the only Python was 3.10: `pip install --user uv`, then
  `uv python install 3.11`. Call it as `python -m uv` if its Scripts dir is not on PATH.
- My plan said 16 document fields; the spec has 14 (13 `Field[T]` + `line_items`).
- A latency test asserting `>= 50 ms` after `asyncio.sleep(0.05)` failed once: Windows timer
  ticks (~15 ms) let sleeps return early. Now sleeps 100 ms and asserts `>= 80`.
- PowerShell-written stub files came out with CRLF/BOM; ruff `line-ending = "lf"` pins it.
- `git commit -F -` does not take a PowerShell here-string; use `-m @'...'@`.
- Global git identity (`fajruniceguy`) differs from the account email; left untouched.

**Not verified (no API key in this environment).**
- No live provider call has been made. The smoke script was run only against a local fake
  server (real sockets, canned reply), which proves wiring, not model behavior.
- Whether Gemini's compat endpoint accepts the simplified json_schema, or falls back cleanly.
- Whether the strict `Field` invariants are too strict for real model output.
- Token accounting for "thinking" models: reasoning tokens may or may not be inside
  `completion_tokens`; cost could be understated.
