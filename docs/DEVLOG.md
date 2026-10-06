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

## M1.1 - Fixes before M2 (2026-10-06)

**Done**, one commit per item (hashes in brackets):
1. Three-way status `extracted | absent | abstained` on `Field` and `LineItem` [93ea29e]; line-item
   arithmetic rule in `validate/` that fills `flagged_cells` [718adbc].
2. Model JSON parsed with `parse_float=Decimal` [0b8a7b4].
3. Repair of unusable output, `LLM_MAX_REPAIRS` [56f8e04]; fake LLM server promoted to
   `tests/fakes/` with real-socket integration tests [4f9ccba].
4. Reasoning tokens costed without double counting, `LLM_REASONING_EFFORT` [64ce2e2].
5. `MAX_RUN_COST_USD` run budget [56fc3de].
6. `CLAUDE.md` commands match this machine (`python -m uv ...`) [54f56c0].
Follow-ups found while doing the above or on the live run: repair history [30f8eb5], smoke
`--raw` + JSON artifact [a52d375, 67f4cfb], provider message kept when retries run out
[858b3d7, 01fc892], retired example model replaced in `.env.example` [530e632].

**Decisions.**
- `flagged_cells`: arithmetic mismatch only, tolerance `max(0.01, 0.5% x |amount|)` (rounded
  Rupiah, items sold by weight). It lives in `validate/`, returns copies, and the model is never
  asked for it (hidden from the JSON schema). A mismatch adds reason `qty_x_price_ne_amount` and
  caps row confidence at 0.5. A cap, not a multiplier, so validating twice cannot lower it twice;
  the status is never changed (a per-line discount can legitimately follow on the next line).
- Reasoning tokens are settled per response from `total_tokens` (gap > 0: provider left them out,
  bill them on top; gap = 0: already inside, do not add again; no total: assume OpenAI
  convention, flagged unverified). Providers disagree, so neither field is trusted alone.
- The cost cap is checked before every call and repair and is not a `ProviderError`; settings
  refuse a cap while both prices are 0. A repair refused by the cap keeps the parse error as cause.
- Repair lives in the client (summed usage/cost, skipped on truncation) and records what it
  papered over, so a repair cannot hide a strict-rule violation.
- Tests never read the repo's `.env`: smoke tests pass `--env-file` (empty) and `--output` (tmp).

**Problems.**
- `git add -A` swept a `.env.example` deletion (a user-side move) into the first commit; caught
  immediately, amended, and CLAUDE.md now says to stage explicit paths.
- Claude Code's Bash tool un-escapes backslashes once more inside heredocs: two patch scripts
  silently corrupted a string. The new tests and a compile check caught both. Use the Write tool.
- Mutation checks found two vacuous tests (repair-prompt cap, usage aggregation); fixed.
- The 0.5 s default `serve_forever` poll made every fake-server test slow; now 20 ms. The suite is
  still ~14 s because the smoke tests spawn real subprocesses.
- A hidden `2>$null` swallowed a failed `git commit` once; the retry showed it, so it is gone.

**Live run (Google AI Studio OpenAI-compatible endpoint, synthetic receipt).**
- `.env` asks for `gemini-2.5-flash`: HTTP 404 "no longer available to new users" (clean error, not
  retried). That id came from this repo's own `.env.example`. `.env` was NOT edited: model and
  list prices were passed as per-process env overrides.
- `gemini-3.8/3.7/3.6-flash`: 503 "high demand" through several retries (once a 120 s read
  timeout), nothing billed. `gemini-3.5-flash` answered ($1.50 in / $9.00 out per 1M, list).
- Run 1 (default reasoning): `json_schema` accepted, no fallback; 0 repairs; all 13 fields and 3
  rows valid under the strict rules; raw and parsed agree and money keeps its scale (`4.50`,
  `9.00`, `16.00`); validators flagged nothing. Tokens 1388 in, 1185 completion + 1320 reasoning
  (completion_tokens excludes reasoning; inferred from total_tokens 3893) = 2505 billable out;
  cost $0.024627 at list price; a naive cost would have been $0.012747, 48% too low. 28 s,
  including one 503 retry.
- Run 2 (`reasoning_effort=low`): accepted; raw usage was just `{completion_tokens: 1192,
  prompt_tokens: 1388, total_tokens: 2580}` (no `completion_tokens_details`), no hidden reasoning,
  $0.01281, 21.5 s, 0 repairs.
- Caveat: one clean synthetic receipt says little about the strict rules on messy real ones.

**Not verified.**
- Real (Indonesian) receipts: Rupiah formats such as `9.116,8` fail the plain-decimal contract and
  would go to repair; that normalization is not built.
- The json_object fallback and a repair against a real model: json_schema was accepted and no repair
  was needed, so both are proven only against the fake server.
- Gemini 3.6-3.8 behaviour: overloaded during the run, so only 3.5 was exercised.
