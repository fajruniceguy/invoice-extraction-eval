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

## M2 - Dataset, ingestion and ground-truth tooling (2026-10-06)

**Done.** Everything between a folder of raw documents and a labeled, split, redacted dataset.
Groundwork first (slow-test marker; content-hash response cache; shared page geometry with a
request-time resolution; the number/date normalizer run BEFORE any repair), then the ten steps,
one commit each:

1. `manifest.csv` + Pydantic model + `manifest_init` (merge-safe, infers only on strong evidence)
2. `ingest`: EXIF upright, all metadata stripped, PDFs at 200 dpi, 2400 px ceiling (elongated
   receipts capped by area), PNG for PDF/PNG sources and JPEG q92 for photos, idempotent by content hash
3. redaction: `redactions.yaml` burned in before encoding, a hard gate, `--preview` for coordinates
4. `assign_split`: ~35/65 by content group, stratified, never moves an existing assignment, 8-10 blind picks
5. `degrade_docs`: light/medium/heavy variants (Augraphy, OpenCV fallback) of born-digital documents
6. `construct_invoices`: 3 multi-page invoices (3 layouts, 30-60 items, totals on the last page only)
   with labels known by construction
7. ground-truth format (`label_mode`, `labeled_at`, `field_notes`), `gt_store`, `LABELING.md`,
   `check_ground_truth`
8. `prelabel`: a DIFFERENT model drafts, a human verifies (PRELABEL_* config, same client class)
9. the labeling app (Streamlit) over a pure, tested form state machine
10. an end-to-end test of the real scripts in order, which found one more bug (below)

1772 default tests (~52 s) + 128 `slow` ones (~2 min) = 1900, all offline. Every new piece of logic
was mutation-tested (break it, confirm a test fails): over 400 mutants in steps 3-10 alone, plus those
of the groundwork and steps 1-2. Survivors were either fixed with a stronger test or shown to be
equivalent (a redundant branch, an implementation choice) and the code simplified.

**Decisions.**
- **Privacy by location.** Originals and unredacted previews live in `data/private/` (gitignored);
  only redacted pages, the manifest, redaction boxes and labels are committed. Tooling never reads real
  page content: it uses synthetic fixtures, and for real files only metadata and counts.
- **Redaction is a hard gate and its parser is strict**: a repeated YAML key (PyYAML silently keeps the
  last), an empty entry, an unknown doc_id or boxes on a variant stop the run before any page is written.
  A box left untouched while its page changed size (new DPI or cap) is refused.
- **The stored resolution is only a ceiling**; what the model is sent is a request-time setting, so a
  resolution sweep needs no re-ingest.
- **The unit of the split is the content group** (a gt_from row must share its target's group), and
  assignments are ranked by a hash of (seed, stratum, group), so adding a document never reshuffles
  the others. A label_mode already in the manifest is a decision: a document that may carry a
  model-influenced label can never turn blind.
- **Label once, never copy**: variants and duplicate photos borrow their owner's label (`gt_from`)
  and must not have a file of their own.
- **Pre-labeling by a different model**, enforced in code: blind documents are refused (zero requests
  are made), same endpoint+model as the extractor is refused, only stored redacted pages are sent, and
  what the normalizer cannot settle without guessing becomes `"__unlabeled__"` with a note.
- **Save is blocked until every field is decided** (verified / absent / unlabeled), and a label is never
  written without its CHANGELOG line (if the log fails, the label file is put back).
- **Values are labeled AS PRINTED** (Indonesian "1.250.000,00" -> "1250000.00", printed scale kept), a
  faktur with no printed grand total has `total = null`; see `ground_truth/LABELING.md`.

**Problems found and fixed** (the ones worth remembering):
- A 218-character file name made `Path.is_file()` False on Windows, so the file silently vanished from
  the scan. Everything now goes through `fsutil.long_path`.
- Pillow re-emitted comment/ICC/DPI from `.info`, and a transparent PNG became black on `convert("RGB")`.
- Pydantic accepted an ambiguous "1.250" the normalizer had refused to resolve: normalizer problems now
  force a parse error.
- Augraphy: its seed goes to `cv2.setRNGSeed` (signed 32-bit) and 0 means "unseeded"; `BadPhotoCopy` is
  not reproducible inside a pipeline (dropped); it writes 13 MB into `./augraphy_cache` (now run inside
  `outputs/cache/augraphy`). Found only by running the real pipeline several times over.
- The labeling form trusted a saved "verified" blindly and crashed on malformed saved work; both fixed
  (shape checked, every verified field re-verified on load).
- The end-to-end test found that `assign_split` refused to give the constructed invoices (which have
  generated labels from the start) a FIRST split: a lock now stops moves, not first assignments.
- Heredoc backslash-halving corrupted scripts more than once, even after it was written down: anything
  containing `\n` goes through the Edit/Write tools, never `cat <<EOF`.

**Real documents** (metadata only, never viewed): 31 files ingested into a scratch folder, 133 pages,
~88 MB, 0 failures, a second run all unchanged. `doc_017` (78 pages, 51 MB) and `doc_010` (20 pages,
22 MB) are bundles that need `page_range` rows before they are usable. WhatsApp photos carry no EXIF
(so `capture` cannot be inferred for them), one `.png` is really a JPEG, and the raw file names are in
the (uncommitted) manifest: review them before committing it.

**Not done / left for the human.** `doc_type`, `template_group`, `needs_redaction` (and the boxes),
`license_note`, duplicates (`gt_from`) and bundle splitting for the 31 real documents; then
`ingest`, `assign_split`, `degrade_docs`, `prelabel` and labeling, in that order. The prelabel path
has been tested against a fake server only: no live call was made.

## Baseline - ROADMAP vs. repo (2026-10-07)
**State.** `pytest -q`: 1773 passed, 1 skipped (+128 slow); ruff clean. Roadmap vs. tree:
- M1 done: `config.py`, `providers/openai_compat.py`, `extract/schema.py`; live smoke in `outputs/smoke-run*.json`.
- M2 tooling done, deliverable not: ingest, redact, split, degrade, construct, prelabel and the label app are
  built and tested, but the manifest has 34 rows (target 40), 31 real ones with no doc_type, none with a
  split, `data/pages/` is empty, only `ground_truth/cons_00{1,2,3}.json` exist, no GT hash is recorded.
- M3 partial: repair in the client, `Field` evidence/page/reasons, `extract/preprocess.py`; no prompt/extractor/runner.
- M4 partial: `validate/arithmetic.py` only; `confidence/` empty (no date/currency/tax-ID, grounding, abstention).
- M5, M6 untouched: `evalkit/{comparators,matching,metrics,taxonomy,scorecard,compare}.py` are 1-line stubs.
- M7 partial: per-call tokens/cost/latency + `RunBudget` exist; `api/` empty, no FastAPI, no batch runner.
- M8-M10 untouched: no sweep, Dockerfile, compose, README or CI. Beyond the roadmap: N3 cache (`providers/cache.py`).
**Deviations.** ROADMAP.md is at the repo root and untracked (CLAUDE.md says `docs/ROADMAP.md`); `docs/milestones/` is absent.
**Next: M3** (extractor + dev-set runner), buildable offline on the fake server and the 3 constructed invoices.
Its DoD run on "all dev docs" is gated by M2's human steps (classify, redact, split, blind-label), not by code.

## M3 - Extraction v1 (2026-10-07)
**Outcome.** All 8 spec criteria pass; `pytest -m "slow or not slow"` 2012 passed, 1 skipped; ruff clean. 82 mutants, 5 survivors closed with stronger tests.
Criterion 8 (live) passed on **gemini-3.6-flash**, not the planned 3.5-flash: 3.5 answered 503 for cons_001 on ~14 attempts, then HTTP 429 (free tier, 20 requests/day
per model; 503s and their retries count). On 3.6: cons_001/2/3 = 45/57/33 rows (= labels), subtotal/tax/total = labels, every value cited on a real page,
0 repairs, json_schema accepted. Cost at list price: $0.19 on 3.5 (cons_002/003 earlier), $0.11 on 3.6.
**New interfaces.** `ingest.stored.stored_pages` (ingest.json or degrade.json, hash-checked; prelabel delegates); `extract.prompt` (SYSTEM_PROMPT,
`user_text`, `PROMPT_SHA256`); `extract.preprocess.check_contract` / `extraction_preprocess` + `PreprocessResult.violations` (uncited or off-page
values are repaired); `extract.extractor.extract_pages` (all pages, one call); `extract.runner` (`run_extraction`, `DocRecord`, `RunRecord`, blind rule,
exit codes) + `scripts/extract.py` -> `outputs/runs/<id>/`; `LLM_MAX_PAGES` (15).
**Deviations.** No dev split exists: DoD met on a synthetic dev set offline + the constructed invoices live (on 3.6-flash). A repair never happened live (offline only).
**Tech debt.** smoke.py's default image folder is the old `data/raw`; attempts that time out or drop return no usage, so their cost (if billed) is unknown; HTTP
retries burn a daily request quota: use `LLM_MAX_RETRIES=0-1` on a limited key.
**Next.** M2's human steps (classify, redact, split, label), then M4; fix the extractor model (3.6-flash vs 3.5-flash) before M5 so scorecards compare like with like.
