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

## M4 - Validators, confidence and abstention (2026-10-07)
**Outcome.** All 10 spec criteria pass (9 offline + the live one); `pytest -m "slow or not slow"` 2204 passed, 1 skipped; ruff clean. 18 mutants; two vacuous tests found and closed (the tolerance floor was never exercised — the relative part always dominated; the "no `parsed` output" guard of the checker was untested).
**New interfaces.** `validate/codes.py` (`REASON_CODES`, `HARD_CODES`); `arithmetic.validate_document_totals`; `fields.validate_fields`; `grounding.validate_grounding`; `validate.validate_document(*, abstain_below, normalizations, today)`; `confidence/formula` (`ARITHMETIC_CAP`/`SHAPE_CAP`/`NORMALIZED_CAP`, `CAP_BY_CODE`, `cap_for`, `apply_confidence`) + `confidence/abstain` (`should_abstain`, `apply_abstention`); `ABSTAIN_BELOW` (0.3); `normalize.currency_marker` (+ a "rupiah" marker); `preprocess.change_label`; `scripts/check_run.py`.
**Criterion 10 (live, cons_001..003).** Replayed from M3's response cache: $0.00 paid ($0.1141 avoided; the prompt and the JSON schema are unchanged), 3 ok, 0 repairs, 39 fields / 135 rows all carrying value/confidence/evidence/status/reasons, 0 capped, 0 abstained, 0 flagged cells, `check_run` exit 0. These are the easiest documents there are, so it says nothing about messy receipts.
**Decisions.** Grounding is self-consistency (a value must be traceable in its own evidence quote), not "against page text" as M3's spec phrased it: the stack has no page text (rasterized pages, no OCR), and a PDF text sidecar would survive redaction as text — decided with you. Confidence is the model's score lowered by caps (never raised, idempotent); an ungrounded value and a due date before the document date are HARD (abstain), shape findings and a rewritten value only cap.
**Deviations.** The DoD checker cannot demand codes-only reasons: the model writes its own prose on fields it calls absent/abstained, so only reasons the validators ADDED (absent from `parsed`) must be vocabulary codes. `validate_document` lives in `validate/__init__.py` with the `confidence` imports inside the function, so importing either package first works (its own test).
**Tech debt.** Caps 0.5/0.7/0.85 and `ABSTAIN_BELOW=0.3` are uncalibrated placeholders (S1); the checks lean Indonesian (`convention_for`'s 23 codes, NPWP 15/16 digits); `validate_fields`'s `today` is the wall clock; self-consistency cannot catch a quote and a value invented together (S4's probe set measures that).
**Next.** M2's human steps (classify, redact, split, label), then M5; fix the extractor model (3.6-flash vs 3.5-flash) before M5 so scorecards compare like with like.

## M5 - Eval harness: comparators, row matching, taxonomy, scorecard (2026-10-07)
**Outcome.** All 8 spec criteria pass; `pytest -m "slow or not slow"` 2336 passed, 1 skipped; `ruff check` clean. Criterion 8 on M4's live cons run (offline): 3 documents, 39/39 fields correct, 135 rows matched, 0 missed/spurious, 0 errors, `score/` written. 32 mutants over the five modules and the CLI: all caught except one shown equivalent (the padding cost is never decisive).
**New interfaces.** `evalkit.comparators` (`FIELD_KIND`, `CELL_KIND`, `Comparison`, `compare_field`/`compare_cell`, `normalize_text`, `FUZZY_NAME_MIN`, `NUMERIC_REL_TOL`/`NUMERIC_ABS_TOL`); `evalkit.matching` (`match_rows`, `RowMatch`, `RowPair`, `assign_min_cost`, `compare_rows`, `cell_verdict`, `row_cost`); `evalkit.taxonomy` (`ERROR_CLASSES`, `FieldOutcome`, `classify_field`, `count_errors`); `evalkit.metrics` (`score_document`/`score_field`/`score_rows`, `FieldMetrics`, `RowsScore`, `DocumentScore`); `evalkit.scorecard` (`score_run`, `Scorecard`, the hash helpers) and `scripts/score.py` -> `<run_dir>/score/{manifest,scorecard}.json|md`.
**Decisions.** Scoring is "both numbers": abstentions and missing predictions are errors in the headline `accuracy`, and `selective_accuracy`/`coverage` report them apart. Only `validated` is scored. The run manifest is written at score time (git SHA + dirty, prompt/schema hashes, model, config, GT hash, dataset hash) and `run.json` is untouched.
**Deviations.** Two mechanisms the spec named are inert and were simplified away: the tie-break sum is permutation-invariant and the padding cost is never decisive, so ties rest on the algorithm's fixed scan order (pinned by a test) and a fully-disjoint pair is unwound into one missed + one spurious row. `cell_mismatch` left `ERROR_CLASSES` (11, not 12): row cells reuse the field vocabulary (numeric/date/string/name/hallucinated/missing), so no class is dead. The DoD run covers cons_001..003 only (the labeled set).
**Tech debt.** Only three labels exist, so the scorecard is structurally complete but statistically thin until M2's labeling lands; the comparator thresholds are documented defaults, not calibrated (recorded per scorecard); difflib is weaker than a token-set similarity on company-name noise; scorecard artifacts sit under the gitignored `outputs/`, so scores are reproducible, not committed.
**Next.** M2's human steps (classify, redact, split, label), then M6 (diff two runs, reusing `ERROR_CLASSES`).

## M6 - Regression comparison (2026-10-08)
**Outcome.** All 7 spec criteria pass (6 offline + the live one); `pytest -m "slow or not slow"` 2444 passed, 1 skipped; `ruff check` clean. The first mutation pass (36 mutants) left 10 alive (bootstrap upper bound, metrics over unpaired documents, page hashes with no pages, endpoint/schema in "differs", blank slice values, flip-list caps): closed, then all killed.
**New interfaces.** `evalkit.compare` (`mcnemar_exact`, `paired_bootstrap`, `compare_runs`, `Thresholds`/`decide`, `Comparison.to_json/to_markdown/to_text`), `scripts/compare.py` (exit 0 ok / 1 regression / 2 cannot compare) -> `outputs/compare/<b>__vs__<c>/`; `extract.py --system-prompt-file` + `prompt_sha256()` (the run records the hash it used; defaults and cache keys unchanged); `docs/demo/degraded_prompt.txt` (frozen, hash pinned).
**Live demo (criterion 7, `docs/demo/regression_demo.md`).** Degraded prompt on cons_001..003: REGRESSION, 11 regressions / 0 improvements, p_worse 0.0005, accuracy 1.000 -> 0.718, "differs" = prompt hash only, byte-identical on rerun. Cost $0.00 on the free tier (4 live calls).
**Decisions.** Gate = exact one-sided McNemar p <= 0.05 AND drop >= 2 pts (exact fractions); bootstrap by label owner is reported, not gated; scalar fields flip by flip, line items in aggregate (decided with you).
**Deviations / findings.** The first compare used `m4-live` as baseline and was confounded: `.env` had moved to gemini-3.5-flash and lost the cost cap; "differs" showed it, so the baseline was re-run (`m6-baseline`, 1 live call + 2 cached). The spec said E2 (amounts /1000) would pass every internal validator; wrong: grounding abstained on all 135 rows and the money fields, so accuracy fell 28 pts but selective accuracy only 3. E3 (never-guess removed) caused no hallucination on this model. Commit 3678560 was made with one failing test (a masked exit code), fixed in 9bec1e4. `ruff format tests` once reformatted four unrelated tests; reverted.
**Tech debt.** No per-row flips (M5 keeps no per-GT-row verdict); `compare.py` is ~900 lines (report code could move out); only 3 independent documents, so the demo shows the gate catches a blunt change, not its sensitivity; M5's "ratio over nothing = 1.0" shows as n/a only inside compare.
**Next.** Your M2 steps, dev first (classify all 31 -> assign_split -> dev docs: redact, ingest, prelabel, verify), then repeat the demo on `--split dev`; test/blind labels before the final run; then M7.

## Interlude - M2 dev-first progress and the pre-labeler fix (2026-10-08)
**Done.** Manifest filled from `M2_manifest_guide.xlsx` (35 real rows, `content_group` = own doc_id except `gt_from` copies, only `doc_017` keeps a page_range), `assign_split` applied (34 groups -> 12 dev / 22 test, 10 blind, 31 hand labels: 11 dev + 20 test), the 12 dev rows ingested (0 redactions needed). Manifest and `data/pages/` are left uncommitted for the user to review (raw file names).
**Found by one live call on cons_001 (never tried before, only a fake server).** Anthropic's OpenAI-compatible endpoint now rejects `response_format` json_object (even for Haiku 4.5), wants strict json_schema (closed objects, no regex patterns, at most 16 nullable fields; the draft schema has 19), and Claude 5.x rejects `temperature`. Schema-in-prompt only works: Haiku 4.5 13/13 fields + 45/45 rows exact ($0.0196, 21.6 s), Haiku 5.5 the same ($0.002, 8.9 s).
**Fix.** `LLM_STRUCTURED_MODE=prompt` (no response_format, schema in the text; now the pre-labeler's default) and `LLM_SEND_TEMPERATURE` / `PRELABEL_SEND_TEMPERATURE`; mutation-checked (9 mutants), 2451 tests pass. The user's `.env` needs `PRELABEL_SEND_TEMPERATURE=false` for Haiku 5.5.
**Sizing.** Exact-binomial power of the gate: ~130 labeled fields (10 docs) catch a 5-pt drop 78% of the time, ~400 (30 docs) a 3-pt drop 99% and a 2-pt drop 90%; accuracy CIs reach +-3 pts only near 100 docs, so template diversity matters more than count.
**Next.** Draft the 11 dev papers (`prelabel.py`, ~3 cents, cap with `PRELABEL_MAX_RUN_COST_USD`), verify them in the label app, then repeat the M6 demo on `--split dev`; test and blind labels before the final run; then M7.

## M7 - API, batching, cost and latency (2026-10-08)
**Outcome.** All 10 spec criteria pass (9 offline + the live API demo, $0 by cache replay); `pytest -m "slow or not slow"` 2600 passed, 1 skipped; `ruff check` clean. Mutation pass: 39 mutants, 6 survived the first time (cap flag on the last document, exact-size and exact-batch boundaries) and were closed; 3 more for the PDFium lock.
**New interfaces.** `uvicorn --factory docextract.api.app:create_app`: GET /health, POST /extract (one document from uploaded files), POST /extract/batch (one file per document, 200 with per-document status); answers are `DocRecord`s built by the runner's own `extract_record`; `X-Request-Id`, optional bearer `API_TOKEN`, `API_MAX_BATCH` / `API_MAX_UPLOAD_MB`. `extract/concurrent.run_bounded` and `extract.py --concurrency N` (default 1; `RunRecord.concurrency`). `calllog.CallLog` (`CALL_LOG_PATH`, one JSON line per request, no names or content). `api/uploads.pages_from_uploads` (byte-identical to ingested pages). `evalkit/perf` and a "Cost and latency" section in scorecard.md/json plus a `score.py` line (m6-degraded: p50 43.2 s / p95 48.3 s over 3 live calls; m4-live: $0.0380/doc at list price, no live calls).
**Found by the live demo.** One PDF of a concurrent batch rendered to other pixels than alone (cache miss, one real call): PDFium is not thread-safe, and offline 3 threads crashed the interpreter (access violation). Fixed by one process-wide lock around every PDFium call in the upload path (`98f5221`) with a lock-held test and a subprocess stress test. The earlier tests used images and blank PDFs only.
**Decisions (with you).** Uploads, one file per document; synchronous batch; `--concurrency` for the runner. Latency percentiles over live calls only, cost shown paid and at list price; `MAX_RUN_COST_USD` is a server-lifetime budget and in-flight calls finish after it is crossed.
**Deviations.** Backoff already existed (proved through the API, no new code); the per-call log is one line per document request, not per HTTP attempt.
**Tech debt.** No global request-body limit (uploads are spooled before the size check); no async jobs or by-reference batch; PDF rendering is serialized; `compare` shows no cost/latency deltas; `compare.py` is ~900 lines; the first demo attempt used one provider request by accident.
**Next.** Verify the 11 dev drafts in the label app, repeat the M6 demo on `--split dev`, then M8 (paid key) and M9 (Docker is installed).

## Interlude - three dev documents left out (2026-10-08)
`doc_004` (two invoices on one page, so two of every field), its copy `doc_017`, and `doc_010` (a 20-page bundle with many companies, not one document) cannot be labeled under one-document-per-label, so they were removed from the manifest with their stored pages and drafts (decided with the user; CHANGELOG lines and a new "One document per input" rule in `ground_truth/LABELING.md`). The split of every other document is unchanged (`assign_split --dry-run` reports no change). Dev is now 10 documents, all labeled: 9 by hand (Haiku 5.5 drafts, each verified) and 1 constructed. The test split still needs its 20 labels (10 blind). `LLM_MAX_PAGES` can go back to 15 (`doc_010` was the only longer document).

## M8 - one optimization with measured impact: resolution and thinking-budget sweep (2026-10-08)
**Outcome.** All 6 criteria pass (5 offline + the live sweep, about $0.21 in total with the baseline); `pytest -m "slow or not slow"` 2653 passed, 1 skipped; `ruff check` clean. Mutation pass: 28 mutants over the rule, the scorecard section and both CLIs, all killed after 3 tests were added (equal cost is not "cheaper", the 10-document boundary, a resumed point's failure still sets exit 1). Result and caveats: `docs/demo/m8_sweep.md`.
**New interfaces.** `evalkit/sweep.py` (`build_sweep`, `choose`, `before_after`, markdown/JSON report), `scripts/sweep.py` (`--baseline-run`, `--resolution`, `--effort`, `--confirm`, `--dry-run`, `--resume`; exit 0/1/2/3), `score.py RUN --baseline BASE` adds a "Before / after" section to the scorecard.
**Result.** The rule, fixed in code before any result, chose `effort-low`: accuracy 0.811 -> 0.803 (-0.8 pts, p_worse 0.50), cost per doc -23%, p50 -21%, p95 -41%. All accuracy deltas are inside the noise (intervals about +-8 pts on 122 fields).
**Deviations.** The extractor changed twice during the milestone (Gemini free tier exhausted -> GLM 5.3 Flash -> DeepSeek V4 Flash Vision exp, 122 s -> 16.6 s p50 at the same accuracy on dev), so the sweep baseline is a fresh DeepSeek run, not the GLM `dev-baseline`. "Token budget" is reasoning effort (an output cap breaks JSON). The spec's criterion 5 counts 6 runs x 10 documents; the baseline already existed, so the plan is 5 runs = 50 requests.
**Findings.** The rule covers scalar fields only; the chosen run's row F1 fell 0.792 -> 0.684 (other points rose), so line-item quality needs its own gate or a measured noise floor. `effort-off` (2.4x cheaper, 2.5x faster) missed the 2-point tolerance by half a point; the tolerance was not moved afterwards. `doc_022` is a shared cache replay in three runs.
**Tech debt.** No noise floor (one sample per point; a `--cache refresh` repeat of the baseline would give it); prices come from a third-party tracker; DeepSeek's `reasoning_effort` values were only seen to work, not documented; `.env` prices must be non-zero for the sweep.
**Next.** Decide the extractor and effort setting (your call), then calibration (S1) and the hallucination probe (S4) on it; M9 (Docker, committed run artifacts, replay); M10; recreate `.env.example` and scan the history before any push; label the test split.

## Interlude - DeepSeek effort decision, item-code prompt, M6 on dev (2026-10-08)
**Done.** `medium` verified on dev (`dev-medium`: accepted, 10 ok, no gain over the default: 0.803 vs 0.811, 8 regressed / 7 improved). Prompt: one sentence keeps an item-code column out of the line-item description (`dd715f1`, hash `b154b4cc8ac0` -> `7ec75dbb2906`); it removed the `cons_003` code-in-description flip (33 `name_near_miss`) and `dev-final` scores 0.811 (selective 0.917, coverage 0.885, row accuracy 0.873, $0.0040/doc). The M6 gate was repeated on dev against the frozen degraded prompt: REGRESSION, -9.8 pts, p_worse 0.0113 (`docs/demo/regression_demo.md`).
**Decision (you).** `medium` for the rest of the project although the sweep rule picked `low`; recorded in `docs/demo/m8_sweep.md`.
**Withdrawn.** I read `dev-medium` as "medium thinks 37% more than the default"; `dev-final` (same setting) used 41% fewer tokens than `dev-medium`, so that was run-to-run noise. Latency p95 from one 10-document run (one slow document: 33-106 s) must not be quoted.
**Open.** `.env` prices were still 0.22/0.66 when checked (Command Code lists 0.15/0.60 off-peak, 0.30/1.20 peak). Next: S1 calibration on `dev-final`, S4 hallucination probe, M9, M10, test-split labels.

## S1 - failure analysis, run-to-run noise and calibration (2026-10-08)
**Outcome.** Criteria 1-7 of `docs/milestones/S1.md` pass; `pytest -m "slow or not slow"` 2734 passed (81 new), `ruff check` clean. Mutation pass: 34 mutants over the three modules, 9 survived the first time (the `should_be_value` case, a cached call, ECE with unequal bins, a point exactly at the target...), all killed after 9 more tests. A test of the new SVG axis also caught a real edge case (the in-force marker off the plot).
**New interfaces.** `evalkit/failures.py` (`collect`, `FieldCase`: each labeled field scored as returned AND as the model said it), `noise.py` (`measure`), `calibration.py` (`analyse`, rule: lowest threshold with selective accuracy >= 0.95, hand-written SVGs); `scripts/analyze_failures.py`, `noise_report.py`, `calibrate.py` (`--notes` appends a hand-written "Reading"); reports in `docs/eval/`.
**Findings.** (1) Of 14 abstentions on dev-final, 13 are `evidence_not_grounded` and 11 withheld a CORRECT value: the quote prints an ambiguous number (`208,179`, `84.000`) and the grounding rule treats "not confirmed" as "contradicted". The same code abstains 33 of 81 rows; the model's own rows pair 78/81 with 0 spurious (as returned: 55/81, 23 spurious). Kept, dev would be 110/122 (90.2%) not 99/122 (81.1%). (2) Four identical-setting runs: accuracy 0.795-0.828, 2-12 of 122 fields flip between two runs, output tokens +39%, p95 33-78 s: the sweep's 2-point tolerance is inside the noise. (3) The model's own confidence is better calibrated than the capped one (ECE 0.071 vs 0.116); `ABSTAIN_BELOW` 0.3 never fires; the rule's 0.92 would cut coverage to 0.57, not recommended.
**Deviations.** The spec says 13 abstentions; it is 14 (13 on a labeled value, 1 where the label says absent); the 11 holds. The script tests use hand-built run folders, not the fake server (no model call is involved). The noise report recomputes dollars at one price because `.env` prices changed while the runs were made. The threshold is a dev fit; reporting it on test waits for test labels.
**Not done, needs a decision.** The grounding fix (options in `docs/eval/failure_analysis.md`): `validate/` was out of scope, so nothing about validators, caps or the threshold changed.
**Tech debt.** The SVGs are only structurally tested, not viewed; reliability bins are coarse on 10 documents.
**Next.** The grounding decision; S4 hallucination probe (own spec); M9; M10; the 20 test labels (yours).

## L1 - labeling app: paste rows from a spreadsheet, and the grid that lost the first edit (2026-10-08)
**Why.** The owner was labeling the 10 blind test documents by hand (doc_019 has 20+ rows) and reported that a value typed once into the line-item grid vanished and had to be typed again.
**Done.** `labeling.parse_pasted_rows` + `LabelForm.paste_rows` and a "Paste rows from a spreadsheet" expander (replace / add after the last row; a bad paste changes nothing and says why); `labeling_ui.frozen_input` so the grid is given the same input on every rerun until `rows_version` moves; a "Long tables" rule in `ground_truth/LABELING.md` (label every row or mark the table `unlabeled`, never part). 36 new tests; the labeling suites pass (234 with the slow app tests), `ruff` clean. Mutation pass: 22 mutants, 1 survived (the replace button behaving as add), killed by one more test.
**Cause of the lost edit (found by reading, not reproduced).** `rows_widget` rebuilt the `st.data_editor` input from `form.rows` on every run while `form.rows` is rewritten from the editor's own output, so the editor's stored deltas were applied to a different table. Headless `AppTest` cannot type into a data editor, so the fix is tested at its seam (the input is frozen per version, and the app holds it in session state); whether the browser behaviour is fixed is for the owner to confirm.
**Found along the way.** The app tests are marked `slow`, so a plain `pytest -q` had not been running them; and a mutation pass edits the very files a running Streamlit app imports, so it was run on an isolated copy of the repo.
**Not done.** No label, draft or saved work was touched.

## E1 - secret scan hook, mutation-check tool and skill, `.env.example`, CI (2026-10-08)
**Outcome.** Spec criteria 1-5 pass; `pytest -m "slow or not slow"` and `ruff check` green (counts below). `tools/secret_scan.py` (stdlib only: `--staged`, `--worktree`, `--history`, `--env-values`; the report never holds a value), the Claude Code hook `.claude/hooks/pre_commit_secret_scan.py` + `.claude/settings.json` (blocks a `git commit` that would add a secret; widens to the whole working tree when the same command also stages), `tools/mutation_check.py` + `.claude/skills/mutation-check` (mutants on an isolated copy, green baseline first, ambiguous mutant refused), a new `.env.example` (placeholders only, every setting the code reads, enforced by a test), `.github/workflows/ci.yml` (windows-latest: secret scan, ruff, the full suite).
**Scans of this repository.** `--history` over 124 commits (about 57,000 diff lines): 0 findings. `--env-values`: the 2 real secrets in `.env` (LLM_API_KEY, PRELABEL_API_KEY; names only are printed) appear in no commit. `--worktree`: clean.
**Found by the mutation pass (23 mutants over the scanner and hook; 6 survived the first time, all killed now).** A real performance bug: the assignment regex was quadratic in line length, so one 100 KB line stalled the scan (the pass itself hung for 30 minutes on the mutant that removed the size guard). Fixed by looking only in a window around each keyword: a 2 MB line now takes 0.2 s. Also dead code deleted (an `.env.example` exclusion that could never matter, a `/dev/null` special case), and tests added for repeated-character and `$VAR` values, digits-only values, `git.exe`. A vacuous `... or True` I wrote hid a hook crash on a non-object payload; fixed.
**Deviations.** The hook runs BEFORE the command, so it cannot see what `git add x && git commit` will stage: it scans the working tree in that case (a stricter, occasionally over-eager scan, chosen over missing the common form). Hooks load at session start: they are active from the next session or after `/hooks`.
**Not run.** CI has not run on GitHub; the suite has only ever run on Windows (hence `windows-latest`). The offline re-score of committed predictions joins CI with M9.
**Tech debt.** The word list for placeholders is a judgement (`here`, `sample`, `paste`...); a real key containing one would be skipped by the pattern rule (not by `--env-values`).
**Next.** Your call on the grounding fix; S4 probe; M9; M10; the frozen final run on the test split.

## M9 - Docker, eval through the API, committed results (2026-10-09)
**Outcome.** Spec criteria 1-8 pass (5 with a fake model inside the compose network, not on the host); the live run (6) is committed as `results/test-final`: 25 test documents through `docker compose run --rm eval`, 25 ok, accuracy 0.793 (selective 0.859, coverage 0.923), $0.1992, p50 12.5 s; `check_results.py` reproduces it byte for byte. `pytest -m "slow or not slow"` 2941 passed + 1 pre-existing flake (`test_mutation_check`, green alone and when rerun; looks like a stale-`.pyc` race), `ruff check` clean. Mutation passes: 23 + 14 + 19 + 9 mutants (client, promote, Docker files, check_results), each first pass left 1-3 alive (spend as a lifetime total, a dead `or message` line, the git/time reuse), all closed.
**New interfaces.** `Dockerfile`/`docker-compose.yml`/`.dockerignore` (allow-list; `api` with env_file + healthcheck, `eval` with profile `eval`, no key); `scripts/eval_api.py` (eval through `POST /extract`, same run folder and exit codes as `extract.py`, no selection = test split); `GET /health` also reports endpoint, prompt/schema hashes and result settings; `scripts/promote_run.py` -> `results/<run_id>/`; `scripts/check_results.py` (CI step) with LF checkouts forced in `ci.yml`.
**Found.** (1) Arguments to `docker compose run` REPLACE a compose `command`, so `run --rm eval --run-id X` lost `--split test`; the default moved into the script (found by running the container, not by the tests). (2) Uploading a stored JPEG re-encodes it (q92 generation loss, another cache key): 11 of 25 test documents are photos, so the API run does not send the same pixels as `extract.py`; PNG pages are byte-identical. CLAUDE.md's old "byte-identical" sentence was wrong for JPEG and is corrected. (3) The score hashes label and manifest BYTES, so a CRLF checkout would not reproduce a committed scorecard; the CI job sets `core.autocrlf false` (a Windows clone with autocrlf will see different hashes in `check_results.py`).
**Deviations.** `scripts/check_results.py` and its test/mutants are new files beyond the spec's list (the CI step needed a script). Criterion 5 ran the fake model as a container in the compose network (the fake binds 127.0.0.1, unreachable from a container), via a throwaway override file in the scratchpad, not committed. Criterion 4's "one character edited" holds for edits that move a score; an edit of `raw_text` alone changes nothing (git covers that). Docker needed network (an earlier build attempt failed on Docker Desktop DNS, retried).
**Tech debt.** CI has still never run on GitHub, and no Docker job (the runner is Windows); the image is 501 MB; a bind-mounted `./outputs` may need permissions on a Linux host (uid 10001); `env_file: .env` hands the pre-labeler key to the API container unused; the server's cost cap is process-lifetime (`docker compose restart api` resets it); the committed run is one noisy sample with the grounding flaw unfixed (23 of its 67 field errors are withheld values).
**Next.** Your call on the grounding fix, then a second run compared with `compare.py` against `results/test-final`; S4 probe; M10 (README, architecture, agentic note); final history scan before any push.

## G1 - grounding fix: an ambiguous number that contains the value is soft (2026-10-09)
**Outcome.** All 8 spec criteria pass; `pytest -m "slow or not slow"` 2954 passed + the known `test_mutation_check` flake (see below), `ruff check` clean. Mutation pass: 15 mutants, 2 survived the first time (a candidate test that was not exact decimal equality, a non-number money value), both killed by new tests.
**What changed.** `validate/grounding.py`: number grounding is three-valued; a money value (scalar or row cell) whose quote is an ambiguous number containing the value gets the soft code `evidence_ambiguous` (new in `validate/codes.py`, cap `SHAPE_CAP` 0.7 in `confidence/formula.py`); no candidate matches -> still `evidence_not_grounded` (hard); dates untouched. The value is never rewritten.
**Result (pre-registered on dev, replayed on test).** The test split replayed from the cache through the Docker API: all 25 calls cached, $0, `parsed`/`raw_text` byte-identical to `test-final`. Accuracy 0.793 -> 0.845 (256 -> 273 of 323), exactly the 17 predicted fields, 0 regressions (McNemar p_better < 0.0001, bootstrap +5.3 pts [+1.8, +9.9]); coverage 0.923 -> 0.975, selective accuracy 0.859 -> 0.867; 21 of 34 withheld rows restored (only 8 move the row score). `results/test-g1` committed; `docs/eval/grounding_fix.md` has the numbers and the reading.
**Found.** Restored rows are now compared cell by cell, so `hallucinated_value` 24 -> 27, `name_near_miss` 9 -> 12, `missing_value` 10 -> 12: errors that were hidden inside abstained rows, not new mistakes. And the `test_mutation_check` flake is a real bug in `tools/mutation_check.py`: a mutant left a `.pyc` whose mtime (seconds) and size match the next mutant's source, so the next mutant can run the PREVIOUS mutation and be reported killed (a false kill). Not fixed: `tools/` was outside this spec.
**Deviations.** None from the spec. Dev was not re-extracted (its cache holds a later `--cache refresh` sample, so it would not be paired).
**Tech debt.** The 0.7 cap is a placeholder; `ABSTAIN_BELOW` (0.3) still never fires; ambiguous dates and the `Jl.`/`J1.` OCR slip are still hard; a model that picks the wrong candidate of an ambiguous quote is kept at 0.7 (none seen).
**Next.** A1 (breakdown of the `hallucinated_value` errors, on `test-g1`), then S4 (hallucination probe).

## A1 - what the `hallucinated_value` errors are, and an audit of copies that share a label (2026-10-09)
**Outcome.** Spec criteria 1-6 pass; `tests/test_hallucinations.py` 27 tests; mutation pass 24 mutants (1 refused for a wrong text, fixed, then all killed); `ruff check` clean. The verdicts (wrong label or invented values) are NOT in: they need the owner to look at the pages.
**New interfaces.** `evalkit/hallucinations.py` (`collect`, `Hallucination`, `lineage` audit with `is_suspect`, `to_markdown`, `review_markdown`), `scripts/analyze_hallucinations.py RUN [--out] [--review] [--notes]` (the review list holds quotes, so the script refuses a path inside docs/ results/ ground_truth/ data/ src/ scripts/ tests/); `docs/eval/hallucinations.md` (+ `.notes.md`).
**Found.** `test-g1`: 27 = 12 scalar (12 of 116 labeled-absent fields, 10.3%) + 15 row cells (15 of 49 null cells); `test-final` 12 + 12. 11 of 12 scalar cases come from doc_016's blind label and the three phone photos that borrow it (`gt_from`): the label describes an electricity-token bill (one row, total 503100) but the run reads doc_032/033/034 as three different perfume-shop receipts (4 distinct totals, 4 distinct dates). The audit flags all three (1/5 right vs 4/5 for the owner). Sensitivity, not a result: without them accuracy is 257/284 = 0.905 (vs 0.845) and the scalar hallucination rate 2/92 = 2.2% (vs 10.3%). The 15 cells are mostly fees, vouchers, discounts or an amount-0 item returned as line items (the model's own notes say so): a row-definition question. 9 of 27 have confidence >= 0.90; only 3 carry a validator code; all 27 are in the model's raw answer.
**Deviations.** None. I did not view a page; the reading is from the run's structure and the model's own notes, and says so.
**Tech debt.** The audit is a heuristic on 5 fields; the "copies read alike" check covers only `document_number`, `document_date` and `total`.
**Next.** The owner's review of `outputs/hallucination_review.md` (doc_016 family first), then the relabel decision if (A) holds; S4.

## S4 - hallucination probe: fields erased from constructed invoices (2026-10-09)
**Outcome.** Spec criteria 1-7 pass; `tests/test_probes.py` 34 tests (one `slow`: the committed `data/probes` equals a regeneration); mutation pass 27 mutants, 2 survived the first time (a control that misread the field was never tested; a redundant `not in erased` clause: deleted, not tested), all killed. Live run `probe-s4`: 12 documents, $0.108 (c03 a cache hit).
**Result.** 0 of 19 erased fields were answered with a value (15 `absent`, 4 `abstained` on the classic invoice with its last page dropped); exact 95% interval 0.0%-17.6%. The model alone gave the same answers, so the validators are not credited. Not at the price of answering nothing: 84 of 84 non-erased fields right, controls 13/13, 9/9, 13/13, and the dropped-page probes returned exactly the remaining rows (30/30, 24/24, none invented). The spec's prediction (p08, the derivable totals, would be highest) was wrong: the model left the three blank amounts `absent` and noted the blanks instead of summing 57 rows.
**New interfaces.** `evalkit/probes.py` (`PROBES`, `build_probe`, `erase_box`, `write_probe_set`, `check_sources`), `evalkit/probe_report.py` (`analyse`, `to_markdown`, `exact_interval`), `scripts/make_probes.py [--check] [--force]`, `scripts/probe_report.py`; `data/probes/` (manifest, `probes.json`, labels, pages: a mini-dataset that `extract.py` and `score.py` take through `--manifest/--pages-dir/--ground-truth-dir`); `docs/eval/hallucination_probe.md`; `results/probes/probe-s4`.
**Found.** The probe's strongest guard is structural: `make_probes.py` refuses a probe whose controls or kept pages are not byte-identical to `data/pages`, and every erase box must have held ink and end one flat colour (I also looked at the synthetic pages). The real test-split hallucinations are not this kind: they sit on blurred photos and a possibly wrong label (A1), which this clean-page probe cannot speak to.
**Deviations.** The report is a second module (`probe_report.py`) rather than part of `probes.py`. `results/probes/probe-s4` sits one level down because its manifest is not `data/manifest.csv`, so `check_results.py` does not cover it (its re-score was checked by hand: the scorecard reproduces apart from git and time).
**Tech debt.** 19 cases is a small sample (upper bound 17.6%); a smudge probe (value blurred instead of painted out) and real-document probes are the follow-up; PNG bytes of the probe pages depend on the PDFium and Pillow versions (a regeneration on another platform may differ, which `--check` would report).
**Next.** The owner's review of `outputs/hallucination_review.md` (the doc_016 family); the relabel decision; a smudge probe; M10.

## M10 - README, `.env.example`, repository hygiene pass (2026-10-09)
**Outcome.** Spec criteria 1-7 pass. `README.md` has every section the brief lists; each figure was checked against the committed artifact it cites, all 42 links and every backticked repo path resolve, and the offline commands in it were run (`compare.py` exit 0, `analyze_failures.py`, `check_results.py`, `docker compose config`). `.env.example` gained the Docker notes (it is the compose `env_file`; the cost cap lives as long as the server process; `host.docker.internal` for a host-side model) and the measured test numbers; `tests/test_ci_and_env_example.py` passes.
**Hygiene pass.** Clean: no `.env`, cache, venv, `outputs/`, raw document, preview, spreadsheet or PDF tracked; no local path, email or TODO in tracked text; history and `.env` values scanned 0 findings; the brief's PDF is not in the repo; no file over 2 MB. A clean clone (Windows-style checkout) ran `ruff check` (clean), the full suite (3015 passed, 2 failed), `check_results.py` (both results reproduce) and `make_probes.py --check` (identical). The 2 failures: `test_mutation_check` (the known flake: a suspected stale-bytecode race, passes alone) and one Augraphy test that failed only because my scratch clone's path was 297 characters (Numba's cache file past 260); from a short path all 103 degrade tests pass.
**Found and fixed.** A default Windows checkout (`core.autocrlf=true`) did NOT reproduce the committed scorecards (the scorecard hashes label and manifest bytes, so CRLF changes them): `.gitattributes` forces LF (`* text=auto eol=lf`, binaries marked); after it the same clone reproduces both results. Noted in CLAUDE.md.
**Found, not fixed (reported to the owner).** (1) 12 downloaded documents have no `source_url` in the manifest (only "internet downloaded"): provenance is a gap a reviewer may ask about. (2) `tools/mutation_check.py` can report a false kill (stale `.pyc` with the same mtime and size): `PYTHONDONTWRITEBYTECODE=1` for the test subprocess would stop it; outside every spec so far. (3) 6 files are not `ruff format`-clean (`scripts/check_run.py`, `tests/test_api.py`, `tests/test_validate_fields.py`, `tests/test_check_run_script.py`, `tests/test_imports.py`, `.claude/skills/mutation-check/SKILL.md`); CI runs `ruff check` only. (4) CI has not run on GitHub and has no Docker job. (5) The three test documents that may carry the wrong label (doc_032..034) are stated in the README as open.
**Deviations.** `.gitattributes` was committed before the M10 spec (the spec lists it; the order was mine). The README is written in the owner's voice ("I"); the claims about what the owner did (labels, redaction, decisions) come from the log and the memory of the project, and should be read by the owner before the push.
**Next.** The owner reads the README (especially "Agentic development" and the dataset provenance lines), decides the doc_032..034 question, and pushes.

## Fix - `tools/mutation_check.py` false kill from stale bytecode (2026-10-09)
**Why.** `test_mutation_check` failed in every full-suite run since M9 and would have made the first GitHub CI run red. The cause: a mutant of the same size as the one before it, written in the same second, was served from the `.pyc` that earlier run left in the work tree (Python trusts a `.pyc` whose source mtime in whole seconds and size match), so it ran the PREVIOUS mutation and was reported killed: a false kill, which could also hide a surviving mutant.
**What.** The test subprocess gets `PYTHONDONTWRITEBYTECODE=1` (one line, with the reason beside it). Two deterministic tests were written first and failed without it: a same-size edit with the same mtime must not be served from old bytecode, and no `.pyc` is left in the work tree. `tests/test_mutation_check.py` passes 7 runs in a row (it flaked in about every full run before); a real mutation pass (`tests/mutants/probes.py`, 27 mutants) still ends with no survivors.
**Scope.** Outside every milestone spec; done at the owner's request (2026-10-09). The README no longer lists it as open.

## Relabel of doc_032..034, `test-g2`, and the milestone status in the README (2026-10-09)
**What happened.** The A1 audit had flagged doc_032, 033 and 034 (scored against doc_016's blind label). The owner looked at the pages: 032 and 034 are two photos of doc_019's order (a buyer name is printed that doc_019's label lacks), 033 is a cafe drink receipt. Manifest rows changed (doc_type, template_group, content_group, gt_from, label_mode; `check_manifest` clean, `assign_split --dry-run`: 0 changes), drafts by claude-haiku-5-5 for 032 and 033 ($0.0016), both verified by hand in the label app; doc_034 takes doc_032's label. All logged in `ground_truth/CHANGELOG.md`; not blind (the extractor's output had been seen).
**Result.** `results/test-g2` (the same raw answers replayed from the cache, $0, labels corrected, grounding fix): accuracy 0.898 (290/323), selective 0.921, coverage 0.975, rows F1 0.841. The unfixed run re-scored on the corrected labels: 0.845. So the two effects are separate and about equal: the label correction +5.2 points (0.793 -> 0.845), the grounding fix +5.3 (0.845 -> 0.898, the same 17 fields, 0 regressed). Scalar hallucinations 12 -> 2 (1.8% of labeled-absent fields). `hallucinations.md` and `grounding_fix.md` updated.
**Results folder.** `test-final` and `test-g1` moved to `results/superseded/` unedited (their scorecards are on the old labels, so they cannot reproduce against the current ones and `check_results.py` skips that folder); `test-g2` is promoted and reproduces.
**README.** New section "Milestones: what was reached and what was not", checked against `docs/ROADMAP.md` (there is no RUNBOOK.md): M2 partly (35 documents, not 40; 10 blind), S1 partly, S2/S3/S5/S7 and N1/N2/N4/N5/N6 not done, S6 and N3 partly, the rest reached, each with its reason.
**Next.** Push (the repo has no remote yet), watch the first CI run.

## First push and CI (2026-10-09)
The repository was pushed to a private GitHub repo and the CI workflow (secret scan over the history, `ruff check`, the full suite, `check_results.py`) **passed on the first run**, and again after the owner's one-line README edit on the website (pulled with a rebase before the next push). README (S6) and CLAUDE.md updated: CI is no longer listed as never run. No Docker job (the runner is Windows).

## P1 - make the repository public (2026-10-14)
**Outcome.** The 12 downloaded documents (11 faktur, 1 invoice; no source URL or licence recorded) and the 3 earlier-excluded ones (doc_004, 010, 017) are gone from the tree AND from all 151 commits (`git filter-branch`, authorship kept, SHAs changed; pre-rewrite bundle at `../document-extraction-pre-public.bundle`). 23 documents remain (20 own, 3 constructed; 7 dev, 16 test; 6 blind). `results/test-public` replaces `results/test-g2`: accuracy 0.899 (186/207), rows F1 0.823 overall but 0.532 on the 14 real documents (33 matched, 10 missed, 48 spurious; the 2 constructed invoices give 102 of 135 matched rows). Three original runs kept as aggregate scorecards in `results/original-dataset/`. MIT licence, README rewritten for the 23 documents with a "Known limitations" section.
**Found.** (1) The reported grounding gap is real: `_number_candidates` splits on whitespace, `x1`/`2x` never ground, 9 rows withheld in `test-public` (all 7 of doc_027, 2 of doc_020), the model had them right. (2) `data/redactions.yaml` is empty: no published document was ever redacted. (3) The Docker run hashed the PNG pages as other bytes than the stored ones (doc_012 `70181c..` vs `c0a073..`), so its cache entries cannot be replayed outside Docker; the README claim "a PNG comes back byte for byte" was false there.
**Deviations.** Criterion 4 (all 16 documents cached) was NOT met: Docker was not available, a first attempt under a $0.0001 cap made one live call ($0.0055, aborted), and with the owner's approval the run was live for doc_016 and doc_019 ($0.0238); 14 of 16 answers were cache hits. Total spent $0.029. The spec says the dropped set was "11 faktur plus one e-commerce bill"; it was 11 faktur plus one invoice (the spec is not edited). `.env.example` still cites `results/test-g1` and the 25-document figures: not in the spec's file list, left for the owner. docs/eval, docs/demo and older DEVLOG entries were not touched and describe the original 35 documents.
**Tech debt.** The analysis reports cannot be reproduced from this repository. `results/original-dataset/*` do not reproduce. Dev numbers (noise, calibration, sweep) rest on 10 dev documents of which 3 are unpublished.
**Next.** Fix the grounding gap (`x1`, `2x`), re-run, `compare.py test-public <new>`; the owner looks at the 20 unredacted pages and publishes through a NEW repository (see the hand-over).
