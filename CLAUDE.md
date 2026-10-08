# CLAUDE.md

Project rules for Claude Code. Read this before making changes.

A document extraction system for invoices/receipts plus a rigorous evaluation harness.

## Stack and tooling
- Python 3.11 (project venv in `.venv`, created by uv), `uv` for dependencies, `ruff` + `pytest`.
- Type hints everywhere (ruff `ANN` enforces it in `src/` and `scripts/`). Pydantic v2 for all schemas.

## Layout
```
src/docextract/{config.py, calllog.py, providers/, ingest/, extract/, validate/, confidence/, api/{app.py, uploads.py}}
src/evalkit/{comparators.py, matching.py, metrics.py, taxonomy.py, scorecard.py, perf.py, compare.py, sweep.py, failures.py, noise.py, calibration.py, hallucinations.py, probes.py, probe_report.py, groundtruth.py, gt_store.py, construct.py, construct_io.py, prelabel.py, printed.py, labeling.py, labeling_ui.py}
data/{manifest.csv, redactions.yaml, pages/, probes/, private/{raw,preview}}
ground_truth/{, drafts/}
outputs/
results/
scripts/
Dockerfile, docker-compose.yml, .dockerignore
tools/{label_app.py, secret_scan.py, mutation_check.py}
.claude/{settings.json, hooks/pre_commit_secret_scan.py, skills/mutation-check/SKILL.md}
.github/workflows/ci.yml
tests/{fakes/, fixtures/}
docs/
```
`groundtruth.py` (the ground-truth schema), `gt_store.py` (label files), `construct*.py` (constructed invoices) and `prelabel.py` (model drafts), `printed.py` (printed text -> label values, shared by the pre-labeler and the labeling app) and `labeling*.py` (the labeling app's logic) are the files in `evalkit` not in the original layout brief.

## Provider
- The provider is OpenAI-compatible and fully configured via env: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, plus per-token prices for cost (`LLM_PRICE_INPUT_PER_MTOK`, `LLM_PRICE_OUTPUT_PER_MTOK`, USD per 1M tokens).
- Pre-labeling has its own `PRELABEL_*` variables (below).
- **`.env.example`** was removed on 2026-10-08 after a real key was nearly pasted into it, and rewritten at the end of E1 with placeholders only. `tests/test_ci_and_env_example.py` keeps it true to the code (it must name every setting `Settings` reads, load as is, and pass the secret scanner), and the commit hook and CI scan (below) refuse a real value in it. Never edit `.env` itself; put new variables in the example.
- Optional env: `LLM_MAX_REPAIRS` (default 1: re-asks when output is not valid JSON / does not match the schema), `LLM_REASONING_EFFORT` (passed through as `reasoning_effort` when set), `LLM_STRUCTURED_MODE` (auto|json_schema|json_object|prompt: `prompt` sends no `response_format`, only the schema in the text), `LLM_SEND_TEMPERATURE` (false for models that reject `temperature`), `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `LLM_BACKOFF_BASE_S`, `LLM_BACKOFF_MAX_S`, `LLM_MAX_PAGES` (default 15: a document with more pages is refused before any request; all its pages go in one call), and `MAX_RUN_COST_USD` (below). `.env.example` lists them all.
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
- **Order in the pipeline: parse JSON -> normalize -> validate -> repair only if that still fails** (`chat(preprocess=...)`; the extractor's hook is `extraction_preprocess(page_count)`: the normalizer, then the evidence contract, so an extracted value without evidence or page, or a page that was not sent, is a violation that is repaired). A value the normalizer declined to resolve is an error even if pydantic could parse it (`Decimal("1.250")` parses fine, and accepting it would make the guess the normalizer refused to make). Every rewrite is reported in `LLMResponse.normalizations`.
- **Validators, confidence and abstention (`validate/`, `confidence/`).** `validate_document` (the extractor runs it on the RAW parsed output) chains row arithmetic -> document reconciliation (lines vs subtotal; subtotal +/- tax/discount vs total) -> field checks (dates, currency, tax id) -> evidence grounding -> `apply_confidence` -> `apply_abstention`; each step is idempotent and returns a copy. Every reason a validator adds is a code from `validate/codes.py` (`REASON_CODES`); a model's own free-text reason is left alone. Grounding is self-consistency: an extracted value must be traceable in its own quote (numbers/dates through the shared normalizer, strings case- and punctuation-insensitive; a reworded `description` and a number whose quote is ambiguous but contains the value are soft). Confidence is the model's score lowered by the caps its codes imply (`confidence/formula.py`: arithmetic 0.5, shape 0.7, a normalized value 0.85), never raised. An extracted field or row below `ABSTAIN_BELOW` is abstained (`confidence_below_threshold`); a HARD code (`HARD_CODES`: ungrounded evidence, a due date before the document date) abstains whatever the confidence. Caps and the threshold are still placeholders: S1 measured them (see "Failure analysis, noise and calibration") but changed nothing. `validate/__init__.py` imports `confidence` inside `validate_document` on purpose (either package may be imported first).

## Extraction runs (`extract/runner.py`, `scripts/extract.py`)
- **A BLIND document, or a variant of one (`gt_from`), is never extracted until its label is finished** (`ground_truth/<doc_id>.json` exists): it is refused with zero requests, because no model output may exist for it before a human has labeled it from the page alone.
- `--system-prompt-file PATH` sends another system prompt (UTF-8, universal newlines; a missing or blank file is exit 2 before any request). The run records the hash of the prompt it used (`prompt_sha256()`), so a changed prompt is a different run; the default prompt, its hash and the cache keys are unchanged.

## Pre-labeling (`evalkit/prelabel.py`, `scripts/prelabel.py`)
- A model drafts ground truth into `ground_truth/drafts/<doc_id>.json` (a label file with `label_mode: prelabel` and no `labeled_at`); a human verifies every field. It is configured by `PRELABEL_BASE_URL/API_KEY/MODEL` (+ prices, `PRELABEL_MAX_RUN_COST_USD`) and runs through the same `OpenAICompatClient` via `Settings.prelabel_settings()`: `prompt` mode pinned (no `response_format`, the schema is in the prompt: Anthropic's OpenAI-compatible endpoint rejects json_object and takes only strict json_schema; `PRELABEL_SEND_TEMPERATURE=false` for Claude 5.x, which rejects `temperature`), page images downscaled in memory to 1568 px and under 4.5 MB, the extractor's reasoning effort and cost cap not inherited.
- **Hard rules, enforced in code and tested to make zero requests:** a BLIND document is never drafted (`BlindDocumentError`; `save_draft` re-checks, `check_labels` flags a blind draft); only hand-labeled `label_mode=prelabel` documents qualify (not borrowers, constructed, or blank label_mode); the model must differ from the extractor's (same endpoint + model is refused, `--allow-same-model` is the escape hatch); only the STORED, REDACTED pages are sent (verified against `ingest.json`, raw files never read); a finished label is never drafted over.
- The model transcribes what is PRINTED as strings; `docextract.normalize` turns them into values. Anything it cannot resolve without guessing (ambiguous `1.250` with no currency, `05/03/2024`) becomes `"__unlabeled__"` with a field note. Null = not printed, a missing key = not answered (unlabeled).

## Labeling app (`tools/label_app.py`, logic in `evalkit/labeling.py` + `labeling_ui.py`)
- `python -m uv run streamlit run tools/label_app.py` (streamlit is in the default `tools` group). Left: the STORED (redacted) pages with the pixel under the cursor and a draggable box that prints `[x, y, w, h]` and a `redactions.yaml` snippet (an opt-in UNREDACTED preview toggle exists only for finding box coordinates). Right: the form.
- Every field starts PENDING and must be decided: `verified` (value as printed, normalized with the shared normalizer; an ambiguous or unreadable value is an inline error, never a guess), `absent` (null) or `unlabeled` (`"__unlabeled__"`). **Save is blocked until nothing is pending**, so a draft never becomes ground truth by being left alone. A prelabel document starts from the draft (still all pending); a BLIND document starts empty and `BlindDraftError` is raised if a draft exists for it.
- Saving writes `ground_truth/<doc_id>.json` with `label_mode` + `labeled_at` AND appends its own `CHANGELOG.md` line(s) (per changed field when replacing; replacing a finished label needs a reason; if the log cannot be written the label file is put back). An identical label is not rewritten.
- Work in progress is kept in `outputs/labeling/<doc_id>.json` (gitignored; written only on real edits, never for just looking) and nothing in it is trusted on reload: its shape is checked and every field it calls verified must verify again.
- Line items: the grid's input is frozen per `rows_version` (`labeling_ui.frozen_input`), because `st.data_editor` keeps edits as deltas against its first input and rebuilding that input from the edited rows loses the first edit. "Paste rows from a spreadsheet" (`labeling.parse_pasted_rows`: tab-separated, quoted cells, optional header, 4 columns, any problem refuses the whole paste) fills the grid; rows still go through `verify rows`. A long table is labeled in full or marked `unlabeled`, never in part (`ground_truth/LABELING.md`).
- The viewer's JavaScript geometry and event glue are tested under Node (against a Python reference and the redaction code); the app itself is tested headlessly with Streamlit's `AppTest`.

## Core logic
- Extraction logic and the eval harness are hand-written. No LangChain / LlamaIndex for core logic.

## Scoring (`evalkit/`, `scripts/score.py`)
- A run is scored offline against the labels: type-aware comparators, line items paired by a hand-written Hungarian assignment, every miss classified in `evalkit.taxonomy.ERROR_CLASSES`. The scorecard (markdown + JSON), the per-doc results and the run manifest (git SHA + dirty, prompt/schema hashes, model, config, GT hash, dataset hash) go to `<run_dir>/score/`.
- The scorecard also reports cost per document (paid, and at list price = paid + avoided) and latency p50/p95 overall and per slice (`evalkit/perf.py`): every attempted document counts (ok or failed, labeled or not), latency is over LIVE calls only (a cache replay is 0 ms), percentiles are nearest-rank.
- Only `validated` is scored, never `parsed`. An abstained or missing prediction is an error in the headline `accuracy`; `selective_accuracy` and `coverage` report it apart. Unlabeled GT is skipped, never a denominator.
- The comparator thresholds are documented defaults, recorded in the manifest; scoring stays deterministic for the same inputs. How `abstained` scores is a scoring decision, not a validator one.

## Secrets, mutation checks and CI (`tools/secret_scan.py`, `tools/mutation_check.py`, `.claude/`, `.github/`)
- `secret_scan.py` (stdlib only) scans ADDED lines for key shapes, `NAME_KEY=<value>` assignments and staged `.env` files; its output never contains a value. `--staged`, `--worktree`, `--history`, and `--env-values` (the real `.env` values searched in the history, in memory; prints variable names only). The Claude Code hook `.claude/hooks/pre_commit_secret_scan.py` runs it before any `git commit` (the whole working tree when the same command also does `git add` or `commit -a`, because the hook runs before the command) and blocks with exit 2; CI runs `--history`. A false positive is fixed by making the value an obvious placeholder (`your-key-here`), not by disabling the scan. Hooks load when Claude Code starts: review them with `/hooks` or restart the session.
- `mutation_check.py MUTANTS.py` is the mutation procedure of every milestone (skill: `.claude/skills/mutation-check`): mutants run on an ISOLATED COPY (a pass edits files a running Streamlit app imports), after a green baseline, and an ambiguous mutant is refused. Committed mutants: `tests/mutants/`.
- CI (`.github/workflows/ci.yml`) is `windows-latest` (the only platform the suite has run on) and runs the secret scan, `ruff check` and `pytest -q -m "slow or not slow"`. It has not run on GitHub yet. The app tests are `slow`-marked: a plain `pytest -q` skips them.

## HTTP API (`docextract/api/`, `calllog.py`, `extract/concurrent.py`)
- `python -m uv run uvicorn --factory docextract.api.app:create_app` (config from the environment/`.env`, nothing at import). `GET /health`; `POST /extract` = ONE document from the uploaded `files` (pages or photos in order; optional `page_range`, `doc_id`, `?compact=true`); `POST /extract/batch` = one file per document, answered synchronously (200, `{"run": ..., "documents": [...]}`: a bad file is `refused`, a provider failure `failed`, after the cost cap the rest are `not_run`). Answers are `DocRecord`s, built by the runner's own `extract_record`, so an API answer has the shape of a run-folder entry.
- One process-wide semaphore (`API_CONCURRENCY`) bounds documents in flight across all requests; `extract.py --concurrency N` uses the same `run_bounded` helper (default 1 = sequential). Calls already in flight finish after the cost cap is crossed, so spend can pass `MAX_RUN_COST_USD` by up to that many calls; parallel calls also spend a provider's per-minute/per-day quota faster.
- Uploads: bytes only, never a file name as a path; the same `load_pages` + `store_pages` + `IngestSettings` as ingestion minus redaction, so a PNG page is byte-identical to the ingested one (same response-cache key); a JPEG page is decoded and encoded again at q92, so it is NOT (slightly different pixels, other cache key); a temp dir per request, PDFs page-counted before rendering, `API_MAX_UPLOAD_MB`/`API_MAX_BATCH` limits, optional bearer `API_TOKEN`. Nothing is written under `data/`.
- Every request leaves one JSON line (request id, route, tokens, cost paid/avoided, cache hit, latency, attempts, repairs; never a file name or content) on the `docextract.calls` logger and in `CALL_LOG_PATH` when set.

## Docker, eval through the API, committed results (`Dockerfile`, `docker-compose.yml`, `scripts/eval_api.py`, `promote_run.py`, `check_results.py`, `results/`)
- One image (python 3.11-slim, uv `--frozen`, runtime deps only, non-root) for two services. `.dockerignore` is an ALLOW-LIST (`*` then `!src`...), so `.env`, `data/private`, `outputs`, `.git`, `ground_truth/drafts` never reach a layer. `api` takes `env_file: .env` (the pre-labeler key rides along unused); `eval` (profile `eval`, so plain `up` does not start it) gets no `env_file`, no model key, only `API_URL`/`API_TOKEN`. `./outputs` is mounted on both (cache, call log, runs).
- `eval_api.py` is the `extract.py` of the API: the same `docs/` + `run.json` folder and the same exit codes (0/1/2/3), every document POSTed to `/extract` with its stored pages, the blind rule enforced client-side, `run.json` filled from `GET /health` (model, endpoint, prompt/schema hashes, settings; `budget_spent_usd` = what the server's budget moved during the run). No selection = `--split test`; the default lives in the script because arguments to `docker compose run` REPLACE a compose `command`. The server's cost cap lasts as long as its process: `docker compose restart api` resets it.
- `promote_run.py RUN` copies a finished, scored, complete run to `results/<run_id>/` (refuses unscored, stopped, incomplete, existing, or key-shaped content). `check_results.py` re-scores every `results/*` with the committed git SHA/time and requires `score/` to match byte for byte (CI runs it; checkouts must be LF: the scorecard hashes label and manifest BYTES). A result is evidence of one sample of a noisy model (`docs/eval/noise.md`), never edited by hand.

## Regression comparison (`evalkit/compare.py`, `scripts/compare.py`)
- `compare` re-scores both runs against the labels as they are now (never a stored `score/`) and pairs the labeled scalar fields of the documents scored in BOTH; line items are compared in aggregate only.
- A regression needs BOTH the exact one-sided McNemar p <= `--alpha` (0.05) AND an accuracy drop >= `--max-drop` (0.02), compared as exact fractions. The document-level paired bootstrap (clusters = label owner) is reported, never gated; fewer than 10 independent documents carries a caution, because fields inside a document are not independent.
- Exit codes: 0 no regression, 1 REGRESSION, 2 cannot compare. This differs from `score.py` (1 = unscorable) on purpose: a gate needs 1 to mean "regressed".

## Sweep (`evalkit/sweep.py`, `scripts/sweep.py`)
- Varies ONE setting at a time against an existing baseline run (resolution: `LLM_IMAGE_MAX_LONG_EDGE`; thinking budget: `LLM_REASONING_EFFORT`, since output-token caps break the JSON of a reasoning model), then optionally runs the winning combination. COSTS MONEY (one run of the split per point; `--dry-run` calls nothing); refuses zero prices; `--resume` skips finished points; a half-finished run folder is never overwritten.
- The rule for the setting to keep is code, fixed before any result (`choose`): the cheapest variant with no failed document, accuracy within `--tolerance` (2 points) of the baseline AND no M6 regression verdict, strictly cheaper (or as cheap and faster) than the baseline; otherwise the baseline stays. A non-significant large drop is rejected: with ~10 documents "no significant loss" mostly means "cannot tell", and every report says how few independent documents back it.
- `score.py RUN --baseline BASE` adds a "Before / after" section (accuracy, cost per document, latency, tokens, the paired McNemar test, the bootstrap interval) to that run's scorecard.

## Failure analysis, noise and calibration (`evalkit/failures.py`, `noise.py`, `calibration.py`; reports in `docs/eval/`)
- `failures.collect` scores every labeled field twice with the scorer's own `score_field`: as returned (`validated`) and as the model said it (`parsed`). That is how "the guard withheld a correct value" is told from "the model was wrong". `noise.measure` compares N runs of ONE setting; `calibration.analyse` pools runs into reliability, risk-coverage and the threshold chosen by a rule fixed in code (lowest threshold with selective accuracy >= 0.95). Each script prints a data report and appends an optional hand-written `--notes` file as "Reading".
- **Noise rule.** Runs of the same setting differ (dev: accuracy up to 3.3 pts, 2-12 of 122 fields flip, output tokens +39%). A difference smaller than that is not a finding, and a p95 from one run of 10 documents is the single slowest call: quote medians and ranges. `docs/eval/noise.md` has the numbers.
- **Hallucination probe (S4, `evalkit/probes.py`, `probe_report.py`, `data/probes/`).** 9 copies of the constructed invoices cons_001..003 with a value painted out (found in the PDF text layer, filled with the colour around it) or the last page dropped, plus 3 unmodified controls; stored through the real ingest, labels derived by construction (erased = null, a dropped page nulls the totals and keeps the earlier rows, what is no longer decided is `__unlabeled__`). It is a mini-dataset: `extract.py/score.py --manifest data/probes/manifest.csv --pages-dir data/probes/pages --ground-truth-dir data/probes/labels`. `make_probes.py [--check]` is deterministic and refuses a probe whose pages are not byte-identical to the dataset's; `probe_report.py RUN` counts erased fields answered with a value (exact 95% interval; recovered vs other value; the control's read; the fields NOT erased). Result `probe-s4` (`docs/eval/hallucination_probe.md`, `results/probes/probe-s4`): 0 of 19, interval 0-17.6%, on clean pages; it says nothing about smudged or blurred text.
- `hallucinations.collect` takes the scorer's `hallucinated_value` apart (scalar fields and row cells, with denominators so it is a rate) and audits documents that share a label through `gt_from`: copies of one page should read alike, and a copy whose exact-match rate is 0.5 or more below its owner's is flagged `suspect` for a human to look at the pages (it flags, it never relabels). `docs/eval/hallucinations.md`: on `test-g1` 11 of the 12 scalar cases sit on doc_016 and its three photos doc_032..034, which the audit flags; the verdict (wrong label or invented receipts) is the owner's.
- Grounding flaw found by S1 (`docs/eval/failure_analysis.md`) and fixed by G1 (`docs/eval/grounding_fix.md`): a value whose quote prints an *ambiguous* number (`208,179`, `84.000`) was hard-abstained even when it was one of the two candidates. Now that is the soft code `evidence_ambiguous` (capped at 0.7, value kept, never rewritten); a value matching neither candidate, and every ambiguous DATE, stays hard. Replayed on the test split from the cache: 256 -> 273 of 323 correct, 17 fields improved, 0 regressed (`results/test-final` is the baseline, `results/test-g1` the fixed run).

## Dataset and ingestion (`src/docextract/ingest/`)
- **Privacy by location.** Real originals live in `data/private/raw/` and unredacted previews in `data/private/preview/`: that whole tree is gitignored and must never be committed. Only normalized, REDACTED pages (`data/pages/<doc_id>/p<N>.png|jpg`), `data/manifest.csv`, `data/redactions.yaml` and ground truth are committed. Do not read page content of real documents while building tooling: use synthetic fixtures (`tests/files.py`), and for real files only metadata, counts and sizes.
- `data/manifest.csv` has one row per LOGICAL document (14 columns + optional `gt_from`, `variant`). `scripts/manifest_init.py` scans the raw folder and appends rows for unreferenced files only (merge-safe, neutral `doc_NNN` ids, only strong-evidence inference); the rest is filled by hand. A row is never overwritten.
- `scripts/ingest.py` makes pages: EXIF upright -> PDFs at `INGEST_DPI` -> size cap (`INGEST_MAX_LONG_EDGE`, elongated pages by area) -> redaction -> encode (PDF/PNG sources -> PNG, JPEG/HEIC photos -> JPEG q92, ALL metadata stripped). Idempotent via a content hash recorded in `ingest.json` (which holds sizes, hashes and page mapping but never a raw file name). Never encode a page except through `imaging.encode_page`.
- **Redaction is a hard gate:** blank `needs_redaction`, or `y` without boxes, is refused (nothing is written); `n` with boxes is refused; boxes must fit the stored page. Burn BEFORE encoding. Boxes are in stored-page pixels.
- `data/redactions.yaml` (`ingest/redactions.py`): `doc_id -> page -> [[x, y, w, h], ...]`, optionally `size: [w, h]` per page (the page size the boxes were measured on; ingestion refuses a different one). The parser is strict on purpose: a repeated key (PyYAML would silently keep the last), an empty entry, a bad page number or box, an unknown doc_id, or boxes on a degraded variant all stop the run with exit 2 BEFORE any page is written, because anything accepted quietly is a page stored unredacted. Changing the DPI or size cap under boxes you did not touch is refused (`--accept-resize` if they are still right).
- `scripts/ingest.py --preview [--doc ID]` writes the UNREDACTED normalized pages to `data/private/preview/` (same pixel space as the stored pages, no `ingest.json`) so you can measure boxes before any exists. It refuses a directory that is, contains or sits inside `--pages-dir`, never reads `redactions.yaml`, and its output must never be committed or shared.
- **Splits (`ingest/splits.py`, `scripts/assign_split.py`):** the unit is the `content_group` (a group never straddles dev/test; a `gt_from` row must share its target's group), stratified by (real|constructed, doc_type), ~35% dev. Groups are ranked by a hash of (seed, stratum, group) so adding a document never reshuffles the others; assignments already in the manifest are NEVER moved (`--force` starts over, and refuses documents that already have ground truth). A doc_type with one group goes to test. Then 8-10 REAL test documents become `label_mode=blind`, round-robin over doc types, whole groups; other hand-labeled documents default to `prelabel`. A label_mode already in the manifest is a decision and is kept: a document that may carry a model-influenced label never turns blind.
- **Degraded variants (`ingest/degrade.py`, `scripts/degrade_docs.py`):** light/medium/heavy variants of `born_digital` documents, made from the source's STORED (already redacted) pages, never from raw files. Pixel transforms only: page count and size are preserved, the whole page stays inside the canvas (a skewed page is scaled down, not clipped), nothing is added or erased. Deterministic per (seed, source, level, page); the seed is in [1, 2**31-1] because Augraphy passes it to `cv2.setRNGSeed` and treats 0 as "unset". The manifest row (`<src>-deg-<level>`, capture=synthetic_degraded, `gt_from` the label owner, same content_group/split) is added right after the source. Augraphy is used when it imports, else OpenCV; the backend and its version are recorded in `degrade.json` and part of the content hash. `BadPhotoCopy` is deliberately not used (not reproducible inside a full pipeline). numpy/OpenCV/Augraphy live in the `degrade` dependency group (default groups: dev + degrade), imported lazily; Augraphy takes ~20 s to import and writes its textures to `outputs/cache/augraphy` (we chdir there), so its tests are `slow`.
- **Constructed invoices (`evalkit/construct.py`, `scripts/construct_invoices.py`):** three reportlab invoices (cons_001..003: classic, modern, compact layouts), 30-60 rupiah line items over 2-3 pages, subtotal/discount/PPN/total on the LAST page only, one layout with a repeated "(lanjutan)" header. The label is derived from the same `Invoice` value that is printed and records what the page PRINTS (Indonesian money like "1.250.000,00" -> decimal string "1250000.00" at the printed scale; a field a layout does not print is explicit `null`). Seeded and byte-reproducible; PDFs go to `data/private/raw/constructed/` (gitignored, regenerated). A label that already exists is never overwritten silently: identical = untouched, different = refused unless `--force`, and every created/replaced label gets a `ground_truth/CHANGELOG.md` line. Constructed docs have no label_mode (nobody labels them) and are their own `constructed` stratum in the split.
- Windows: a long raw file name (200+ chars) makes plain `Path.is_file()` return False and the file silently vanishes: every listing and read of raw files goes through `fsutil.long_path`.

## Ground truth
- Never edit `ground_truth/*.json` silently. Every change gets a line in `ground_truth/CHANGELOG.md`.
- Format and rules: `ground_truth/LABELING.md`. A label records HOW it was made (`label_mode` prelabel|blind, `labeled_at` with a timezone) and may carry per-field `field_notes`. `evalkit/gt_store.py` reads, writes (atomically, every field explicit, fixed order) and resolves labels: a document with `gt_from` has NO file of its own and gets the owner's label under its own id; `check_labels` / `scripts/check_ground_truth.py` check the lot, including that a BLIND document never has a model draft in `ground_truth/drafts/`.

## Workflow
- Small conventional commits after each working step.
- Append a short entry to `docs/DEVLOG.md` per milestone (what you did, decisions, problems).
- Run `pytest -q` and `ruff check` before every commit.
- Commit with the existing global git identity; do not change git config or rewrite authorship.
- Stage explicit paths, not `git add -A`: the working tree can hold edits that are not yours (e.g. a moved `.env.example`).

## Milestone procedure
Roadmap: `docs/ROADMAP.md`. Per-milestone spec: `docs/milestones/<ID>.md`.
State: the latest entry in `docs/DEVLOG.md` (read the tail only, not the whole file).
- One milestone per session. Do not touch files outside the spec's scope.
- Write the spec from the approved plan to its file BEFORE any code, and commit it
  on its own (`docs: <ID> spec`). Contents: goal, scope + explicit out-of-scope,
  files to create/change, files that must NOT be touched, acceptance criteria
  (runnable commands + expected output), risks.
  Check it against the actual code; a stale roadmap is reported as a deviation.
  Once written, the spec is not changed without my approval.
- Acceptance criteria must be offline and deterministic. Costly commands
  (smoke, prelabel, live eval) may appear only if marked [COSTS MONEY] with an
  estimate, and run only after I approve.
- Done = every criterion passes, and `pytest -q -m "slow or not slow"` and
  `ruff check` are green. Never loosen a criterion to make it pass.
- Wrap-up: a DEVLOG entry (max 15 lines: outcome, new public interfaces,
  deviations, tech debt, what's next). Edit CLAUDE.md only for permanent
  rules/modules (e.g. a new file in Layout), as little as possible.
- Spec format: follow the template in `docs/milestones/_TEMPLATE.md`.

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
python -m uv run python scripts/ingest.py --preview --doc ID   # UNREDACTED pages -> data/private/preview, to measure boxes
python -m uv run python scripts/assign_split.py [--dry-run] [--force]   # dev/test split + blind picks -> manifest
python -m uv run python scripts/degrade_docs.py (--doc ID | --all) [--levels light,heavy] [--check]   # degraded variants + manifest rows
python -m uv run python scripts/construct_invoices.py [--check] [--force]   # 3 constructed multi-page invoices + labels
python -m uv run python scripts/check_ground_truth.py   # label files vs manifest, blind-draft rule, coverage per split
python -m uv run python scripts/prelabel.py (--doc ID | --all) [--dry-run] [--force]   # COSTS MONEY: model drafts -> ground_truth/drafts
python -m uv run python scripts/extract.py (--split dev|test | --doc ID ...) [--run-id ID] [--cache on|off|refresh] [--concurrency N] [--dry-run]   # COSTS MONEY unless cached: -> outputs/runs/<run_id>/
python -m uv run python scripts/check_run.py outputs/runs/<run_id>   # records vs the value/confidence/evidence/status/reasons contract (offline)
python -m uv run python scripts/score.py outputs/runs/<run_id>   # scorecard (markdown + JSON) + per-doc results + run manifest -> <run_dir>/score/ (offline)
python -m uv run python scripts/compare.py outputs/runs/<baseline> outputs/runs/<candidate>   # flips + exact McNemar + bootstrap + verdict -> outputs/compare/<b>__vs__<c>/ (offline); exit 1 = regression
python -m uv run python scripts/sweep.py --baseline-run outputs/runs/<run_id> --sweep-id ID [--resolution 1024,1568] [--effort off,low] [--confirm] [--dry-run] [--resume]   # COSTS MONEY: one-setting-at-a-time sweep -> outputs/sweeps/ID/sweep.md
python -m uv run python scripts/analyze_failures.py outputs/runs/<run_id> [--out F] [--notes F]   # what was withheld and why, wrong fields, row misses per document (offline)
python -m uv run python scripts/noise_report.py RUN RUN [RUN ...] [--price-in X --price-out Y]   # spread between runs of ONE setting (offline); make the repeats with extract.py --cache refresh
python -m uv run python scripts/calibrate.py RUN [RUN ...] [--out F] [--svg-dir D]   # reliability, risk-coverage, the threshold the rule picks (offline)
python -m uv run python scripts/analyze_hallucinations.py RUN [--out F] [--review F] [--notes F]   # hallucinated_value by level/document/label owner/confidence + the shared-label audit (offline); --review (quotes, outputs/ only) is the list for a human
python -m uv run python scripts/make_probes.py [--check] [--force]   # the hallucination probe set -> data/probes (offline, deterministic)
python -m uv run python scripts/probe_report.py outputs/runs/<probe_run> [--out F] [--notes F]   # erased fields answered with a value, exact interval (offline); the run is extract.py over data/probes (see the S4 bullet)
python tools/secret_scan.py (--staged | --worktree | --history | --env-values)   # secrets in what a commit adds / the whole history / the real .env values (names only); exit 1 = found; stdlib only
python -m uv run python tools/mutation_check.py tests/mutants/<topic>.py   # mutation pass on an isolated copy; exit 1 = a mutant survived
python -m uv run uvicorn --factory docextract.api.app:create_app --port 8000   # the API: GET /health, POST /extract, POST /extract/batch (see HTTP API); calls the configured model, so it costs money unless cached
docker compose up -d --wait                          # the API in a container (needs .env); GET /health, POST /extract
docker compose run --rm eval [--split dev] [--doc ID] [--run-id ID]   # COSTS MONEY unless cached: eval through that API -> ./outputs/runs/<id>, scored
python -m uv run python scripts/eval_api.py [--api-url URL] [--dry-run]   # the same client without Docker, against any running API
python -m uv run python scripts/promote_run.py outputs/runs/<run_id>   # scored complete run -> results/<run_id>/ (committed evidence)
python -m uv run python scripts/check_results.py   # re-score results/* offline; exit 1 = a committed scorecard does not reproduce
python -m uv run streamlit run tools/label_app.py   # the labeling app
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
