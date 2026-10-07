# CLAUDE.md

Project rules for Claude Code. Read this before making changes.

A document extraction system for invoices/receipts plus a rigorous evaluation harness.

## Stack and tooling
- Python 3.11 (project venv in `.venv`, created by uv), `uv` for dependencies, `ruff` + `pytest`.
- Type hints everywhere (ruff `ANN` enforces it in `src/` and `scripts/`). Pydantic v2 for all schemas.

## Layout
```
src/docextract/{config.py, providers/, ingest/, extract/, validate/, confidence/, api/}
src/evalkit/{comparators.py, matching.py, metrics.py, taxonomy.py, scorecard.py, compare.py, groundtruth.py, gt_store.py, construct.py, construct_io.py, prelabel.py, printed.py, labeling.py, labeling_ui.py}
data/{manifest.csv, redactions.yaml, pages/, private/{raw,preview}}
ground_truth/{, drafts/}
outputs/
scripts/
tools/{label_app.py}
tests/{fakes/, fixtures/}
docs/
```
`groundtruth.py` (the ground-truth schema), `gt_store.py` (label files), `construct*.py` (constructed invoices) and `prelabel.py` (model drafts), `printed.py` (printed text -> label values, shared by the pre-labeler and the labeling app) and `labeling*.py` (the labeling app's logic) are the files in `evalkit` not in the original layout brief.

## Provider
- The provider is OpenAI-compatible and fully configured via env: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, plus per-token prices for cost (`LLM_PRICE_INPUT_PER_MTOK`, `LLM_PRICE_OUTPUT_PER_MTOK`, USD per 1M tokens).
- Pre-labeling has its own `PRELABEL_*` variables (below); `.env.example` lists them.
- Optional env: `LLM_MAX_REPAIRS` (default 1: re-asks when output is not valid JSON / does not match the schema), `LLM_REASONING_EFFORT` (passed through as `reasoning_effort` when set), `LLM_STRUCTURED_MODE`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `LLM_BACKOFF_BASE_S`, `LLM_BACKOFF_MAX_S`, `LLM_MAX_PAGES` (default 15: a document with more pages is refused before any request; all its pages go in one call), and `MAX_RUN_COST_USD` (below). `.env.example` lists them all.
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

## Extraction runs (`extract/runner.py`, `scripts/extract.py`)
- **A BLIND document, or a variant of one (`gt_from`), is never extracted until its label is finished** (`ground_truth/<doc_id>.json` exists): it is refused with zero requests, because no model output may exist for it before a human has labeled it from the page alone.

## Pre-labeling (`evalkit/prelabel.py`, `scripts/prelabel.py`)
- A model drafts ground truth into `ground_truth/drafts/<doc_id>.json` (a label file with `label_mode: prelabel` and no `labeled_at`); a human verifies every field. It is configured by `PRELABEL_BASE_URL/API_KEY/MODEL` (+ prices, `PRELABEL_MAX_RUN_COST_USD`) and runs through the same `OpenAICompatClient` via `Settings.prelabel_settings()`: json_object pinned (the schema is in the prompt: Anthropic's OpenAI-compatible endpoint ignores `response_format`), page images downscaled in memory to 1568 px and under 4.5 MB, the extractor's reasoning effort and cost cap not inherited.
- **Hard rules, enforced in code and tested to make zero requests:** a BLIND document is never drafted (`BlindDocumentError`; `save_draft` re-checks, `check_labels` flags a blind draft); only hand-labeled `label_mode=prelabel` documents qualify (not borrowers, constructed, or blank label_mode); the model must differ from the extractor's (same endpoint + model is refused, `--allow-same-model` is the escape hatch); only the STORED, REDACTED pages are sent (verified against `ingest.json`, raw files never read); a finished label is never drafted over.
- The model transcribes what is PRINTED as strings; `docextract.normalize` turns them into values. Anything it cannot resolve without guessing (ambiguous `1.250` with no currency, `05/03/2024`) becomes `"__unlabeled__"` with a field note. Null = not printed, a missing key = not answered (unlabeled).

## Labeling app (`tools/label_app.py`, logic in `evalkit/labeling.py` + `labeling_ui.py`)
- `python -m uv run streamlit run tools/label_app.py` (streamlit is in the default `tools` group). Left: the STORED (redacted) pages with the pixel under the cursor and a draggable box that prints `[x, y, w, h]` and a `redactions.yaml` snippet (an opt-in UNREDACTED preview toggle exists only for finding box coordinates). Right: the form.
- Every field starts PENDING and must be decided: `verified` (value as printed, normalized with the shared normalizer; an ambiguous or unreadable value is an inline error, never a guess), `absent` (null) or `unlabeled` (`"__unlabeled__"`). **Save is blocked until nothing is pending**, so a draft never becomes ground truth by being left alone. A prelabel document starts from the draft (still all pending); a BLIND document starts empty and `BlindDraftError` is raised if a draft exists for it.
- Saving writes `ground_truth/<doc_id>.json` with `label_mode` + `labeled_at` AND appends its own `CHANGELOG.md` line(s) (per changed field when replacing; replacing a finished label needs a reason; if the log cannot be written the label file is put back). An identical label is not rewritten.
- Work in progress is kept in `outputs/labeling/<doc_id>.json` (gitignored; written only on real edits, never for just looking) and nothing in it is trusted on reload: its shape is checked and every field it calls verified must verify again.
- The viewer's JavaScript geometry and event glue are tested under Node (against a Python reference and the redaction code); the app itself is tested headlessly with Streamlit's `AppTest`.

## Core logic
- Extraction logic and the eval harness are hand-written. No LangChain / LlamaIndex for core logic.

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
python -m uv run python scripts/extract.py (--split dev|test | --doc ID ...) [--run-id ID] [--cache on|off|refresh] [--dry-run]   # COSTS MONEY unless cached: -> outputs/runs/<run_id>/
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
