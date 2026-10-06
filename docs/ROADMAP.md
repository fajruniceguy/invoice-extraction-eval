Milestones
Must have
ID	Deliverable	Definition of done
M1	Scaffolding: repo layout, CLAUDE.md, config, provider client, Pydantic schema, .env.example, pytest, ruff	Definition of done: pytest passes; a smoke call returns JSON from the provider
M2	Dataset and ground truth: ingestion (PDF/image to normalized page PNGs), manifest with slices (source, pages, quality, split), GT JSON per doc, LABELING.md protocol, ground_truth/CHANGELOG.md 	Definition of done: All 40 docs labeled. Labeling is done blind, before seeing any model output. GT hash recorded.
M3	Extraction v1: multi-page VLM call with JSON-schema output, Pydantic validation, retry on malformed output, evidence quote and page per field, explicit "null if absent" instruction	Definition of done: Runs on all dev docs with no crashes, and malformed outputs get repaired
M4	Validators, confidence and abstention: arithmetic reconciliation (sum of lines ±tax ±discount = total), date/currency/tax-ID checks, evidence grounding, confidence formula, abstention with reason codes	Definition of done: Every field carries value, confidence, evidence, status (extracted/abstained) and reasons
M5	Eval harness: type-aware comparators (normalized strings, fuzzy names with threshold, numeric tolerance, date normalization), line items matched with the Hungarian algorithm, plus an error taxonomy	Definition of done: Scorecard (markdown and JSON) and per-doc results, with a run manifest recording git SHA, prompt hash, model, config, GT hash and dataset hash
M6	Regression comparison: a compare command diffs two runs, lists per-field flips (correct→wrong), runs McNemar or a paired bootstrap, and exits non-zero past a threshold	Definition of done: Demonstrated on a deliberately degraded prompt, which gets caught and documented
M7	API, batching and cost/latency: FastAPI /extract and /extract/batch, async concurrency with a semaphore, backoff, per-call token, cost and latency logging	Definition of done: Scorecard shows cost per doc and latency p50/p95, overall and by slice
M8	One optimization with measured impact: sweep image resolution and token budget across 3 levels, and report accuracy vs. cost vs. latency	Definition of done: A before/after table in the scorecard; the chosen setting is justified
M9	Docker: docker compose up serves the API; docker compose run --rm eval hits the API and writes to ./outputs	Definition of done: Works from a clean clone with only .env filled in. Real run artifacts committed.
M10	Agentic-development note, a custom extension you actually used, and the README	Definition of done: All the README sections the brief lists are present

Should have
S1. Calibration: a reliability diagram and a risk-coverage curve, with the abstention threshold chosen on dev and reported on test.
S2. Self-consistency: a second sample at temperature > 0, where disagreement lowers confidence. Measure whether it improves calibration enough to justify the extra cost.
S3. Model cascade: a cheap model first, escalating only low-confidence docs or fields. Report cost saved against accuracy lost; this could serve as a second optimization.
S4. Hallucination probe set: 5–8 copies of documents with fields digitally erased (invoice number, tax ID, page 2 removed). The expected output is abstention, and the hallucination rate on this set is reported separately.
S5. Local model run: the full eval on Ollama, compared head-to-head using M6.
S6. CI: GitHub Actions running unit tests and an offline eval that re-scores committed predictions, so no API key is needed.
S7. Robustness slice: rotated, blurred or low-contrast variants of test docs.

Nice to have
N1. Architecture ablation: OCR (PaddleOCR or Tesseract) plus an LLM, compared against the VLM on the same scorecard.
N2. Review UI: a minimal page listing abstained fields next to the page image, which shows the human-in-the-loop story.
N3. Content-hash response cache and idempotent batch jobs.
N4. Metrics endpoint (Prometheus-style) for latency and cost.
N5. Second-annotator pass on 10 docs to report inter-annotator agreement on the ground truth.
N6. Field-level re-ask using a cropped image region around the evidence.