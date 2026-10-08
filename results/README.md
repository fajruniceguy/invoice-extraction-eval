# Committed results

Real runs kept as evidence. Each `results/<run_id>/` holds `run.json` (model, endpoint, prompt and
schema hashes, settings, per-document status, cost), `docs/<doc_id>.json` (one record per document:
the model's raw text, what it said, what the validators made of it) and `score/` (`scorecard.md`
and `.json`, the run manifest, per-document results).

Nothing in here is edited by hand. `python -m uv run python scripts/check_results.py` scores every run
again offline (no model, no key) against the labels as they are now and requires `score/` to come out
byte for byte the same; CI runs it. A change to a label, the manifest, a stored page, a comparator or
the scoring code that moves a committed number makes it fail, and the right answer is a NEW run
(`scripts/promote_run.py`), not an edit.

## test-final: the test split through the Docker API

| | |
| --- | --- |
| command | `docker compose up -d --wait`, then `docker compose run --rm eval --run-id test-final` |
| split | test: 25 documents (11 phone photos, 9 born-digital PDFs, 3 screen captures, 2 constructed; 10 labeled blind) |
| model | `deepseek/deepseek-v4-flash-vision-exp` via Command Code, `json_object`, reasoning effort `medium` |
| prompt | `7ec75dbb29063cbb5146cb5ee5ce6fcbedda0140ec39971c8f018f404126390d` (the production prompt) |
| run on | 2026-10-08, response cache on but empty for these documents: every call was live |
| cost | $0.1992 for 25 documents ($0.0080 per document at the peak list prices in `.env.example`) |
| latency | p50 12.5 s, p95 58.5 s over 25 live calls (one sample: quote medians, not the p95) |
| result | 25 ok, 0 failed, 0 refused; 323 labeled fields, 256 correct: accuracy **0.793**, selective accuracy 0.859, coverage 0.923; rows F1 0.777 |

How to read it:

- **One sample of a noisy model.** Identical runs on the dev split differed by up to 3.3 accuracy points
  and 2-12 of 122 fields flipped between two of them (`docs/eval/noise.md`). A difference smaller
  than that between two runs is not a finding.
- **The known grounding flaw is NOT fixed in this run** (`docs/eval/failure_analysis.md`): the validators
  withhold a correct value when its evidence quote prints an ambiguous number. 23 of the 67 field errors
  are `abstained_on_value` (a value the model returned and the guard withheld); `total` is 0.52
  accurate but 0.81 when only answered fields count. This run is the baseline that a fix is compared
  against with `scripts/compare.py`.
- **Not the same pages as the dev runs, for JPEG documents.** The eval client uploads each stored page
  to the API, which decodes and re-encodes a JPEG at the same quality (a PNG comes back byte for byte).
  11 of the 25 test documents are phone photos, so their pixels differ slightly from the stored pages
  that `extract.py` sends; the pages hashed in each record are the server's.
- **`git: unknown` in the scorecard** is correct: the container has no `.git`. The scorecard still records
  the prompt hash, the ground-truth hash and the dataset hash.
- **The records echo (redacted) page content** in `raw_text`. They were read for key-shaped content
  by `promote_run.py` and the secret scan; nobody has read them line by line.
