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

## Index

| run | what it is |
| --- | --- |
| `test-g3` | **the current result**: the same 16 raw answers as `test-public`, replayed from the response cache with the quantity-sign fix (`x1`, `2x`; `docs/eval/quantity_marks.md`). Fields unchanged (186/207); rows matched 144 / missed 1 / spurious 39, F1 0.878 overall and **0.677 on the 14 real documents**. All 16 calls were cache hits, $0.0000 paid. Reproduces with `check_results.py` |
| `test-public` | the baseline of `test-g3`: the 16-document test split of the published dataset (14 real, 2 constructed), grounding as it was before the fix. Accuracy 0.899 (186/207), selective 0.916, coverage 0.981; rows F1 0.823 overall but **0.532 on the 14 real documents** (the two constructed invoices supply 102 of the 135 matched rows). Reproduces with `check_results.py`. `compare.py test-public test-g3` gives `VERDICT: OK` |
| `probes/probe-s4` | the hallucination probe (S4): 12 documents of `data/probes`, not the dataset: 0 of 19 erased fields answered with a value. Its own manifest, so `check_results.py` does not cover it; re-score it with `score.py results/probes/probe-s4 --manifest data/probes/manifest.csv --pages-dir data/probes/pages --gt-dir data/probes/labels --out <dir>` |
| `original-dataset/test-final` | the 25-document test split of the ORIGINAL 35-document dataset, first real run through the Docker API, grounding unfixed, on the labels before the relabel of doc_032 to doc_034 (accuracy 0.793; 0.845 on the corrected labels) |
| `original-dataset/test-g1` | the same answers with the grounding fix, on the old labels (0.845 there; 0.898 on the corrected labels) |
| `original-dataset/test-g2` | the same answers with the grounding fix on the corrected labels: accuracy 0.898 (290/323), rows F1 0.841 |

## test-public: how it was made

| | |
| --- | --- |
| command | a local API (`uvicorn --factory docextract.api.app:create_app`, cost cap `MAX_RUN_COST_USD=0.30`), then `scripts/eval_api.py --split test --run-id test-public`. Not the Docker route: Docker was not available on the machine |
| split | test: 16 documents (11 phone photos, 2 born-digital PDFs, 1 screen capture, 2 constructed; 6 labeled blind, all of them real) |
| model | `deepseek/deepseek-v4-flash-vision-exp` via Command Code, `json_object`, reasoning effort `medium`; prompt `7ec75dbb29063cbb5146cb5ee5ce6fcbedda0140ec39971c8f018f404126390d` (unchanged since the original runs) |
| run on | 2026-10-14. **14 of the 16 answers came from the response cache**: 13 made by earlier runs with the same model, prompt and settings, and doc_012 by an aborted attempt earlier the same day; doc_016 and doc_019 were live |
| cost | $0.0238 paid, $0.1606 at list price ($0.0100 per document at the peak prices in `.env.example`); the aborted attempt added $0.0055 outside this run |
| latency | p50 56.3 s, p95 66.9 s over **2 live calls**: not a latency measurement (the original run of 25 live calls had p50 12.5 s, p95 58.5 s) |
| result | 16 ok, 0 failed, 0 refused, 0 repairs; 207 labeled fields, 186 correct; rows matched 135, missed 10, spurious 48 |

How to read it:

- **The row F1 does not separate real from constructed documents.** Split by the document's capture: constructed 102 matched / 0 missed / 0 spurious (F1 1.000); the 14 real documents 33 / 10 / 48 (F1 **0.532**).
  Compute it per document from `score/docs/*.json` (`rows.matched|missed|spurious`).
- **Nine rows are withheld by a grounding gap that `test-g3` fixes** (a quantity printed `x1` or `2x` is never found in its own quote, so the row is `evidence_not_grounded`): all 7 rows of doc_027 and 2 of doc_020. This run keeps the flaw on purpose: it is the baseline the fix is measured against.
- **A mix of days.** Because most answers are cache hits from earlier runs, this is one sample of a noisy model assembled from several: do not read a difference smaller than the noise (`docs/eval/noise.md`) between it and a later run as a finding.
- **The records echo page content** in `raw_text`, including vendor tax ids and addresses, because no redaction box was applied to any published document. They were read for key-shaped content by `promote_run.py` and the secret scan, not line by line.

## test-g3: how it was made

The same command as `test-public` (a local API with `MAX_RUN_COST_USD=0.0001` and concurrency 1, then `scripts/eval_api.py --split test --run-id test-g3`) after commit `7349d51`, on 2026-10-14. The
validators are not part of the response-cache key, so all 16 answers were cache hits ($0.0000 paid, $0.1606 answered from the cache) and every document's `parsed` is byte-identical to
`test-public`: the difference between the two runs is the validators alone, with no sampling noise in it. Exactly nine rows changed (doc_027: 7, doc_020: 2, all `abstained` -> `extracted`), none
the other way, and no scalar field moved. There is no latency figure (no live call). Row by row and what is still withheld: [docs/eval/quantity_marks.md](../docs/eval/quantity_marks.md).

## original-dataset: scorecards only

`original-dataset/` holds `run.json`, `score/scorecard.md`, `score/scorecard.json` and `score/manifest.json` of three runs on the 25-document test split of the original 35-document dataset. The per-document
records (`docs/`, `score/docs/`) were **removed** because 9 of those 25 documents (and 12 of the 35) are not published: they were downloaded from the internet and their source and licence were never recorded.
So these three do **not** reproduce and `check_results.py` does not look at them (it covers `results/*/run.json`); they stay as the aggregate evidence behind the numbers the README and `docs/eval/` quote
(0.793 -> 0.845 from the label correction, 0.845 -> 0.898 from the grounding fix, the same raw answers). Where their `git` field is set it names a commit of the history as it was before it was rewritten to remove the 12 documents.

### test-final: the test split through the Docker API

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
- **The known grounding flaw is NOT fixed in this run (it is in `test-g2`)** (`docs/eval/failure_analysis.md`): the validators
  withhold a correct value when its evidence quote prints an ambiguous number. 23 of the 67 field errors
  are `abstained_on_value` (a value the model returned and the guard withheld); `total` is 0.52
  accurate but 0.81 when only answered fields count. This run is the baseline that a fix is compared
  against with `scripts/compare.py`.
- **Not the same pages as the dev runs, for JPEG documents.** The eval client uploads each stored page
  to the API, which decodes and re-encodes a JPEG at the same quality (a PNG comes back byte for byte).
  11 of the 25 test documents are phone photos, so their pixels differ slightly from the stored pages
  that `extract.py` sends; the pages hashed in each record are the server's.
  **Found later: the PNG half of that was not true in the Docker run.** The server hashed doc_012's page as `70181c46...`, while the stored page (and the same upload through a server on the
  machine that ingested it) is `c0a073b6...`. The cache keys follow those bytes, so a replay outside Docker missed the cache for the PNG documents that had been answered in Docker
  (doc_012, doc_016, doc_019, the three real PNG documents of the split): that is why doc_016 and doc_019 were live in `test-public`, and doc_012 in the attempt before it. The
  JPEG documents hit the cache. Why the container's bytes differ was not investigated.
- **`git: unknown` in the scorecard** is correct: the container has no `.git`. The scorecard still records
  the prompt hash, the ground-truth hash and the dataset hash.
