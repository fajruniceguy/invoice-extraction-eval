# API demo: upload documents, get run-shaped records

M7 acceptance criterion 10. The real server, started from the repository's `.env`, serving the three
constructed invoices (`cons_001..003`, PDFs) through both endpoints. The model answers were already in the
response cache, so nothing was sent to any provider and nothing was spent: the demo shows the plumbing, not
a new accuracy number.

```
python -m uv run uvicorn --factory docextract.api.app:create_app --port 8765     # CALL_LOG_PATH=outputs/logs/api-demo.jsonl
```

## What was run, and what came back

```
$ curl http://127.0.0.1:8765/health
{"status":"ok","model":"gemini-3.5-flash","concurrency":4,"max_batch":20,"budget":{"spent_usd":"0","limit_usd":null,"exceeded":false},"calls":0,"cached_calls":0}

$ curl -F files=@cons_001.pdf "http://127.0.0.1:8765/extract?compact=true"
HTTP/1.1 200 OK            x-request-id: 20ce32f6c629

$ curl -F files=@cons_001.pdf -F files=@cons_002.pdf -F files=@cons_003.pdf http://127.0.0.1:8765/extract/batch
HTTP/1.1 200 OK            x-request-id: f3f48b7ad13f
```

Checked against the `m6-baseline` run folder (the same three documents, extracted by `scripts/extract.py`):

```
batch run: documents 3, ok 3, failed 0, refused 0, not_run 0, cost_usd 0, cached_calls 3, stopped None, wall_ms 1047.0, concurrency 4
single /extract cons_001: status ok, cost 0, cache_hit True, attempts 0 | validated == m6-baseline: True
batch cons_001: ok, cache_hit True, cost 0, 45 rows | validated == m6-baseline: True | page hashes equal: True
batch cons_002: ok, cache_hit True, cost 0, 57 rows | validated == m6-baseline: True | page hashes equal: True
batch cons_003: ok, cache_hit True, cost 0, 33 rows | validated == m6-baseline: True | page hashes equal: True
```

The call log (`CALL_LOG_PATH`), one JSON line per document request, four lines for these two requests:

```
/extract        index 0  ok  2 pages  2954 in / 10163 out  cost 0  cache_hit True  latency 8.9 ms
/extract/batch  index 0  ok  2 pages  2954 in / 10163 out  cost 0  cache_hit True  latency 8.9 ms
/extract/batch  index 1  ok  3 pages  4046 in / 11563 out  cost 0  cache_hit True  latency 16.4 ms
/extract/batch  index 2  ok  2 pages  2954 in / 7977 out   cost 0  cache_hit True  latency 7.3 ms
```

No file name and no document content is in the log. `/health` afterwards: 4 cached calls, 0 live calls.

## Why the records match the runner's

An upload goes through the same `load_pages` + `store_pages` + ingest settings as `scripts/ingest.py`, minus
redaction, so its pages are byte-identical to the stored ones (`page hashes equal: True`), and the response
cache key, which hashes the page images, is the same. The record itself is built by the runner's own
`extract_record`, so an API answer has exactly the shape of a run-folder entry and can be scored or compared.

## What the first attempt found

The first run of this demo was not clean, which is why it is worth keeping in the record. In the batch,
`cons_001` came back with different page hashes than the same file sent alone, missed the cache and made one
real model call (32.7 s); the other two replayed. The cause: the batch renders its PDFs in worker threads and
**PDFium is not thread-safe**. Offline, 3 threads rendering the same invoices crashed the interpreter with an
access violation; on the server it silently produced other pixels. The offline tests had missed it because they
used images and blank PDFs, and the offline stress test was added only after the demo found the bug.

Fixed in `98f5221`: every PDFium call in the upload path holds one process-wide lock (page counting and
rendering; images do not). Model calls, the slow part, stay concurrent. Tests: the lock is held on every call
and released on error, and a subprocess stress test renders four real text PDFs from four threads, eight rounds,
and compares every render with the serial one (0 mismatches; without the lock the subprocess crashes, which is
why it runs in a subprocess). That one first attempt spent one request of the provider's daily quota
(`gemini-3.5-flash`, cost $0.00 at the prices currently configured).

## Caveats

- The demo is a cache replay: it proves the pipeline, the parity with the runner and the logging, not live model
  latency. The live-call latency numbers are in the scorecards (`score.py outputs/runs/m6-degraded`).
- Concurrency spends a provider's quota faster. `API_CONCURRENCY` defaults to 4; lower it on a limited key.
- `MAX_RUN_COST_USD` is a server-lifetime budget here, and calls already in flight finish after it is crossed.
