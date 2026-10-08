# M8 sweep: one setting at a time, on the dev split

M8 acceptance criterion 6. Image resolution (`LLM_IMAGE_MAX_LONG_EDGE`) and thinking budget
(`LLM_REASONING_EFFORT`) were varied one at a time against a baseline, then the winning level of each
was run together once. The rule for the setting to keep is code (`evalkit.sweep.choose`) and was fixed
before any result existed.

Model `deepseek/deepseek-v4-flash-vision-exp` through Command Code, production prompt, `json_object`
mode, 10 dev documents (9 real + `cons_003`). The runs were costed at $0.22 / $0.66 per million tokens
(input / output), a third-party tracker's figure. Command Code's own page (pricing-limits) says $0.15 /
$0.60 off-peak and $0.30 / $1.20 at peak (01-04 and 06-10 UTC, Monday to Friday). Every point used the
same prices, so the comparisons stand; absolute costs are about 15% too high against off-peak and about 40%
too low against peak: the baseline is $0.0047 per document as recorded, about $0.0040 off-peak and
$0.0081 at peak.

```
python -m uv run python scripts/extract.py --split dev --run-id dev-baseline-ds --concurrency 3
python -m uv run python scripts/sweep.py --baseline-run outputs/runs/dev-baseline-ds --sweep-id m8 --confirm --concurrency 3
python -m uv run python scripts/score.py outputs/runs/sweep-m8-effort-low --baseline outputs/runs/dev-baseline-ds
```

## Result

| point | setting | failed | accuracy | change | p (worse) | 95% interval | selective | coverage | row F1 | cost/doc | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | stored size, provider-default effort | 0 | 0.811 | - | - | - | 0.925 | 0.877 | 0.792 | $0.0047 | 16.6 s | 81.4 s |
| res-1024 | long edge 1024 | 0 | 0.762 | -4.9 pts | 0.132 | [-13.0, +2.4] | 0.853 | 0.893 | 0.688 | $0.0039 | 13.4 s | 56.9 s |
| res-1568 | long edge 1568 | 0 | 0.836 | +2.5 pts | 0.969 | [-1.6, +8.3] | 0.927 | 0.902 | 0.815 | $0.0041 | 13.7 s | 44.6 s |
| effort-off | `reasoning_effort=off` | 0 | 0.787 | -2.5 pts | 0.274 | [-9.2, +4.0] | 0.873 | 0.902 | 0.813 | $0.0020 | 6.7 s | 13.4 s |
| effort-low | `reasoning_effort=low` | 0 | 0.803 | -0.8 pts | 0.500 | [-8.7, +6.8] | 0.899 | 0.893 | 0.684 | $0.0037 | 13.1 s | 48.0 s |
| combined | 1568 + low | 0 | 0.803 | -0.8 pts | 0.500 | [-8.7, +6.7] | 0.916 | 0.877 | 0.684 | $0.0037 | 15.2 s | 28.8 s |

Cost is at list price (paid + what cache replays would have cost). Latency is over live calls only.

**Chosen by the rule: `effort-low`.** The rule keeps the cheapest variant with no failed document,
accuracy within 2 points of the baseline and no regression verdict, that is strictly cheaper than the
baseline; ties go to the lower p50. `effort-off` and `res-1024` fell more than 2 points; `res-1568`,
`effort-low` and `combined` qualified; `effort-low` and `combined` cost the same and `effort-low` is
faster at p50. Before / after for the chosen run (`score.py ... --baseline`):

| metric | before | after | change |
| --- | --- | --- | --- |
| accuracy | 0.811 | 0.803 | -0.8 pts |
| cost per doc (list) | $0.0047 | $0.0037 | -23% |
| latency p50 / p95 | 16.6 s / 81.4 s | 13.1 s / 48.0 s | -21% / -41% |
| output tokens per doc | 5393 | 3947 | -27% |
| row F1 | 0.792 | 0.684 | -10.9 pts |

## What this does and does not show

- **The accuracy changes are inside the noise.** Every interval spans zero, with half-widths of about
  8 points on 122 fields from 10 documents; one field is 0.8 points. "No detectable loss" is the most
  this supports, not "no loss". Single documents moved by up to 3 fields in both directions against the
  baseline (doc_008 +3, doc_022 -3), which looks like sampling noise on one run each. **No noise floor
  was measured**: repeating the baseline with `--cache refresh` would measure it (about $0.05).
- **The rule cannot see line items, and the chosen run is worse on them.** Row F1 fell from 0.792 to
  0.684 (matched rows 61 -> 54, missed 20 -> 27, spurious 12 -> 23). The M6 comparison, and so the
  rule, covers scalar fields only. But row F1 also swings the other way across points (0.815 for
  `res-1568`, 0.813 for `effort-off`), so it is probably noisy as well, not a clean effect of the
  setting. It is a reason to doubt "free 23% saving", not proof of damage.
- **Row accuracy (all cells of a row right) is dominated by one document.** It is 0.295 in the baseline
  and 0.359 in `res-1568`, against 0.82-0.89 in the other four runs. All of that is `cons_003` (33 of
  the 81 labeled rows): in those two runs every one of its 33 rows is matched but the description
  carries the item code ("BRG-001 Map Plastik Biru"), so each row is a `name_near_miss` and 0 of 33 rows
  are fully right; the other runs, and the GLM baseline, do not do that. Field accuracy and row F1
  cannot see it, because rows are matched fuzzily. It is a prompt/format ambiguity (the schema has no
  item-code field) that flips between samples, not an effect of resolution or effort.
- **`effort-off` was rejected by half a point.** It is 2.4x cheaper and 2.5x faster (p95 13.4 s) at 3
  fewer correct fields out of 122, with p = 0.27. The 2-point tolerance was set before the results and
  was not moved afterwards; whether that trade is acceptable is a product decision, not a measured one.
- `doc_022` was answered from the response cache in the baseline, `res-1568` and `combined`: its page is
  smaller than 1568 px, so those requests are byte-identical to others already made. That is a correct
  cache hit, but those three rows share one sample for that document.
- The sweep ran on 10 documents; the test split is untouched.

## Model context

The baseline itself replaced an earlier one: the same 10 documents, same prompt, with GLM 5.3 Flash
(`dev-baseline`, kept in `outputs/runs/`):

| | GLM 5.3 Flash | DeepSeek V4 Flash Vision (exp) |
| --- | --- | --- |
| accuracy / selective / coverage | 0.820 / 0.926 / 0.885 | 0.811 / 0.925 / 0.877 |
| cost per doc (list) | $0.0050 | $0.0047 |
| latency p50 / p95 | 122.5 s / 586.4 s | 16.6 s / 81.4 s |

Same accuracy within noise at about 7x lower median latency, which is why the extractor was switched
mid-milestone. Both are one run of 10 documents.
