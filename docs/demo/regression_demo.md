# Regression gate demo: a deliberately degraded prompt, caught

M6 acceptance criterion 7. The production prompt was edited three ways on purpose
([degraded_prompt.txt](degraded_prompt.txt), frozen, hash pinned in `tests/test_degraded_prompt.py`),
both versions were run on the same three documents, and `scripts/compare.py` was asked whether the
second is worse than the first.

## The three edits

| edit | what the prompt says now | intended damage |
| --- | --- | --- |
| E1 | `subtotal` is "the grand total or amount to pay", `total` is "the amount before tax" | subtotal and total swapped |
| E2 | "Express every money amount in thousands of rupiah: divide the printed amount by 1000" | a unit error on every amount |
| E3 | "Never leave a field empty ... give your best estimate and use status extracted" instead of "absent ... Never guess" | hallucinated values for fields the document does not print |

## What was run

Documents: `cons_001`, `cons_002`, `cons_003` (the constructed invoices: 39 labeled fields, 135 line
items). Model `gemini-3.5-flash`, same settings both times, `LLM_MAX_RETRIES=1`.

```
python -m uv run python scripts/extract.py --doc cons_001 --doc cons_002 --doc cons_003 --run-id m6-baseline
python -m uv run python scripts/extract.py --doc cons_001 --doc cons_002 --doc cons_003 \
    --system-prompt-file docs/demo/degraded_prompt.txt --run-id m6-degraded
python -m uv run python scripts/compare.py outputs/runs/m6-baseline outputs/runs/m6-degraded   # exit 1
```

The baseline needed 1 live call (the other two came from the response cache), the degraded run 3. No
repairs, no failures. (Cost shows $0.00: the free tier and the prices configured in `.env`.)

## What compare said (verbatim, exit code 1)

```
VERDICT: REGRESSION (p_worse 0.0005 <= alpha 0.05; accuracy drop 28.2 pts >= 2.0)
compare: baseline m6-baseline  vs  candidate m6-degraded
differs:
  prompt_sha256: b154b4cc8ac008754233412be37225c1ef4928d… -> fe74864d766e803f8a8c3fb4744ee5dd7e61475…
paired: 3 document(s), 39 labeled field(s), 3 independent
accuracy 1.000 -> 0.718, selective 1.000 -> 0.966, coverage 1.000 -> 0.744
rows: F1 1.000 -> 0.000, cell accuracy 1.000 -> n/a
McNemar exact: 11 regression(s), 0 improvement(s); p_worse 0.0005, p_better 1.0000, two-sided 0.0010
bootstrap (3 document(s), seed 1, 10000 resamples): delta -28.2 pts, 95% interval [-30.8 pts, -23.1 pts]
note: only 3 independent document(s) (fewer than 10): fields inside a document are not independent, so the McNemar p-value is optimistic and the bootstrap interval is only indicative
errors: missing_value 0 -> 1, abstained_on_value 0 -> 10, row_missed 0 -> 135, row_spurious 0 -> 135
regressions (11):
  cons_001 subtotal: expected 989476500.00; baseline 989476500.00 -> candidate (abstained) (abstained_on_value)
  cons_001 tax: expected 103400294.00; baseline 103400294.00 -> candidate (abstained) (abstained_on_value)
  cons_001 discount: expected 49473825.00; baseline 49473825.00 -> candidate (abstained) (abstained_on_value)
  cons_001 total: expected 1043402969.00; baseline 1043402969.00 -> candidate (absent) (missing_value)
  cons_002 subtotal: expected 1278017500; baseline 1278017500 -> candidate (abstained) (abstained_on_value)
  cons_002 tax: expected 140581925; baseline 140581925 -> candidate (abstained) (abstained_on_value)
  cons_002 total: expected 1418599425; baseline 1418599425 -> candidate (abstained) (abstained_on_value)
  cons_003 subtotal: expected 628952500.00; baseline 628952500.00 -> candidate (abstained) (abstained_on_value)
  cons_003 tax: expected 69131425.00; baseline 69131425.00 -> candidate (abstained) (abstained_on_value)
  cons_003 discount: expected 485000.00; baseline 485000.00 -> candidate (abstained) (abstained_on_value)
  cons_003 total: expected 697598925.00; baseline 697598925.00 -> candidate (abstained) (abstained_on_value)
```

"What differs" lists the prompt hash and nothing else, so the 11 flips are the prompt's. Running compare
again gives a byte-identical `comparison.json`.

## What it shows

- **The money fields broke, exactly where E1 and E2 aim.** Every regression is `subtotal`, `tax`,
  `discount` or `total`; the nine other fields did not move (see "By field" in `comparison.md`). The
  model followed the edits: on `cons_001` it returned subtotal `1043402.969`, which is the grand total
  divided by 1000.
- **The pipeline's own validators caught most of it, and the label comparison measured it.** The scaled
  values no longer match their evidence quotes, so the grounding validator abstained on them
  (`evidence_not_grounded`) and on all 135 line items; `lines_ne_subtotal` fired on `cons_001`. That is
  why accuracy falls 28.2 points but selective accuracy only 3.4 (0.966): what was still answered was
  right, and coverage fell from 1.000 to 0.744. The headline `accuracy` counts abstentions as errors,
  so the gate fires on that; reading selective accuracy alone would have hidden the damage.
  (An earlier plan assumed E2 would slip past every internal check. It did not: the spec's wording
  was wrong, the grounding validator is stronger than assumed.)
- **E3 did nothing visible.** `cons_002` has four fields the document does not print and all four stayed
  correct (`absent`): no `hallucinated_value` appears in the error mix. This model kept the
  absent rule despite being told not to; that is a property of this model and these documents, not of
  the gate.
- **Line items:** all 135 rows are abstained, so none matches a label: row F1 1.000 -> 0.000. Cell
  accuracy has nothing to measure (n/a), which M5 would otherwise print as 1.000.

## A confound the tool caught on the way

The first comparison used M4's `m4-live` run as the baseline. Its "what differs" section listed the
model (`gemini-3.6-flash` -> `gemini-3.5-flash`, because `.env` had changed since), the prompt hash and a
removed cost cap, so the verdict could not be attributed to the prompt. The baseline was re-run under
the current settings (`m6-baseline`) to get the clean comparison above. The 11 regressions and the
p-value were the same in both, but only the second one proves it.

## Caveats

- Three constructed invoices are the easiest documents there are: this shows the gate catches a blunt
  change, not how sensitive it is on real receipts. With 39 paired fields it can only fire at 5 or more
  net regressions (13% of the fields).
- Those 39 fields come from 3 independent documents. Fields in a document are not independent, so the
  McNemar p-value is optimistic; the report says so, and prints the document-level bootstrap interval
  beside it (the interval never gates).
- Repeat this on `--split dev` once real dev documents are labeled (M2's human steps).
