# G3: a quantity printed with a multiplication sign, before and after

Spec: [docs/milestones/G3.md](../milestones/G3.md). Baseline [results/test-public](../../results/test-public/), candidate
[results/test-g3](../../results/test-g3/), both on the 16-document test split of the published dataset.

## What was wrong

The grounding check ([src/docextract/validate/grounding.py](../../src/docextract/validate/grounding.py)) looks for a value in its own evidence quote by
parsing whitespace-separated tokens. A quantity printed `x1` or `2x` is one token, and `x1` is not a number, so the quantity `1` was never found, the row got the
hard code `evidence_not_grounded`, and the whole row was abstained although the model had read it correctly. In `test-public` that was **9 rows: all 7 of doc_027
(`AIR MINERAL 5.000 x1 5.000`) and 2 of doc_020 (`Matcha 2x 169.000 338.000`)**, and every one of them scored as a missed row plus a spurious one.

## What changed

For the `quantity` cell only, a token that is exactly a number with a multiplication sign glued on (`x1`, `2x`, `X3`, `×3`, also `(2x)` or `x1,`) also offers
the bare number to the normalizer. The normalizer is unchanged and still never guesses: `x1.250` with no currency is as ambiguous as `1.250`. What is **not** read:

- a price, a total or any scalar through a sign (`x4.50` is not a unit price): a looser rule widens what a wrong value can hide behind;
- a unit (`250ml`, `2pcs`): `250ml` would ground a quantity of 250 that is a size;
- anything that is not exactly a sign and a number (`MAX1`, `2xl`, `x1x`, a lone `x`).

## The two runs differ only in the validators

`test-g3` replays the same 16 answers from the response cache: all 16 calls were cache hits, $0.0000 paid ($0.1606 answered from the cache), and for every document
`parsed` is byte-identical to `test-public`. The validators are not part of the cache key, so every difference below comes from them and none from sampling noise.

## Prediction against result

The prediction was written in the spec before any code ran.

| | predicted | observed |
| --- | --- | --- |
| scalar fields correct | 186 / 207, unchanged | 186 / 207 |
| line items matched / missed / spurious | 144 / 1 / 39 | 144 / 1 / 39 |
| row F1, all documents | 0.878 | 0.878 |
| row F1, the 14 real documents | 0.677 | **0.677** (was 0.532) |
| row F1, the 2 constructed documents | 1.000, unchanged | 1.000 |
| `compare.py test-public test-g3` | no regression, exit 0 | `VERDICT: OK`, 0 regressions, 0 improvements among the 207 paired fields |

Row by row: exactly **9 of 212 rows changed**, all from `abstained` to `extracted` (doc_027: 7, doc_020: 2), none the other way; the other 203 rows are identical,
and no scalar field changed. The 7 rows of doc_027 and the 2 of doc_020 went from missed + spurious to matched. The two doc_020 rows came back with the soft code
`evidence_ambiguous` (confidence 0.7, value kept): that document printed no currency, so its `169.000` is genuinely 169000 or 169, and the fix does not pretend otherwise.
Per document: doc_027 0 matched / 7 missed / 7 spurious -> 7 / 0 / 0; doc_020 0 / 2 / 2 -> 2 / 0 / 0.

## Read it with care

- **The gain is the two documents that print `x1` / `2x`, and nothing else.** Fields did not move at all, so accuracy is 0.899 in both. Row F1 on the real documents went from 0.532 to
  0.677, but that is nine rows in two documents: with 14 real documents it is evidence the bug is fixed, not an estimate of what the fix does on another dataset.
- **The rule was written after seeing these rows.** It is a fix for an observed bug, not a result that was predicted blind; what was predicted before running was only how many rows it
  would move, and that was exact.
- **Real documents are still at row F1 0.677** (42 matched, 1 missed, 39 spurious): precision is 0.519. The 39 spurious rows are untouched by this change.

## What is still withheld (not fixed here)

Three rows of `test-g3` still carry `evidence_not_grounded`, and they are different problems:

| document | row | what happens |
| --- | --- | --- |
| doc_021 | 2 rows (`1  Es Timun Jumbo`, `1  Sambak Bebek`) | the model's quote holds only the quantity and the name, not the amount it returned (15000 and 44091); the guard is doing its job on a quote that does not support the value |
| doc_033 | 1 row (`Pembulatan  -IDR 455`) | the amount is -455, but the minus sign sits on the currency marker (`-IDR`) and the digits are a separate token, so the quote parses to +455 |

The doc_033 case is the same kind of gap as this one (a sign split from its digits by a marker) and is the next candidate; it is one row, so it is not worth a milestone alone.
