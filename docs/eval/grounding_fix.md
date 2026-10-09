# G1: the grounding fix, before and after

> **Scored on the labels as they were on 2026-10-09 morning.** Three test labels (doc_032 to doc_034) were corrected afterwards
> ([ground_truth/CHANGELOG.md](../../ground_truth/CHANGELOG.md)). The tables below are the original ones and the runs they describe are in
> `results/superseded/`. On the corrected labels the same comparison (`compare.py results/superseded/test-final results/test-g2`) gives accuracy
> **0.845 -> 0.898** (273 -> 290 of 323), selective 0.916 -> 0.921, coverage 0.923 -> 0.975, rows F1 0.801 -> 0.841, and the same 17 fields improved with
> 0 regressed (McNemar p_better < 0.0001; bootstrap +5.3 points, 95% interval [+1.9, +9.4]). The fix's effect did not depend on the labels it was measured on.

Spec: `docs/milestones/G1.md`. Baseline `results/test-final`, candidate `results/test-g1`, both on the 25-document test split.

## What changed

A money value whose evidence quote prints an AMBIGUOUS number (`208,179`: 208179 or 208.179, with no currency printed) used to
be abstained as `evidence_not_grounded` (hard). If the model's value is one of the two readings, the quote now only fails to
CONFIRM it: the field keeps its value and gets the soft code `evidence_ambiguous`, capped at 0.7. A value that matches neither
reading is still hard, and so is every ambiguous date. Nothing rewrites a value; the normalizer still never picks a reading.

## The two runs differ only in the validators

`test-g1` was made by running the committed test split through the Docker API again with the response cache on. All 25 calls
were cache hits (cost paid $0.0000; $0.1992 answered from the cache), and for every document `raw_text` and `parsed` are
byte-identical to `test-final`. So the model said exactly the same thing twice, and every difference below comes from the
validators, with no sampling noise in it.

## Prediction against result

The rule was chosen from the dev analysis (2026-10-08) and replayed offline on `test-final` BEFORE the spec was written; the
prediction was stated in the spec. It was not adjusted afterwards.

| | predicted (spec) | observed |
| --- | --- | --- |
| scalar fields restored | 17 (256 -> 273 of 323) | 17 (256 -> 273 of 323) |
| doc_002's four wrong values (`9704.600` for `9.704.600`) | stay withheld | stay withheld |
| withheld rows restored | 21 of 34 | 21 (doc_021 7, doc_023 3, doc_024 4, doc_026 3, doc_029 4) |
| fields that go from correct to wrong | 0 | 0 |

## Scores (compare.py, exit 0)

| | test-final | test-g1 |
| --- | --- | --- |
| accuracy (abstentions are errors) | 0.793 | **0.845** |
| selective accuracy | 0.859 | 0.867 |
| coverage | 0.923 | 0.975 |
| `abstained_on_value` | 23 | 6 |
| rows: F1 / row accuracy / cell accuracy | 0.777 / 0.884 / 0.944 | 0.817 / 0.859 / 0.945 |

McNemar exact over 323 paired fields: 17 improvements, 0 regressions (p_better < 0.0001). Document-level bootstrap (22
independent documents): +5.3 points, 95% interval [+1.8, +9.9]. The 17 are doc_012 `total`; doc_020, 021, 023 `subtotal`/`tax`/
`total`; doc_024 `subtotal`/`total`; doc_026 `subtotal`/`tax`/`discount`/`total`; doc_029 `total`.

## Reading

- **The guard was withholding correct values, and it was almost all of what the guard did on money.** Selective accuracy
  barely moves (0.859 to 0.867) because the 17 restored fields were right: coverage rose 5 points at no loss in the accuracy of
  what is answered. The 6 remaining `abstained_on_value` are the 4 doc_002 values (the model scaled a printed `9.704.600` to
  `9704.600`, so the guard is right to withhold them), doc_018 `vendor_address` (grounding on a string) and doc_032 `total`
  (the model abstained itself).
- **Restored rows exposed their cells.** `row_missed` 29 to 21 and `row_spurious` 60 to 52, but three error classes went UP:
  `hallucinated_value` 24 to 27, `name_near_miss` 9 to 12, `missing_value` 10 to 12. A restored row is now compared cell by cell
  instead of counting as one missed plus one spurious row, so errors that were hidden inside an abstained row are counted. That
  is more honest, not a regression of the model, and it is the reason the A1 breakdown is run on `test-g1`.
- **Only 8 of the 21 restored rows changed the row score.** doc_021's table is labeled `unlabeled` (not scored); doc_029 has 5
  labeled rows and the model returned 19, so its 4 restored rows pair with nothing.
- **A caveat that matters for the cap.** A restored value carries confidence 0.7 at most, and `ABSTAIN_BELOW` is 0.3, so no
  restored value can ever be withheld by confidence. That cap is a placeholder, not a calibrated number (`docs/eval/calibration.md`
  measured the capped confidences as worse calibrated than the model's own).
- **What this does not show.** One sample of a noisy model (the replay deliberately removes the noise from the comparison, not
  from the estimate of how good the system is). The failure mode that stays: a model that writes the WRONG candidate of an
  ambiguous quote (`84` for a printed weight `84.000`) is kept at confidence 0.7; on this test split none of the 17 restored scalar values was wrong (rows were not checked individually).
