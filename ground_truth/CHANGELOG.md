# Ground-truth changelog

Every change to any `ground_truth/*.json` gets one line here, in the same commit as the change.

Format: `YYYY-MM-DD | file | field | old -> new | reason`

---

2026-10-06 | cons_001.json | (new file) | - -> 2 pages, 45 items | constructed by scripts/construct_invoices.py (seed 'docextract-construct-v1', classic layout)
2026-10-06 | cons_002.json | (new file) | - -> 3 pages, 57 items | constructed by scripts/construct_invoices.py (seed 'docextract-construct-v1', modern layout)
2026-10-06 | cons_003.json | (new file) | - -> 2 pages, 33 items | constructed by scripts/construct_invoices.py (seed 'docextract-construct-v1', compact layout)

2026-10-08 | doc_001.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_008.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_011.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_013.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_015.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_022.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_025.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_028.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_030.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_004 | (draft removed) | excluded from the dataset | two invoices on one page: one label cannot describe it

2026-10-08 | doc_017 | (row removed) | excluded from the dataset | a copy of doc_004, which was excluded

2026-10-08 | doc_010 | (draft removed) | excluded from the dataset | a 20-page bundle of many companies, not one document

2026-10-08 | doc_005.json | (new file) | - -> label_mode=blind | labeled by hand

2026-10-08 | doc_006.json | (new file) | - -> label_mode=blind | labeled by hand

2026-10-08 | doc_007.json | (new file) | - -> label_mode=blind | labeled by hand

2026-10-08 | doc_012.json | (new file) | - -> label_mode=blind | labeled by hand

2026-10-08 | doc_018.json | (new file) | - -> label_mode=blind | labeled by hand

2026-10-08 | doc_019.json | (new file) | - -> label_mode=blind | labeled by hand

2026-10-08 | doc_021.json | (new file) | - -> label_mode=blind | labeled by hand

2026-10-08 | doc_026.json | (new file) | - -> label_mode=blind | labeled by hand

2026-10-08 | doc_027.json | (new file) | - -> label_mode=blind | labeled by hand

2026-10-08 | doc_016.json | (new file) | - -> label_mode=blind | labeled by hand

2026-10-08 | doc_002.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_003.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_009.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_014.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_031.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_020.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_023.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_024.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_029.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-08 | doc_035.json | (new file) | - -> label_mode=prelabel | labeled by hand

2026-10-09 | data/manifest.csv | doc_032 | doc_type bill -> ecommerce; template_group tokopedia_electricity -> tokped_invoice; content_group doc_016 -> doc_019; gt_from doc_016 -> -; label_mode - -> prelabel | the owner found (2026-10-09) it is NOT a photo of doc_016, which the audit in docs/eval/hallucinations.md had flagged; it is a photo of doc_019's order with a buyer name printed that doc_019's label lacks, so it gets its own label (draft by claude-haiku-5-5, verified by hand; not blind: the extractor's output had been seen)
2026-10-09 | data/manifest.csv | doc_033 | doc_type bill -> receipt; template_group tokopedia_electricity -> receipt_food_and_drink; content_group doc_016 -> doc_033; gt_from doc_016 -> -; label_mode - -> prelabel | the owner found (2026-10-09) it is NOT a photo of doc_016, which the audit in docs/eval/hallucinations.md had flagged; it is a cafe drink receipt: own label (draft by claude-haiku-5-5, verified by hand; not blind)
2026-10-09 | data/manifest.csv | doc_034 | doc_type bill -> ecommerce; template_group tokopedia_electricity -> tokped_invoice; content_group doc_016 -> doc_019; gt_from doc_016 -> doc_032 | the owner found (2026-10-09) it is NOT a photo of doc_016, which the audit in docs/eval/hallucinations.md had flagged; it is the same document as doc_032 (another photo), so it takes doc_032's label
