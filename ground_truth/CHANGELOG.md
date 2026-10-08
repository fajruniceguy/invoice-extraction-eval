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
