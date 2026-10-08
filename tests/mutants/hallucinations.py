"""Mutants for evalkit/hallucinations.py and its script.

  python -m uv run python tools/mutation_check.py tests/mutants/hallucinations.py

Every entry breaks one rule on purpose; the tests must notice (see tools/mutation_check.py).
This file is not collected by pytest (its name does not start with `test_`).
"""

H = "src/evalkit/hallucinations.py"
S = "scripts/analyze_hallucinations.py"
T = ["tests/test_hallucinations.py"]

MUTANTS = [
    (
        H,
        T,
        "the gap boundary is exclusive",
        "return owner.rate - member.rate >= LINEAGE_GAP",
        "return owner.rate - member.rate > LINEAGE_GAP",
    ),
    (H, T, "the gap is 0.6", "LINEAGE_GAP = Fraction(1, 2)", "LINEAGE_GAP = Fraction(3, 5)"),
    (
        H,
        T,
        "the owner needs no minimum",
        "if owner.labeled < LINEAGE_MIN_FIELDS or member.labeled < LINEAGE_MIN_FIELDS:",
        "if member.labeled < LINEAGE_MIN_FIELDS:",
    ),
    (
        H,
        T,
        "the member needs no minimum",
        "if owner.labeled < LINEAGE_MIN_FIELDS or member.labeled < LINEAGE_MIN_FIELDS:",
        "if owner.labeled < LINEAGE_MIN_FIELDS:",
    ),
    (H, T, "the minimum is 4", "LINEAGE_MIN_FIELDS = 3", "LINEAGE_MIN_FIELDS = 4"),
    (
        H,
        T,
        "the gap sign is reversed",
        "owner.rate - member.rate >=",
        "member.rate - owner.rate >=",
    ),
    (
        H,
        T,
        "abstentions count as hallucinations",
        'if score_field(name, None, field).outcome != "should_be_absent":',
        'if score_field(name, None, field).outcome == "correct":',
    ),
    (
        H,
        T,
        "labeled values count as absent",
        "if is_unlabeled(expected) or expected is not None:",
        "if is_unlabeled(expected):",
    ),
    (H, T, "the scalar denominator is dropped", "            absent_fields += 1\n", ""),
    (
        H,
        T,
        "cells with a label value count",
        "if cell not in pair.cells or getattr(labeled_row, cell) is not None:",
        "if cell not in pair.cells:",
    ),
    (H, T, "the cell denominator is dropped", "                null_cells += 1\n", ""),
    (
        H,
        T,
        "a missing cell value counts",
        "                if value is None:\n                    continue\n",
        "",
    ),
    (
        H,
        T,
        "codes stay in the note",
        'return " | ".join(reason for reason in reasons if reason not in REASON_CODES)',
        'return " | ".join(reasons)',
    ),
    (
        H,
        T,
        "the note keeps the codes only",
        "return tuple(reason for reason in reasons if reason in REASON_CODES)",
        "return tuple(reasons)",
    ),
    (H, T, "raw_also is always true", 'raw_also=raw.status == "extracted",', "raw_also=True,"),
    (H, T, "the owner is the document itself", '"owner": resolved.owner,', '"owner": doc_id,'),
    (
        H,
        T,
        "the committed report holds the quote",
        "                    _short(i.note),\n",
        "                    _short(i.note + i.quote),\n",
    ),
    (
        H,
        T,
        "the review list drops the quote",
        "            f\"- its quote: {item.quote or '(none)'}\",",
        '            "- its quote: (omitted)",',
    ),
    (
        H,
        T,
        "the review list asks about every copy",
        "for member in group.suspects:",
        "for member in group.members:",
    ),
    (H, T, "bands overlap at 0.9", '(">= 0.90", 0.9, 1.01),', '(">= 0.90", 0.91, 1.01),'),
    (
        H,
        T,
        "notes are not trimmed",
        'lines += ["## Reading", "", notes.strip(), ""]',
        'lines += ["## Reading", "", notes, ""]',
    ),
    (
        S,
        T,
        "the review path is never refused",
        "if args.review is not None and inside_tracked_folder(args.review, root):",
        "if False:",
    ),
    (
        S,
        T,
        "results is not a tracked folder",
        'TRACKED_FOLDERS = ("docs", "results",',
        'TRACKED_FOLDERS = ("docs",',
    ),
    (
        S,
        T,
        "a missing run exits 0",
        '        print(f"{args.run_dir}: not a directory", file=sys.stderr)\n        return 2',
        '        print(f"{args.run_dir}: not a directory", file=sys.stderr)\n        return 0',
    ),
]
