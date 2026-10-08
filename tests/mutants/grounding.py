"""Mutants for G1: the ambiguous-but-consistent number is soft, anything else is not.

  python -m uv run python tools/mutation_check.py tests/mutants/grounding.py

Every entry breaks one rule on purpose; the tests must notice (see tools/mutation_check.py).
This file is not collected by pytest (its name does not start with `test_`).
"""

G = "src/docextract/validate/grounding.py"
C = "src/docextract/validate/codes.py"
F = "src/docextract/confidence/formula.py"
T = ["tests/test_validate_grounding.py", "tests/test_validate_document.py"]
TC = ["tests/test_validate_grounding.py", "tests/test_confidence.py"]

MUTANTS = [
    (
        G,
        T,
        "any ambiguous span is consistent",
        'result.status == "ambiguous" and value in result.candidates',
        'result.status == "ambiguous"',
    ),
    (
        G,
        T,
        "ambiguous never counts",
        'result.status == "ambiguous" and value in result.candidates',
        "False",
    ),
    (
        G,
        T,
        "the candidate test is not exact",
        "value in result.candidates",
        "any(abs(value - c) < 10 for c in result.candidates)",
    ),
    (
        G,
        T,
        "ambiguous beats grounded",
        '            return "grounded"\n        if result.status',
        "            ambiguous = True\n        if result.status",
    ),
    (
        G,
        T,
        "ambiguity is hard for scalars",
        "            result = with_document_reason(result, name, EVIDENCE_AMBIGUOUS)",
        "            result = with_document_reason(result, name, EVIDENCE_NOT_GROUNDED)",
    ),
    (
        G,
        T,
        "ambiguity is dropped for scalars",
        '        elif grounding == "ambiguous":\n'
        "            result = with_document_reason(result, name, EVIDENCE_AMBIGUOUS)",
        '        elif grounding == "ambiguous":\n            pass',
    ),
    (
        G,
        T,
        "ambiguity is hard for rows",
        "            result = _flag_cells(result, index, ambiguous)\n"
        "            result = with_row_reason(result, index, EVIDENCE_AMBIGUOUS)",
        "            result = _flag_cells(result, index, ambiguous)\n"
        "            result = with_row_reason(result, index, EVIDENCE_NOT_GROUNDED)",
    ),
    (
        G,
        T,
        "ambiguous row cells are not flagged",
        "            result = _flag_cells(result, index, ambiguous)\n",
        "",
    ),
    (
        G,
        T,
        "a hard cell no longer beats an ambiguous one",
        "        if ungrounded:\n            result = _flag_cells(result, index, ungrounded)",
        "        if ungrounded and not ambiguous:\n"
        "            result = _flag_cells(result, index, ungrounded)",
    ),
    (
        G,
        T,
        "dates become soft too",
        "        grounded = isinstance(value, date) and _date_grounded(quote, value, day_first)",
        "        grounded = True",
    ),
    (
        G,
        T,
        "a non-decimal money value is accepted",
        '        if not isinstance(value, Decimal):\n            return "ungrounded"',
        '        if not isinstance(value, Decimal):\n            return "grounded"',
    ),
    (
        C,
        TC,
        "the new code is hard",
        "frozenset({DUE_DATE_BEFORE_DOCUMENT_DATE, EVIDENCE_NOT_GROUNDED})",
        "frozenset({DUE_DATE_BEFORE_DOCUMENT_DATE, EVIDENCE_NOT_GROUNDED, EVIDENCE_AMBIGUOUS})",
    ),
    (
        C,
        TC,
        "the new code is not in the vocabulary",
        "    EVIDENCE_NOT_GROUNDED,\n    EVIDENCE_AMBIGUOUS,\n"
        "    DESCRIPTION_NOT_VERBATIM,\n    CONFIDENCE_BELOW_THRESHOLD,\n)",
        "    EVIDENCE_NOT_GROUNDED,\n"
        "    DESCRIPTION_NOT_VERBATIM,\n    CONFIDENCE_BELOW_THRESHOLD,\n)",
    ),
    (
        F,
        T,
        "the new code has the hard cap",
        "    EVIDENCE_AMBIGUOUS: SHAPE_CAP,",
        "    EVIDENCE_AMBIGUOUS: ARITHMETIC_CAP,",
    ),
    (
        F,
        T,
        "the new code has no cap value",
        "    EVIDENCE_AMBIGUOUS: SHAPE_CAP,",
        "    EVIDENCE_AMBIGUOUS: 1.0,",
    ),
]
