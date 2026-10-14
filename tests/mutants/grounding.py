"""Mutants for G1 (the ambiguous-but-consistent number is soft, anything else is not) and G3
(a quantity printed with a multiplication sign glued on, "x1" / "2x", is grounded).

  python -m uv run python tools/mutation_check.py tests/mutants/grounding.py

Every entry breaks one rule on purpose; the tests must notice (see tools/mutation_check.py).
This file is not collected by pytest (its name does not start with `test_`).
"""

G = "src/docextract/validate/grounding.py"
C = "src/docextract/validate/codes.py"
F = "src/docextract/confidence/formula.py"
T = ["tests/test_validate_grounding.py", "tests/test_validate_document.py"]
TG = ["tests/test_validate_grounding.py"]
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
    # G3: a quantity printed with a multiplication sign glued on ("x1", "2x")
    (
        G,
        TG,
        "the sign is read for every number, not only a quantity",
        "    for token in _quantity_candidates(quote) if quantity else _number_candidates(quote):",
        "    for token in _quantity_candidates(quote):",
    ),
    (
        G,
        TG,
        "the sign is never read",
        "            cell: _number_grounding(\n"
        '                quote, getattr(row, cell), currency, quantity=cell == "quantity"\n'
        "            )",
        "            cell: _number_grounding(quote, getattr(row, cell), currency, quantity=False)",
    ),
    (
        G,
        TG,
        "only a price is read through a sign",
        'quantity=cell == "quantity"',
        'quantity=cell != "quantity"',
    ),
    (
        G,
        TG,
        "a leading sign is not read",
        r'r"^(?:[x×](\d[\d.,]*)|(\d[\d.,]*)[x×])$"',
        r'r"^(?:(\d[\d.,]*)[x×])$"',
    ),
    (
        G,
        TG,
        "a trailing sign is not read",
        r'r"^(?:[x×](\d[\d.,]*)|(\d[\d.,]*)[x×])$"',
        r'r"^(?:[x×](\d[\d.,]*))$"',
    ),
    (
        G,
        TG,
        "the sign is case-sensitive",
        r'[x×])$", re.IGNORECASE)',
        r'[x×])$")',
    ),
    (
        G,
        TG,
        "the multiplication sign character is not read",
        r'r"^(?:[x×](\d[\d.,]*)|(\d[\d.,]*)[x×])$"',
        r'r"^(?:[x](\d[\d.,]*)|(\d[\d.,]*)[x])$"',
    ),
    (
        G,
        TG,
        "any letters around a number are a sign",
        r'r"^(?:[x×](\d[\d.,]*)|(\d[\d.,]*)[x×])$"',
        r'r"^(?:[a-z×]+(\d[\d.,]*)|(\d[\d.,]*)[a-z×]+)$"',
    ),
    (
        G,
        TG,
        "the token need not be only a sign and a number",
        r'r"^(?:[x×](\d[\d.,]*)|(\d[\d.,]*)[x×])$"',
        r'r"^(?:[x×](\d[\d.,]*)|(\d[\d.,]*)[x×])"',
    ),
    (
        G,
        TG,
        "the number is not offered, the token is",
        "            candidates.append(match.group(1) or match.group(2))",
        "            candidates.append(token)",
    ),
    (
        G,
        TG,
        "a signed number is taken as its digits and nothing else",
        "            candidates.append(match.group(1) or match.group(2))",
        "            candidates.append(\n"
        '                (match.group(1) or match.group(2)).replace(".", "").replace(",", "")\n'
        "            )",
    ),
]
