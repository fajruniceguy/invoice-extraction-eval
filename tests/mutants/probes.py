"""Mutants for the S4 probe set (evalkit/probes.py and evalkit/probe_report.py).

  python -m uv run python tools/mutation_check.py tests/mutants/probes.py --copy LIST

LIST = the tool's default copy plus what the tests read: src,tools,tests,scripts,docs,
ground_truth,.claude,pyproject.toml,.gitignore,data  (one argument, no spaces; data/probes and
data/pages are needed by the regeneration test).

Every entry breaks one rule on purpose; tests/test_probes.py must notice.
This file is not collected by pytest (its name does not start with `test_`).
"""

P = "src/evalkit/probes.py"
R = "src/evalkit/probe_report.py"
T = ["tests/test_probes.py"]

MUTANTS = [
    (P, T, "no padding around the text box", "PAD = 3", "PAD = 0"),
    (P, T, "nothing printed is never refused", "MIN_INK = 20", "MIN_INK = 0"),
    (P, T, "a box needs a lot of ink", "MIN_INK = 20", "MIN_INK = 100000"),
    (P, T, "every pixel counts as ink", "INK_DELTA = 40", "INK_DELTA = -1"),
    (P, T, "no pixel counts as ink", "INK_DELTA = 40", "INK_DELTA = 1000"),
    (
        P,
        T,
        "the fill is always white",
        "    colour = _ring(image, rect)\n",
        "    colour = (255, 255, 255)\n",
    ),
    (
        P,
        T,
        "a number is erased on the first page only",
        '    if place == "all":\n        return tuple(range(total))',
        '    if place == "all":\n        return (0,)',
    ),
    (
        P,
        T,
        "totals are looked for on the first page",
        'return (0,) if place == "first" else (total - 1,)',
        "return (0,)",
    ),
    (
        P,
        T,
        "a missing value is accepted",
        "                    if len(found) != 1:",
        "                    if len(found) > 1:",
    ),
    (
        P,
        T,
        "two matches are accepted",
        "                    if len(found) != 1:",
        "                    if len(found) < 1:",
    ),
    (
        P,
        T,
        "the dropped page is kept",
        "        images = images[:-1]",
        "        images = images[:]",
    ),
    (
        P,
        T,
        "the rows of the dropped page stay in the label",
        "        keep = sum(plan[:-1])",
        "        keep = sum(plan)",
    ),
    (
        P,
        T,
        "the currency stays labeled after a drop",
        '        label["currency"] = UNLABELED',
        "        pass",
    ),
    (
        P,
        T,
        "erased fields keep their value",
        "        label[name] = None\n    if spec.drop_last_page:",
        "        pass\n    if spec.drop_last_page:",
    ),
    (
        P,
        T,
        "a control is not compared with the dataset",
        "    if (probe.spec.is_control or probe.spec.drop_last_page) and hashes != expected:",
        "    if probe.spec.drop_last_page and hashes != expected:",
    ),
    (
        P,
        T,
        "a drop is compared with all the pages",
        "    expected = theirs[: len(hashes)] if probe.spec.drop_last_page else theirs",
        "    expected = theirs",
    ),
    (
        P,
        T,
        "a changed source PDF is not noticed",
        "            if mine.read_bytes() != theirs.read_bytes():",
        '            if mine.read_bytes() == b"":',
    ),
    (
        P,
        T,
        "a field the layout lacks is erased anyway",
        "        if not _printed(invoice, name):\n            raise ProbeError(",
        "        if False:\n            raise ProbeError(",
    ),
    (
        R,
        T,
        "an abstention counts as absent",
        '{"correct": ABSENT, "abstained": ABSTAINED}',
        '{"correct": ABSENT, "abstained": ABSENT}',
    ),
    (
        R,
        T,
        "a value is never recovered",
        "compare_field(\n"
        "                    name, _typed(name, original), validated_field.value\n"
        "                ).matched",
        "False",
    ),
    (
        R,
        T,
        "the control read is always yes",
        "control_read = score_field(\n"
        "                    name, _typed(name, original), getattr(control.validated, name)\n"
        "                ).correct",
        "control_read = True",
    ),
    (
        R,
        T,
        "the interval's lower end is the point estimate",
        "lower = 0.0 if k == 0 else solve(upper_tail, alpha / 2, True)",
        "lower = k / n",
    ),
    (
        R,
        T,
        "the interval is one-sided",
        "upper = 1.0 if k == n else solve(lower_tail, alpha / 2, False)",
        "upper = 1.0 if k == n else solve(lower_tail, alpha, False)",
    ),
    (
        R,
        T,
        "the model's answer is scored as the system's",
        "            raw = _outcome(name, getattr(record.parsed, name))",
        "            raw = returned",
    ),
    (
        R,
        T,
        "a probe that did not run is counted",
        '        if record.status != "ok" or record.validated is None or record.parsed is None:\n'
        "            not_ok.append(probe_id)\n            continue",
        "        if False:\n            continue",
    ),
    (
        R,
        T,
        "the none sentence always shows",
        "            if returned_x\n            else",
        "            if False\n            else",
    ),
]
