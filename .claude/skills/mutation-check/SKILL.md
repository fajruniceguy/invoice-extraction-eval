---
name: mutation-check
description: Mutation-test new or changed logic in this repo - break it on purpose, in an isolated copy, and confirm a test fails. Use after writing logic or its tests, before committing, and whenever someone asks whether the tests really test anything.
---

# Mutation check

Passing tests prove little until you have seen them fail. In this project every milestone ended with
a mutation pass, and it found vacuous tests more than once (`assert ... or True`, a test world with
equal bin sizes that could not tell a weighted mean from a plain one, an app test skipped because it
was marked `slow`). `tools/mutation_check.py` is that procedure as a tool.

## Use

1. Write the mutants as a Python file, e.g. `tests/mutants/<topic>.py` (see `tests/mutants/secret_scan.py`):

   ```python
   MUTANTS = [
       # (file, tests to run, name, text to replace (must occur exactly once), replacement)
       ("src/evalkit/calibration.py", ["tests/test_calibration.py"],
        "target is exclusive", "point.selective >= target", "point.selective > target"),
   ]
   ```

2. Run it with the project interpreter (not a bare `python`):

   ```
   python -m uv run python tools/mutation_check.py tests/mutants/<topic>.py
   ```

   Exit 0: every mutant killed. 1: a survivor or a refused mutant. 2: the baseline is red.

3. Read each `SURVIVED` line. It is one of: a missing test (write it, then run again), an equivalent
   mutant (the change cannot alter behaviour: restructure the code so the line has no equivalent, as
   was done for the `!=`-then-`<` comparison in `evalkit/sweep.py`), or dead code (delete it).

## Which mutants to write

One per rule, aimed where tests tend to be soft: flip a comparison (`>=` to `>`) and an off-by-one at a
boundary; drop a filter or a branch; swap an order (ascending to descending, replace to append); change a
constant or a default; skip a normalization (`strip`, `casefold`); make an exit code lie. Aim for the
behaviour you would be embarrassed to ship broken, not for the number of mutants.

## Rules the tool enforces (each cost us time once)

- It edits an ISOLATED COPY, never the working tree. A running Streamlit app imports the same files; a
  reload during an in-place pass runs broken code.
- A green baseline run comes first. A harness that cannot run the tests (wrong interpreter, a file the
  tests read that was not copied) reports every mutant as killed.
- A mutant whose text is missing or ambiguous is refused, not run.
- Slow-marked tests are included (`-m "slow or not slow"`): the app tests are slow-marked.

## Report

Say how many mutants, how many survived the first time, what the survivors revealed, and that all are
killed now. Never report a pass you did not run.
