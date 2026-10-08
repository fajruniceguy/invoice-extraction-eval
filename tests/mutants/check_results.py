"""Mutants for scripts/check_results.py.

  python -m uv run python tools/mutation_check.py tests/mutants/check_results.py

Every entry breaks one rule on purpose; the tests must notice (see tools/mutation_check.py).
This file is not collected by pytest (its name does not start with `test_`).
"""

P = "scripts/check_results.py"
T = ["tests/test_check_results.py"]

MUTANTS = [
    (P, T, "the committed git sha is not reused", "git_sha=sha,", 'git_sha="unknown",'),
    (P, T, "the committed time is not reused", "now=scored_at,", "now=None,"),
    (P, T, "the dirty flag is not reused", "git_dirty=dirty,", "git_dirty=False,"),
    (P, T, "differing files are not reported", "if fresh[name] != committed[name]", "if False"),
    (P, T, "stray committed files are not reported", "committed.keys() - fresh.keys()", "set()"),
    (P, T, "missing committed files are not reported", "fresh.keys() - committed.keys()", "set()"),
    (P, T, "a differing result exits 0", "            status = 1\n", "            status = 0\n"),
    (
        P,
        T,
        "nothing to check exits 0",
        "        return 2\n    status = 0",
        "        return 0\n    status = 0",
    ),
    (
        P,
        T,
        "an unreadable result exits 1",
        "print(str(exc), file=sys.stderr)\n            return 2",
        "print(str(exc), file=sys.stderr)\n            return 1",
    ),
]
