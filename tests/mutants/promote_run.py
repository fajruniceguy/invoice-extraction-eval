"""Mutants for scripts/promote_run.py.

  python -m uv run python tools/mutation_check.py tests/mutants/promote_run.py

Every entry breaks one rule on purpose; the tests must notice (see tools/mutation_check.py).
This file is not collected by pytest (its name does not start with `test_`).
"""

P = "scripts/promote_run.py"
T = ["tests/test_promote_run.py"]

MUTANTS = [
    (P, T, "a stopped run is promoted", "if run.stopped is not None:", "if False:"),
    (P, T, "incomplete runs are always allowed", "if not allow_incomplete:", "if False:"),
    (P, T, "allow-incomplete is ignored", "if not allow_incomplete:", "if True:"),
    (P, T, "an existing result is overwritten", "if target.exists():", "if False:"),
    (P, T, "the score of another run is accepted", 'or scored.get("run_id") != run.run_id', ""),
    (P, T, "a missing score file is accepted", "if missing:", "if False:"),
    (
        P,
        T,
        "records are not matched to statuses",
        "if records != sorted(run.statuses):",
        "if False:",
    ),
    (P, T, "the secret scan is skipped", "check_secrets(run_dir, paths, scanner)", "pass"),
    (
        P,
        T,
        "the scan stops at the first line",
        "for number, line in enumerate(text.splitlines(), start=1):",
        "for number, line in list(enumerate(text.splitlines(), start=1))[:1]:",
    ),
    (P, T, "run.json is not copied", 'paths = [Path("run.json")]', "paths = []"),
    (
        P,
        T,
        "the score folder is not copied",
        'path.relative_to(run_dir) for path in score.rglob("*") if path.is_file()',
        'path.relative_to(run_dir) for path in score.glob("*.md") if path.is_file()',
    ),
    (P, T, "the folder name is not checked", "or run.run_id != run_dir.name", ""),
    (
        P,
        T,
        "refusal exits 0",
        '        print(f"refused: {exc}", file=sys.stderr)\n        return 1',
        '        print(f"refused: {exc}", file=sys.stderr)\n        return 0',
    ),
    (
        P,
        T,
        "a missing folder exits 1",
        '        print(f"{args.run_dir}: not a directory", file=sys.stderr)\n        return 2',
        '        print(f"{args.run_dir}: not a directory", file=sys.stderr)\n        return 1',
    ),
]
