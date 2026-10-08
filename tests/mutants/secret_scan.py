"""Mutants for tools/secret_scan.py and the Claude Code hook around it.

  python -m uv run python tools/mutation_check.py tests/mutants/secret_scan.py

Every line below breaks one rule on purpose; the tests must notice (see tools/mutation_check.py).
This file is not collected by pytest (its name does not start with `test_`).
"""

SCAN = "tools/secret_scan.py"
HOOK = ".claude/hooks/pre_commit_secret_scan.py"
T = ["tests/test_secret_scan.py"]

MUTANTS = [
    (
        SCAN,
        T,
        "env file rule ignores prod.env",
        r'r"(^|/)(\.env|[^/]*\.env)$"',
        r'r"(^|/)(\.env)$"',
    ),
    (
        SCAN,
        T,
        "the example file matches the env rule",
        r'r"(^|/)(\.env|[^/]*\.env)$"',
        r'r"(^|/)(\.env|[^/]*\.env)"',
    ),
    (SCAN, T, "the prefix b/ is kept", 'target.removeprefix("b/")', "target"),
    (
        SCAN,
        T,
        "removed lines are scanned",
        'elif raw.startswith("+") and path:',
        'elif raw.startswith(("+", "-")) and path:',
    ),
    (
        SCAN,
        T,
        "context lines do not advance",
        'elif raw.startswith(" ") and path:\n            number += 1',
        'elif raw.startswith(" ") and path:\n            number += 0',
    ),
    (
        SCAN,
        T,
        "added lines do not advance",
        '            number += 1\n        elif raw.startswith(" ")',
        '            number += 0\n        elif raw.startswith(" ")',
    ),
    (SCAN, T, "repeated characters are secrets", "or len(set(value)) == 1", "or False"),
    (SCAN, T, "variable references are secrets", 'or value.startswith("$")', "or False"),
    (
        SCAN,
        T,
        "key shape needs no digit and letter",
        "return any(c.isalpha() for c in value) and any(c.isdigit() for c in value)",
        "return True",
    ),
    (
        SCAN,
        T,
        "key shape needs only a digit",
        "return any(c.isalpha() for c in value) and any(c.isdigit() for c in value)",
        "return any(c.isdigit() for c in value)",
    ),
    (
        SCAN,
        T,
        "double report for a key shape",
        "and not is_placeholder(value) and not found:",
        "and not is_placeholder(value):",
    ),
    (SCAN, T, "binary untracked files scanned", 'if b"\\0" in data:', "if False:"),
    (
        SCAN,
        T,
        "huge untracked files scanned",
        "if path.stat().st_size > MAX_UNTRACKED_BYTES:",
        "if False:",
    ),
    (SCAN, T, "history looks at HEAD only", '"log", "--all", "-p"', '"log", "-p"'),
    (
        SCAN,
        T,
        "short env values are searched",
        "and len(value) >= 8 and",
        "and len(value) >= 0 and",
    ),
    (
        SCAN,
        T,
        "env placeholders are searched",
        "len(value) >= 8 and not is_placeholder(value)",
        "len(value) >= 8",
    ),
    (SCAN, T, "findings exit clean", "return 1 if found else 0", "return 0"),
    (
        SCAN,
        T,
        "openai key shape lost",
        r'"openai-style key": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")',
        r'"openai-style key": re.compile(r"\bsk_[A-Za-z0-9_\-]{20,}")',
    ),
    (HOOK, T, "git.exe is not git", '("git", "git.exe")', '("git",)'),
    (
        HOOK,
        T,
        "git add does not widen the scan",
        'any(name == "add" for name, _, _ in calls)',
        "False",
    ),
    (
        HOOK,
        T,
        "commit -a does not widen the scan",
        'arg == "--all" or (arg.startswith("-") and not arg.startswith("--") and "a" in arg)',
        'arg == "--all"',
    ),
    (
        HOOK,
        T,
        "non-object payload crashes",
        "if not isinstance(payload, dict):\n        return 0",
        "if False:\n        return 0",
    ),
    (HOOK, T, "findings do not block", "if done.returncode == 1:", "if done.returncode == 99:"),
]
