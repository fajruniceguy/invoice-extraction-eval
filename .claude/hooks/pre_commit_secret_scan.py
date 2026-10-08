"""Claude Code PreToolUse hook (matcher: Bash): stop a `git commit` that would add a secret.

Claude Code sends the pending tool call as JSON on stdin. If the Bash command is a `git commit`,
the scanner (tools/secret_scan.py) looks at what the commit would add and, on a finding, this hook
exits 2: Claude Code then blocks the command and shows stderr to Claude. Any other command passes.

The hook runs BEFORE the command, so in `git add x && git commit` the index does not hold x yet:
when the command also stages something (`git add`, `commit -a`) the whole working tree is scanned
(index, unstaged edits, untracked files). Standard library only.

It exists because a real key was nearly committed in `.env.example` on 2026-10-08.
"""

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

SCANNER = Path(__file__).resolve().parents[2] / "tools" / "secret_scan.py"
SEGMENTS = re.compile(r"&&|\|\||[;|\n]")
OPTIONS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}


def git_calls(command: str) -> list[tuple[str, list[str], str | None]]:
    """(subcommand, its arguments, the -C directory) for every `git ...` in a shell command."""
    calls = []
    for segment in SEGMENTS.split(command):
        try:
            words = shlex.split(segment)
        except ValueError:
            words = segment.split()
        if not words or Path(words[0]).name not in ("git", "git.exe"):
            continue
        directory = None
        index = 1
        while index < len(words) and words[index].startswith("-"):
            if words[index] == "-C" and index + 1 < len(words):
                directory = words[index + 1]
            index += 2 if words[index] in OPTIONS_WITH_VALUE else 1
        if index < len(words):
            calls.append((words[index], words[index + 1 :], directory))
    return calls


def plan(command: str) -> tuple[bool, bool, str | None]:
    """(is it a commit, must the working tree be scanned too, the -C directory)."""
    calls = git_calls(command)
    commits = [(args, directory) for name, args, directory in calls if name == "commit"]
    if not commits:
        return False, False, None
    stages = any(name == "add" for name, _, _ in calls) or any(
        arg == "--all" or (arg.startswith("-") and not arg.startswith("--") and "a" in arg)
        for args, _ in commits
        for arg in args
    )
    return True, stages, commits[0][1]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return 0  # not a call we understand: never block on a malformed hook payload
    if not isinstance(payload, dict):
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    is_commit, whole_tree, directory = plan(command)
    if not is_commit:
        return 0
    repo = Path(payload.get("cwd") or ".")
    if directory:
        repo = repo / directory
    mode = "--worktree" if whole_tree else "--staged"
    done = subprocess.run(
        [sys.executable, str(SCANNER), mode, "--repo", str(repo)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if done.returncode == 1:
        print(
            "Blocked: this commit would add what looks like a secret.\n"
            f"{done.stdout.strip()}\n"
            "Remove it (or move it to the gitignored .env), stage again, and retry. If it is a "
            "false positive, make the value an obvious placeholder (your-key-here).",
            file=sys.stderr,
        )
        return 2
    return 0  # clean, or the scan itself could not run (not a repo): do not block


if __name__ == "__main__":
    sys.exit(main())
