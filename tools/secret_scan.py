"""Find secrets before they are committed, or after they were.

  python tools/secret_scan.py --staged        what a `git commit` would add (the index)
  python tools/secret_scan.py --worktree      staged + unstaged changes + untracked files
  python tools/secret_scan.py --history       every commit of every ref
  python tools/secret_scan.py --env-values    the values in the real .env, searched in the history

Standard library only and no project imports, so a hook or a CI job can run it with any Python.
Only ADDED lines are scanned. A finding names the place and the rule and shows the line with the
secret replaced by `[REDACTED n chars]`: the value itself is never printed, and `--env-values`
prints only the variable NAME. Exit 0 clean, 1 findings, 2 usage or git error.

It exists because a real key was nearly committed in `.env.example`; it does not replace rotating
a key that was ever pushed.
"""

import argparse
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

KEY_SHAPES: dict[str, re.Pattern[str]] = {
    "openai-style key": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    "google api key": re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    "github token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "aws access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
# NAME = value, where NAME says it is a secret and the value looks like one: 16+ characters of a
# secret alphabet (no slash, dot or colon, so paths and URLs are not values), see `secret_shaped`.
# The name part is BOUNDED: an unbounded prefix made a long line quadratic (a 100 KB line stalled).
ASSIGNMENT = re.compile(
    r"(?P<name>[A-Za-z0-9_.\-]{0,64}(?:key|token|secret|password)[A-Za-z0-9_.\-]{0,64})"
    r"\s*[:=]\s*[\"']?(?P<value>[A-Za-z0-9_+=\-]{16,})",
    re.IGNORECASE,
)
KEYWORD = re.compile(r"key|token|secret|password", re.IGNORECASE)
NAME_ROOM = 64  # characters of NAME before the keyword that are looked at
VALUE_ROOM = 300  # characters after the keyword: the rest of the name, the separator, the value
MAX_KEYWORDS_PER_LINE = 200
PLACEHOLDER_WORDS = (
    "your", "example", "placeholder", "changeme", "xxxx", "<", "...", "${",
    "paste", "replace", "insert", "dummy", "sample", "here",
)  # fmt: skip
SENSITIVE_NAME = re.compile(r"key|token|secret|password", re.IGNORECASE)
ENV_FILE = re.compile(r"(^|/)(\.env|[^/]*\.env)$")
MAX_UNTRACKED_BYTES = 1_000_000
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class ScanError(Exception):
    """git failed, or the arguments make no sense."""


@dataclass(frozen=True)
class Finding:
    where: str  # "path:line" or "commit abcdef1234 path:line"
    rule: str
    excerpt: str  # the line with the secret removed

    def __str__(self) -> str:
        return f"{self.where}: {self.rule}: {self.excerpt}"


def is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return (
        any(word in lowered for word in PLACEHOLDER_WORDS)
        or len(set(value)) == 1
        or value.startswith("$")
    )


def secret_shaped(value: str) -> bool:
    """A letter and a digit: what separates a key from a long identifier in code. Only the
    pattern rule needs it; a value taken from the real .env is searched whatever it looks like."""
    return any(c.isalpha() for c in value) and any(c.isdigit() for c in value)


def redact(line: str, start: int, end: int) -> str:
    text = f"{line[:start]}[REDACTED {end - start} chars]{line[end:]}".strip()
    return text if len(text) <= 100 else text[:99] + "…"


def scan_line(line: str) -> list[tuple[str, str]]:
    """(rule, line with the secret removed) for every secret found in one line."""
    found: list[tuple[str, str]] = []
    for rule, pattern in KEY_SHAPES.items():
        for match in pattern.finditer(line):
            found.append((rule, redact(line, match.start(), match.end())))
    for match in assignments(line):
        value = match.group("value")
        if secret_shaped(value) and not is_placeholder(value) and not found:
            found.append(
                (f"secret assigned to {match.group('name')}", redact(line, *match.span("value")))
            )
    return found


def assignments(line: str) -> Iterator[re.Match[str]]:
    """`NAME = value` matches, looked for only in a small window around each keyword, so the cost
    is linear in the line however long it is (a minified file is one enormous line)."""
    seen: set[int] = set()
    for keyword in islice(KEYWORD.finditer(line), MAX_KEYWORDS_PER_LINE):
        window = (max(0, keyword.start() - NAME_ROOM), min(len(line), keyword.end() + VALUE_ROOM))
        for match in ASSIGNMENT.finditer(line, *window):
            if match.start("name") not in seen:
                seen.add(match.start("name"))
                yield match


def scan_diff(diff: str, prefix: str = "") -> Iterator[Finding]:
    """Findings in the ADDED lines of a unified diff (`-U0` or any context)."""
    path = ""
    number = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            path = target.removeprefix("b/")  # a deleted file ("/dev/null") has no added lines
            if ENV_FILE.search(path):  # ".env.example" does not end in ".env": it never matches
                yield Finding(f"{prefix}{path}", "env file added", "a .env file must stay local")
        elif raw.startswith("@@"):
            hunk = HUNK.match(raw)
            number = int(hunk.group(1)) if hunk else 0
        elif raw.startswith("+") and path:
            for rule, excerpt in scan_line(raw[1:]):
                yield Finding(f"{prefix}{path}:{number}", rule, excerpt)
            number += 1
        elif raw.startswith(" ") and path:
            number += 1


def git(repo: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise ScanError(f"cannot run git: {exc}") from exc
    if done.returncode != 0:
        raise ScanError(f"git {' '.join(args)}: {done.stderr.strip() or 'failed'}")
    return done.stdout


def staged_findings(repo: Path) -> list[Finding]:
    return list(scan_diff(git(repo, "diff", "--cached", "-U0", "--no-color", "--diff-filter=AM")))


def worktree_findings(repo: Path) -> list[Finding]:
    """What a `git add -A && git commit` would take: the index, unstaged edits, untracked files."""
    found = staged_findings(repo)
    found += scan_diff(git(repo, "diff", "-U0", "--no-color", "--diff-filter=AM"))
    listing = git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    for name in filter(None, listing.split("\0")):
        path = repo / name
        if ENV_FILE.search(name):
            found.append(Finding(name, "env file added", "a .env file must stay local"))
            continue
        try:
            if path.stat().st_size > MAX_UNTRACKED_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue  # binary
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            found += [Finding(f"{name}:{number}", r, e) for r, e in scan_line(line)]
    return found


def history_text(repo: Path) -> str:
    return git(
        repo, "log", "--all", "-p", "-U0", "--no-color", "--format=commit %H", "--diff-filter=AM"
    )


def history_findings(repo: Path) -> list[Finding]:
    found: list[Finding] = []
    commit = ""
    block: list[str] = []

    def flush() -> None:
        if commit:
            found.extend(scan_diff("\n".join(block), prefix=f"commit {commit[:10]} "))

    for line in history_text(repo).splitlines():
        if line.startswith("commit ") and len(line) == 47:
            flush()
            commit, block = line[7:], []
        else:
            block.append(line)
    flush()
    return found


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        name, _, value = text.partition("=")
        values[name.strip()] = value.strip().strip("\"'")
    return values


def env_secrets(env_file: Path) -> dict[str, str]:
    """The real .env's secret-looking entries (by NAME) with a value worth searching for."""
    return {
        name: value
        for name, value in read_env(env_file).items()
        if SENSITIVE_NAME.search(name) and len(value) >= 8 and not is_placeholder(value)
    }


def env_value_findings(repo: Path, secrets: dict[str, str]) -> list[Finding]:
    """Where those values appear in the history. They are used in memory and never printed; a
    finding carries the variable NAME and the commit only."""
    found: list[Finding] = []
    commit = ""
    block: list[str] = []

    def flush() -> None:
        text = "\n".join(block)
        for name, value in secrets.items():
            if commit and value in text:
                found.append(
                    Finding(
                        f"commit {commit[:10]}", f"value of {name} from .env", "(value not shown)"
                    )
                )

    for line in history_text(repo).splitlines():
        if line.startswith("commit ") and len(line) == 47:
            flush()
            commit, block = line[7:], []
        else:
            block.append(line)
    flush()
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find secrets in a git repository.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="the index (what a commit adds)")
    mode.add_argument("--worktree", action="store_true", help="index + unstaged + untracked")
    mode.add_argument("--history", action="store_true", help="every commit of every ref")
    mode.add_argument("--env-values", action="store_true", help="the real .env values, in history")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="the repository")
    parser.add_argument("--env-file", type=Path, default=None, help="default: <repo>/.env")
    args = parser.parse_args(argv)
    try:
        if args.staged:
            found = staged_findings(args.repo)
        elif args.worktree:
            found = worktree_findings(args.repo)
        elif args.history:
            found = history_findings(args.repo)
        else:
            env_file = args.env_file or args.repo / ".env"
            if not env_file.is_file():
                print(f"{env_file}: no such file", file=sys.stderr)
                return 2
            secrets = env_secrets(env_file)
            print(f"secret_scan: checking {len(secrets)} value(s): {', '.join(sorted(secrets))}")
            found = env_value_findings(args.repo, secrets)
    except (ScanError, OSError) as exc:
        print(f"secret_scan: {exc}", file=sys.stderr)
        return 2
    for finding in found:
        print(finding)
    scope = "staged changes" if args.staged else "history" if args.history else "working tree"
    if args.env_values:
        scope = "history against the values in .env"
    print(f"secret_scan: {len(found)} finding(s) in the {scope}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
