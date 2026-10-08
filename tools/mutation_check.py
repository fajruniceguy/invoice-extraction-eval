"""Mutation testing the way this project does it: break the code on purpose, expect a test to fail.

  python -m uv run python tools/mutation_check.py MUTANTS.py [--root .] [--copy src,tests,...]

MUTANTS.py defines `MUTANTS = [(path, tests, name, old, new), ...]`: in the file `path`, replace the
text `old` (which must occur exactly once) by `new`, run the pytest `tests` (a path or a list of
paths), and expect a failure. A mutant that no test notices SURVIVED and is a hole in the tests.

Three rules, each learned the hard way here:
  * Mutants are applied to an ISOLATED COPY of the repository, never to the working tree: a running
    Streamlit app imports the very files a mutation edits, and a reload mid-pass would run
    broken code.
  * A baseline run of the tests must be green first. Without it a harness that cannot run the tests
    at all (wrong interpreter, a missing file the tests read) reports every mutant as "killed".
  * A mutant whose `old` text is missing or ambiguous is REFUSED, not run: it would test nothing.

The tests run with `-m "slow or not slow"`, because slow-marked tests are tests too. Exit 0 every
mutant killed, 1 a survivor or a refused mutant, 2 the baseline is red or the arguments are wrong.
"""

import argparse
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_COPY = (
    "src",
    "tools",
    "tests",
    "scripts",
    "docs",
    "ground_truth",
    ".claude",
    "pyproject.toml",
    ".gitignore",
)
PYTEST = ("-q", "-x", "-p", "no:cacheprovider", "-m", "slow or not slow")


class MutationError(Exception):
    """The mutants file or the arguments cannot be used."""


@dataclass(frozen=True)
class Mutant:
    path: str
    tests: tuple[str, ...]
    name: str
    old: str
    new: str


@dataclass(frozen=True)
class Outcome:
    name: str
    verdict: str  # killed | SURVIVED | REFUSED
    detail: str = ""


@dataclass(frozen=True)
class Report:
    baseline_ok: bool
    baseline_tail: str
    outcomes: list[Outcome]

    @property
    def bad(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.verdict != "killed"]


def load_mutants(file: Path) -> list[Mutant]:
    try:
        namespace = runpy.run_path(str(file))
    except OSError as exc:
        raise MutationError(f"{file}: cannot be read ({exc})") from exc
    if "MUTANTS" not in namespace:
        raise MutationError(f"{file}: defines no MUTANTS list")
    mutants = []
    for item in namespace["MUTANTS"]:
        if len(item) != 5:
            raise MutationError(f"{file}: a mutant is (path, tests, name, old, new), got {item!r}")
        path, tests, name, old, new = item
        tests = (tests,) if isinstance(tests, str) else tuple(tests)
        mutants.append(Mutant(str(path), tests, str(name), str(old), str(new)))
    return mutants


def copy_tree(root: Path, destination: Path, names: Sequence[str]) -> None:
    for name in names:
        source = root / name
        if source.is_dir():
            shutil.copytree(
                source, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        elif source.is_file():
            shutil.copy2(source, destination / name)


def run_tests(work: Path, tests: Sequence[str], python: str) -> subprocess.CompletedProcess[str]:
    existing = os.environ.get("PYTHONPATH")
    env = {
        **os.environ,
        "PYTHONPATH": str(work / "src") + (os.pathsep + existing if existing else ""),
    }
    return subprocess.run(
        [python, "-m", "pytest", *PYTEST, *tests],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def check(
    root: Path,
    mutants: Sequence[Mutant],
    *,
    python: str = sys.executable,
    copy: Sequence[str] = DEFAULT_COPY,
) -> Report:
    """Run every mutant against an isolated copy of `root`; `root` itself is never written."""
    scratch = Path(tempfile.mkdtemp(prefix="mutation_check_"))
    try:
        work = scratch / "tree"
        work.mkdir()
        copy_tree(root, work, copy)
        everything = sorted({test for mutant in mutants for test in mutant.tests})
        baseline = run_tests(work, everything, python)
        if baseline.returncode != 0:
            tail = (baseline.stdout + baseline.stderr)[-1500:]
            return Report(False, tail, [])
        outcomes: list[Outcome] = []
        for mutant in mutants:
            target = work / mutant.path
            try:
                original = target.read_text(encoding="utf-8")
            except OSError as exc:
                outcomes.append(Outcome(mutant.name, "REFUSED", f"{mutant.path}: {exc}"))
                continue
            count = original.count(mutant.old)
            if count != 1:
                detail = f"the text to replace occurs {count} times in {mutant.path}, not once"
                outcomes.append(Outcome(mutant.name, "REFUSED", detail))
                continue
            try:
                target.write_text(
                    original.replace(mutant.old, mutant.new), encoding="utf-8", newline=""
                )
                done = run_tests(work, mutant.tests, python)
            finally:
                target.write_text(original, encoding="utf-8", newline="")
            outcomes.append(Outcome(mutant.name, "killed" if done.returncode else "SURVIVED"))
        return Report(True, "", outcomes)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mutation-test a repository on an isolated copy.")
    parser.add_argument("mutants", type=Path, help="a Python file defining MUTANTS")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="the repository")
    parser.add_argument(
        "--copy", default=",".join(DEFAULT_COPY), help="what to copy, comma-separated"
    )
    args = parser.parse_args(argv)
    try:
        mutants = load_mutants(args.mutants)
    except MutationError as exc:
        print(f"mutation_check: {exc}", file=sys.stderr)
        return 2
    report = check(args.root, mutants, copy=[part for part in args.copy.split(",") if part])
    if not report.baseline_ok:
        print(
            "mutation_check: the baseline run is red, so no mutant means anything:", file=sys.stderr
        )
        print(report.baseline_tail, file=sys.stderr)
        return 2
    for outcome in report.outcomes:
        extra = f"  ({outcome.detail})" if outcome.detail else ""
        print(f"{outcome.verdict:10} {outcome.name}{extra}")
    print(f"survivors: {[o.name for o in report.bad]}")
    return 1 if report.bad else 0


if __name__ == "__main__":
    sys.exit(main())
