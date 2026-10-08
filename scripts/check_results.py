"""Re-score every committed run under results/ offline and check the committed scorecards.

Usage: python -m uv run python scripts/check_results.py [RESULT_DIR ...]
           [--results-dir DIR] [--manifest FILE] [--pages-dir DIR] [--gt-dir DIR]

No model, no API key, no network. Each results/<run_id>/ holds a run (run.json, docs/) and the
score/ made from it. This scores the committed run again against the labels as they are now,
passing along the git SHA and the time the committed score recorded, and requires every file of
score/ (scorecard.md/json, manifest.json, docs/*.json) to come out byte for byte the same. It
fails when the labels, the manifest, the stored pages, the comparators or the scoring code changed
in a way that moves a committed number, or when a committed record or scorecard was edited.

It does not guard the model's raw text: an edit there changes no score (git itself covers that).

Exit codes: 0 every result reproduces, 1 one differs (the files are listed), 2 nothing to check
or a result that cannot be read.
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from docextract.paths import DatasetPaths, find_repo_root
from evalkit.gt_store import GroundTruthError
from evalkit.scorecard import ScorecardError, score_run


class Unreadable(Exception):
    """A result folder that cannot be checked at all (and why)."""


def committed_facts(result: Path) -> tuple[str, bool, datetime]:
    """The git SHA, dirty flag and scoring time the committed score recorded."""
    path = result / "score" / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            str(data["git_sha"]),
            bool(data["git_dirty"]),
            datetime.fromisoformat(data["scored_at"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Unreadable(f"{path}: cannot be read as a score manifest ({exc!r})") from exc


def files_of(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def differences(result: Path, manifest: Path, pages_dir: Path, labels_dir: Path) -> list[str]:
    """What differs between the committed score/ and a fresh one (empty = it reproduces)."""
    sha, dirty, scored_at = committed_facts(result)
    try:
        card = score_run(
            result,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=scored_at,
            git_sha=sha,
            git_dirty=dirty,
        )
    except (ScorecardError, GroundTruthError) as exc:
        raise Unreadable(f"{result}: cannot be scored ({exc})") from exc
    with tempfile.TemporaryDirectory(prefix="check-results-") as scratch:
        card.write(Path(scratch))
        fresh = files_of(Path(scratch))
    committed = files_of(result / "score")
    problems = [
        f"missing from the committed score: {name}"
        for name in sorted(fresh.keys() - committed.keys())
    ]
    problems += [
        f"not produced by a fresh score: {name}" for name in sorted(committed.keys() - fresh.keys())
    ]
    problems += [
        f"differs: {name}"
        for name in sorted(fresh.keys() & committed.keys())
        if fresh[name] != committed[name]
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    root = find_repo_root()
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(description="Re-score committed results offline.")
    parser.add_argument("results", nargs="*", type=Path, help="result folders (default: all)")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--manifest", type=Path, default=paths.manifest)
    parser.add_argument("--pages-dir", type=Path, default=paths.pages_dir)
    parser.add_argument("--gt-dir", type=Path, default=paths.ground_truth_dir)
    args = parser.parse_args(argv)

    folders = args.results or sorted(
        path.parent for path in args.results_dir.glob("*/run.json") if path.is_file()
    )
    if not folders:
        print(f"no result to check under {args.results_dir}", file=sys.stderr)
        return 2
    status = 0
    for folder in folders:
        try:
            problems = differences(folder, args.manifest, args.pages_dir, args.gt_dir)
        except Unreadable as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if problems:
            status = 1
            print(f"{folder.name}: DOES NOT REPRODUCE")
            for problem in problems:
                print(f"  {problem}")
        else:
            print(f"{folder.name}: reproduces ({len(files_of(folder / 'score'))} file(s))")
    return status


if __name__ == "__main__":
    sys.exit(main())
