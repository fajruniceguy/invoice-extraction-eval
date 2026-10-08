"""Promote a finished, scored run to results/<run_id>/, the folder whose contents are committed.

Usage: python -m uv run python scripts/promote_run.py outputs/runs/<run_id> [--results-dir DIR]
           [--allow-incomplete]

outputs/ is gitignored by design (caches, logs, scratch runs). A run worth keeping as evidence is
copied here byte for byte: run.json, docs/<doc_id>.json and score/ (scorecard.md/json, the run
manifest and the per-document results). Nothing else is copied.

Refused (exit 1, nothing written):
  * a run with no score/ (run scripts/score.py first), or a score/ made for another run id;
  * a run the cost cap stopped, or one with a document that is not "ok" (failed, refused,
    not_run), unless --allow-incomplete;
  * docs/ that do not match the run's statuses (a stray or a missing record);
  * a results/<run_id>/ that already exists (a committed result is never overwritten);
  * any copied file holding something shaped like a key (the rules of tools/secret_scan.py).
Exit 2: bad arguments or a run folder that cannot be read.

The records hold the model's raw text, which echoes the (already redacted) pages: read results/
before you push it.
"""

import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

from pydantic import ValidationError

from docextract.extract.runner import RunRecord
from docextract.paths import find_repo_root

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SCORE_FILES = ("scorecard.md", "scorecard.json", "manifest.json")


class Refused(Exception):
    """The run must not be promoted (and why)."""


def load_scanner(repo: Path) -> ModuleType:
    path = repo / "tools" / "secret_scan.py"
    spec = importlib.util.spec_from_file_location("secret_scan", path)
    if spec is None or spec.loader is None:
        raise Refused(f"{path}: the secret scanner cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def files_to_copy(run_dir: Path, run: RunRecord) -> list[Path]:
    """Paths relative to `run_dir`, in a fixed order. Raises Refused for a run not fit to keep."""
    score = run_dir / "score"
    missing = [name for name in SCORE_FILES if not (score / name).is_file()]
    if missing:
        raise Refused(f"{score}: not scored yet (missing {', '.join(missing)}): run score.py first")
    try:
        scored = json.loads((score / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused(f"{score / 'manifest.json'}: cannot be read ({exc})") from exc
    if not isinstance(scored, dict) or scored.get("run_id") != run.run_id:
        raise Refused(f"{score}: it was made for another run, score {run.run_id} again")
    records = sorted(path.stem for path in (run_dir / "docs").glob("*.json"))
    if records != sorted(run.statuses):
        raise Refused(
            f"{run_dir / 'docs'}: holds {len(records)} record(s) but run.json lists "
            f"{len(run.statuses)} document(s)"
        )
    paths = [Path("run.json")]
    paths += [Path("docs") / f"{doc_id}.json" for doc_id in sorted(run.statuses)]
    paths += sorted(path.relative_to(run_dir) for path in score.rglob("*") if path.is_file())
    return paths


def check_run(run: RunRecord, *, allow_incomplete: bool) -> None:
    if run.stopped is not None:
        raise Refused(f"the run was stopped ({run.stopped}): it is not a complete result")
    if not allow_incomplete:
        bad = {doc: status for doc, status in run.statuses.items() if status != "ok"}
        if bad:
            listing = ", ".join(f"{doc} {status}" for doc, status in sorted(bad.items()))
            raise Refused(f"not every document is ok ({listing}); --allow-incomplete keeps it")


def check_secrets(run_dir: Path, paths: list[Path], scanner: ModuleType) -> None:
    for relative in paths:
        text = (run_dir / relative).read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            hits = scanner.scan_line(line)
            if hits:
                rule = hits[0][0]
                raise Refused(
                    f"{relative.as_posix()}:{number}: something shaped like a key ({rule})"
                )


def promote(
    run_dir: Path, results_dir: Path, scanner: ModuleType, *, allow_incomplete: bool = False
) -> Path:
    """Copy the run; returns the new folder. Raises Refused before writing anything."""
    try:
        run = RunRecord.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise Refused(f"{run_dir / 'run.json'}: not a run record ({exc})") from exc
    if not _RUN_ID.fullmatch(run.run_id) or run.run_id != run_dir.name:
        raise Refused(f"run.json names the run {run.run_id!r}, but the folder is {run_dir.name!r}")
    target = results_dir / run.run_id
    if target.exists():
        raise Refused(f"{target} already exists: a committed result is never overwritten")
    check_run(run, allow_incomplete=allow_incomplete)
    paths = files_to_copy(run_dir, run)
    check_secrets(run_dir, paths, scanner)
    for relative in paths:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / relative, destination)
    return target


def main(argv: list[str] | None = None) -> int:
    root = find_repo_root()
    parser = argparse.ArgumentParser(description="Copy a finished, scored run to results/.")
    parser.add_argument("run_dir", type=Path, help="outputs/runs/<run_id>")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--repo", type=Path, default=root, help="where tools/secret_scan.py is")
    args = parser.parse_args(argv)
    if not args.run_dir.is_dir():
        print(f"{args.run_dir}: not a directory", file=sys.stderr)
        return 2
    try:
        target = promote(
            args.run_dir,
            args.results_dir,
            load_scanner(args.repo),
            allow_incomplete=args.allow_incomplete,
        )
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    count = sum(1 for path in target.rglob("*") if path.is_file())
    print(f"promoted {args.run_dir.name}: {count} file(s) -> {target}")
    print("read it before you push: the records echo (redacted) page content")
    return 0


if __name__ == "__main__":
    sys.exit(main())
