"""Ground-truth files on disk: read, write, resolve borrowed labels, check integrity.

One JSON file per labeled document, ground_truth/<doc_id>.json. A document that carries the same
content as another (a degraded variant, a second photo of one invoice) does not get a file of its
own: its manifest row says `gt_from`, and `resolve_label` hands out the owner's label under the
borrower's doc_id. Label once, never copy, so the copies cannot drift apart.

`check_labels` is the integrity check the evaluation and the labeling app rely on; it is the only
place that knows all the rules at once (see ground_truth/LABELING.md), most importantly that a
BLIND document never has a model draft next to it.
"""

import json
import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from docextract.ingest.manifest import ManifestRow
from evalkit.groundtruth import SCALAR_FIELD_NAMES, GroundTruth


class GroundTruthError(Exception):
    """A label file is unreadable, inconsistent, or cannot be resolved."""


class MissingGroundTruth(GroundTruthError):
    """The document has no label yet."""


def label_path(directory: Path, doc_id: str) -> Path:
    return directory / f"{doc_id}.json"


# ---------------------------------------------------------------- reading and writing


def _problems(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"])
        parts.append(f"{where}: {item['msg']}" if where else item["msg"])
    return "; ".join(parts)


def parse_label(text: str, *, source: str) -> GroundTruth:
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise GroundTruthError(f"{source}: not valid JSON ({exc})") from exc
    try:
        return GroundTruth.model_validate(data)
    except ValidationError as exc:
        raise GroundTruthError(f"{source}: {_problems(exc)}") from exc


def read_label(path: Path) -> GroundTruth:
    """Load and validate one label file; the doc_id inside must be the file's name."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise GroundTruthError(f"{path.name}: cannot be read ({exc})") from exc
    label = parse_label(text, source=path.name)
    if label.doc_id != path.stem:
        raise GroundTruthError(f"{path.name}: holds doc_id {label.doc_id!r}, not {path.stem!r}")
    return label


def to_json_dict(label: GroundTruth) -> dict[str, Any]:
    """The label as the JSON object we store: every field explicit (a decision is a decision, so
    "__unlabeled__" is written out rather than left to a default), in a fixed, readable order."""
    data = label.model_dump(mode="json")
    result: dict[str, Any] = {"doc_id": data["doc_id"]}
    for key in ("source_file", "label_mode", "labeled_at"):
        if data[key] is not None:
            result[key] = data[key]
    for name in SCALAR_FIELD_NAMES:
        result[name] = data[name]
    result["line_items"] = data["line_items"]
    if data["field_notes"]:
        result["field_notes"] = data["field_notes"]
    if data["notes"] is not None:
        result["notes"] = data["notes"]
    return result


def dumps_label(label: GroundTruth) -> str:
    return json.dumps(to_json_dict(label), indent=2, ensure_ascii=False) + "\n"


def write_label(directory: Path, label: GroundTruth) -> Path:
    """Write the label atomically (a crash never leaves half a file)."""
    path = label_path(directory, label.doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(dumps_label(label))
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


# ---------------------------------------------------------------- borrowed labels


@dataclass(frozen=True)
class ResolvedLabel:
    doc_id: str  # the document that was asked for
    owner: str  # the document whose file holds the labels
    label: GroundTruth  # the labels, under `doc_id`

    @property
    def borrowed(self) -> bool:
        return self.owner != self.doc_id


def resolve_label(doc_id: str, rows: Sequence[ManifestRow], directory: Path) -> ResolvedLabel:
    by_id = {row.doc_id: row for row in rows}
    row = by_id.get(doc_id)
    if row is None:
        raise GroundTruthError(f"{doc_id}: not in the manifest")
    owner = row.gt_from or row.doc_id
    path = label_path(directory, owner)
    if not path.is_file():
        borrowed = f" (it takes its labels from {owner})" if owner != doc_id else ""
        raise MissingGroundTruth(f"{doc_id}: no ground truth yet{borrowed}")
    if owner != doc_id and label_path(directory, doc_id).exists():
        raise GroundTruthError(
            f"{doc_id}: has a label file of its own but also takes its labels from {owner}: "
            "keep one of them"
        )
    label = read_label(path)
    if owner != doc_id:
        label = label.model_copy(update={"doc_id": doc_id})
    return ResolvedLabel(doc_id, owner, label)


def load_labels(
    rows: Sequence[ManifestRow], directory: Path
) -> tuple[dict[str, ResolvedLabel], list[str]]:
    """Every document's label (borrowers resolved), and the doc_ids that have none yet."""
    labels: dict[str, ResolvedLabel] = {}
    missing: list[str] = []
    for row in rows:
        try:
            labels[row.doc_id] = resolve_label(row.doc_id, rows, directory)
        except MissingGroundTruth:
            missing.append(row.doc_id)
    return labels, missing


# ---------------------------------------------------------------- the changelog

PLACEHOLDER = "(no ground-truth files yet)"


def append_changelog(changelog: Path, lines: list[str]) -> None:
    """Append log lines to ground_truth/CHANGELOG.md (replacing its "no files yet" placeholder).
    Every change to a label gets a line here, in the same commit as the change."""
    text = changelog.read_text(encoding="utf-8") if changelog.is_file() else ""
    kept = [line for line in text.splitlines() if line.strip() != PLACEHOLDER]
    while kept and not kept[-1].strip():
        kept.pop()
    changelog.parent.mkdir(parents=True, exist_ok=True)
    changelog.write_text("\n".join([*kept, "", *lines]) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- integrity


def check_labels(
    rows: Sequence[ManifestRow], directory: Path, drafts_dir: Path | None = None
) -> list[str]:
    """Everything wrong with the label files, as readable lines (empty = consistent)."""
    problems: list[str] = []
    by_id = {row.doc_id: row for row in rows}
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    for path in files:
        row = by_id.get(path.stem)
        if row is None:
            problems.append(f"{path.name}: no document {path.stem!r} in the manifest")
            continue
        if row.gt_from is not None:
            problems.append(
                f"{path.name}: {row.doc_id} takes its labels from {row.gt_from}, "
                "so it must not have a label file of its own"
            )
            continue
        try:
            label = read_label(path)
        except GroundTruthError as exc:
            problems.append(str(exc))
            continue
        if label.label_mode != row.label_mode:
            problems.append(
                f"{path.name}: label_mode is {label.label_mode!r} but the manifest says "
                f"{row.label_mode!r}"
            )
    if drafts_dir is not None and drafts_dir.is_dir():
        for row in rows:
            if row.label_mode == "blind" and label_path(drafts_dir, row.doc_id).exists():
                problems.append(
                    f"drafts/{row.doc_id}.json: {row.doc_id} is a BLIND document and must never "
                    "have a model draft"
                )
    return problems


# ---------------------------------------------------------------- report


def format_report(
    rows: Sequence[ManifestRow], directory: Path, drafts_dir: Path | None = None
) -> tuple[str, list[str]]:
    """A coverage summary (how much of each split is labeled) and the integrity problems."""
    problems = check_labels(rows, directory, drafts_dir)
    labels: dict[str, ResolvedLabel] = {}
    missing = [row.doc_id for row in rows]
    with suppress(GroundTruthError):  # a broken file is already one of the problems
        labels, missing = load_labels(rows, directory)
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    borrowed = sum(1 for resolved in labels.values() if resolved.borrowed)
    lines = [
        f"ground truth: {len(files)} label file(s) -> {len(labels)} document(s) labeled "
        f"({borrowed} by borrowing), {len(missing)} without a label yet"
    ]
    by_split: dict[str, list[int]] = {}
    for row in rows:
        counts = by_split.setdefault(row.split or "(no split)", [0, 0])
        counts[0] += 1 if row.doc_id in labels else 0
        counts[1] += 1
    for split in sorted(by_split):
        done, total = by_split[split]
        lines.append(f"  {split:<12}{done:>4} / {total:<4} labeled")
    by_mode: dict[str, int] = {}
    for row in rows:
        if row.doc_id in labels and row.gt_from is None:
            key = row.label_mode or "generated"
            by_mode[key] = by_mode.get(key, 0) + 1
    if by_mode:
        lines.append(
            "  hand/generated labels: " + ", ".join(f"{k} {v}" for k, v in sorted(by_mode.items()))
        )
    lines.append("problems: none" if not problems else f"problems: {len(problems)}")
    lines.extend(f"  - {problem}" for problem in problems)
    return "\n".join(lines), problems
