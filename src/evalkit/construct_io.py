"""Writing the constructed invoices: PDF, ground truth, changelog line, manifest row.

Ground truth is never edited silently (CLAUDE.md): a new label is logged in
ground_truth/CHANGELOG.md, a label that already exists is left alone when it is identical, and when
it differs the run is refused unless `force` is set, in which case the change is logged too.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from docextract.ingest.manifest import ManifestRow, read_manifest, write_manifest
from evalkit.construct import (
    DEFAULT_SEED,
    Invoice,
    generate_invoice,
    ground_truth,
    manifest_row,
    render_invoice,
)
from evalkit.groundtruth import GroundTruth

LAYOUT_ORDER = ("classic", "modern", "compact")
PLACEHOLDER = "(no ground-truth files yet)"


class ConstructError(Exception):
    """The constructed files on disk disagree with what the generator would write."""


@dataclass(frozen=True)
class Action:
    doc_id: str
    pdf: Literal["written", "unchanged"]
    ground_truth: Literal["created", "unchanged", "replaced"]
    manifest: Literal["added", "present"]
    pages: int
    items: int


def doc_ids() -> list[str]:
    return [f"cons_{index:03d}" for index in range(1, len(LAYOUT_ORDER) + 1)]


def build_invoices(seed: str = DEFAULT_SEED) -> list[Invoice]:
    return [
        generate_invoice(doc_id, layout, seed)
        for doc_id, layout in zip(doc_ids(), LAYOUT_ORDER, strict=True)
    ]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _gt_text(label: dict[str, Any]) -> str:
    return json.dumps(label, indent=2, ensure_ascii=False) + "\n"


def _changed_fields(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    return sorted(key for key in {*old, *new} if old.get(key) != new.get(key))


def append_changelog(changelog: Path, lines: list[str]) -> None:
    text = changelog.read_text(encoding="utf-8") if changelog.is_file() else ""
    kept = [line for line in text.splitlines() if line.strip() != PLACEHOLDER]
    while kept and not kept[-1].strip():
        kept.pop()
    changelog.parent.mkdir(parents=True, exist_ok=True)
    changelog.write_text("\n".join([*kept, "", *lines]) + "\n", encoding="utf-8", newline="\n")


def construct(
    *,
    raw_dir: Path,
    ground_truth_dir: Path,
    manifest: Path,
    changelog: Path,
    seed: str = DEFAULT_SEED,
    force: bool = False,
    check: bool = False,
    today: date | None = None,
) -> list[Action]:
    """Make (or verify) the constructed invoices. Nothing is written when `check` is set, and
    nothing is written at all if a ground-truth conflict is found without `force`."""
    invoices = build_invoices(seed)
    labels = {inv.doc_id: ground_truth(inv) for inv in invoices}
    for label in labels.values():
        GroundTruth.model_validate(label)  # what is written must load: checked before any write
    stamp = (today or date.today()).isoformat()

    conflicts: list[str] = []
    existing: dict[str, dict[str, Any] | None] = {}
    for invoice in invoices:
        path = ground_truth_dir / f"{invoice.doc_id}.json"
        existing[invoice.doc_id] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
        old = existing[invoice.doc_id]
        if old is not None and old != labels[invoice.doc_id] and not force:
            fields = ", ".join(_changed_fields(old, labels[invoice.doc_id]))
            conflicts.append(
                f"{path.name} differs from what the generator writes for seed {seed!r} "
                f"({fields}): refusing to overwrite a label (use --force; it is logged)"
            )
    rows: list[ManifestRow] = read_manifest(manifest) if manifest.is_file() else []
    by_id = {row.doc_id: row for row in rows}
    for invoice in invoices:
        row = by_id.get(invoice.doc_id)
        if row is not None and row.capture != "constructed":
            conflicts.append(f"{invoice.doc_id} is in the manifest with capture={row.capture}")
    if conflicts:
        raise ConstructError("\n".join(conflicts))

    actions: list[Action] = []
    log: list[str] = []
    new_rows = list(rows)
    for invoice in invoices:
        label = labels[invoice.doc_id]
        pdf_path = raw_dir / "constructed" / f"{invoice.doc_id}.pdf"
        with tempfile.TemporaryDirectory() as scratch:  # rendered away from the real folder
            plan = render_invoice(invoice, Path(scratch) / "invoice.pdf")
            data = (Path(scratch) / "invoice.pdf").read_bytes()
        pdf_state: Literal["written", "unchanged"] = (
            "unchanged" if pdf_path.is_file() and pdf_path.read_bytes() == data else "written"
        )
        old = existing[invoice.doc_id]
        gt_state: Literal["created", "unchanged", "replaced"] = (
            "created" if old is None else ("unchanged" if old == label else "replaced")
        )
        added = invoice.doc_id not in by_id
        if not check:
            if pdf_state == "written":
                _atomic_write(pdf_path, data)
            if gt_state != "unchanged":
                _atomic_write(
                    ground_truth_dir / f"{invoice.doc_id}.json", _gt_text(label).encode("utf-8")
                )
                where = f"{len(plan)} pages, {len(invoice.items)} items"
                if gt_state == "created":
                    log.append(
                        f"{stamp} | {invoice.doc_id}.json | (new file) | - -> {where} | "
                        f"constructed by scripts/construct_invoices.py "
                        f"(seed {seed!r}, {invoice.layout} layout)"
                    )
                else:
                    fields = ", ".join(_changed_fields(old or {}, label))
                    log.append(
                        f"{stamp} | {invoice.doc_id}.json | {fields} | (see git diff) | "
                        f"regenerated with --force (seed {seed!r}, {invoice.layout} layout)"
                    )
        if added:
            new_rows.append(manifest_row(invoice))
        actions.append(
            Action(
                invoice.doc_id,
                pdf_state,
                gt_state,
                "added" if added else "present",
                len(plan),
                len(invoice.items),
            )
        )
    if not check:
        if log:
            append_changelog(changelog, log)
        if len(new_rows) != len(rows):
            write_manifest(manifest, new_rows)
    return actions
