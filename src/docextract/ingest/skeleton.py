"""Build the manifest skeleton from the raw folder: one row per file, everything that can be
inferred filled in, the rest left blank for a human.

Merge-safe by construction: existing rows are never rewritten, a raw file already referenced by
any row (even a page-range slice of it) is never added again, and new ids continue after the
highest existing `doc_NNN`. Re-running after dropping in more files only appends.

Only strong evidence is filled in: `capture` from camera EXIF (phone_photo) or a PDF text layer
(born_digital), `language` from a decisive stopword majority in a text layer. doc_type,
template_group, needs_redaction, split and label_mode are always left to a human.
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from docextract.fsutil import long_path
from docextract.ingest.manifest import ManifestRow
from docextract.ingest.probe import SUPPORTED_EXTENSIONS, FileProbe, probe_file

RESERVED_DIRS = frozenset(
    {"constructed"}
)  # generated invoices live here, rows come from their script
_IGNORED_NAMES = frozenset({"thumbs.db", "desktop.ini", ".ds_store"})
_DOC_NUMBER = re.compile(r"doc_(\d+)")
# fields a human must always decide, reported as "could not infer" when blank
HAND_FILLED = ("doc_type", "template_group", "needs_redaction", "split", "label_mode")


@dataclass
class SkeletonResult:
    rows: list[ManifestRow]  # existing rows first, untouched, then the new ones
    new_ids: list[str] = field(default_factory=list)
    probes: dict[str, FileProbe] = field(default_factory=dict)  # new files, by relative path
    ignored: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    missing: list[tuple[str, str]] = field(default_factory=list)  # (doc_id, raw file not on disk)
    unreadable: list[tuple[str, str]] = field(default_factory=list)  # (path, why)
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # byte-identical new files


def _natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name.casefold())]


def scan_raw(raw_dir: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Relative POSIX paths of supported files, naturally sorted, and what was skipped and why."""
    found: list[str] = []
    ignored: list[tuple[str, str]] = []
    # extended-length path: on Windows a long file name would otherwise be silently skipped
    root = long_path(raw_dir)
    if not root.is_dir():
        return found, ignored
    for current, _directories, names in os.walk(root):
        for name in names:
            relative = (Path(current) / name).relative_to(root).as_posix()
            parts = relative.split("/")
            if parts[0] in RESERVED_DIRS and len(parts) > 1:
                continue
            if name.startswith((".", "~$")) or name.casefold() in _IGNORED_NAMES:
                continue
            suffix = Path(name).suffix
            if suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
                ignored.append((relative, f"unsupported extension {suffix or '(none)'!r}"))
            elif ";" in relative:
                ignored.append(
                    (relative, "the name contains ';', the manifest's list separator: rename it")
                )
            else:
                found.append(relative)
    return sorted(found, key=_natural_key), sorted(ignored)


def _next_number(rows: list[ManifestRow]) -> tuple[int, int]:
    numbers = [int(m.group(1)) for r in rows if (m := _DOC_NUMBER.fullmatch(r.doc_id))]
    width = max([3] + [len(m.group(1)) for r in rows if (m := _DOC_NUMBER.fullmatch(r.doc_id))])
    return (max(numbers) + 1 if numbers else 1), width


def _row_for(doc_id: str, probe: FileProbe) -> ManifestRow:
    capture = None
    if probe.content_format == "pdf" and probe.has_text_layer:
        capture = "born_digital"
    elif probe.has_camera_exif:
        capture = "phone_photo"
    return ManifestRow(
        doc_id=doc_id,
        raw_files=(probe.name,),
        content_group=doc_id,  # its own group until someone says it shares content
        capture=capture,
        original_format=probe.extension,
        language=probe.language,
    )


def build_skeleton(raw_dir: Path, existing: list[ManifestRow]) -> SkeletonResult:
    files, ignored = scan_raw(raw_dir)
    result = SkeletonResult(rows=list(existing), ignored=ignored)
    on_disk = set(files)

    referenced = {name for row in existing for name in row.raw_files}
    for row in existing:
        # missing means NOT ON DISK, not "not in the scan": the scan skips reserved folders
        # (the constructed invoices live in one) that rows legitimately point into
        result.missing.extend(
            (row.doc_id, name)
            for name in row.raw_files
            if name not in on_disk and not long_path(raw_dir / name).is_file()
        )

    number, width = _next_number(existing)
    by_hash: dict[str, list[str]] = defaultdict(list)
    for relative in files:
        if relative in referenced:
            continue
        probe = probe_file(long_path(raw_dir / relative), name=relative)
        doc_id = f"doc_{number:0{width}d}"
        number += 1
        result.rows.append(_row_for(doc_id, probe))
        result.new_ids.append(doc_id)
        result.probes[doc_id] = probe
        if probe.error:
            result.unreadable.append((relative, probe.error))
        by_hash[probe.sha256].append(doc_id)
    for ids in by_hash.values():
        result.duplicates.extend((ids[0], other) for other in ids[1:])
    return result


# ---------------------------------------------------------------- the human-readable report


def _size(num_bytes: int) -> str:
    return (
        f"{num_bytes / 1_048_576:.1f}MB" if num_bytes >= 1_048_576 else f"{num_bytes / 1024:.0f}KB"
    )


def hints_for(doc_id: str, probe: FileProbe) -> list[str]:
    hints: list[str] = []
    if probe.error:
        return [f"COULD NOT READ: {probe.error}"]
    if probe.extension_mismatch:
        hints.append(f"extension .{probe.extension} but the bytes are {probe.content_format}")
    if probe.content_format == "pdf":
        if (probe.page_count or 0) > 1:
            hints.append(
                f"{probe.page_count} pages: one document, or a bundle to split into rows "
                "with page_range?"
            )
        if not probe.has_text_layer:
            hints.append("image-only PDF (scan or screenshot?): capture left blank")
    elif not probe.has_camera_exif:
        hints.append("no camera EXIF (WhatsApp and screenshots strip it): capture left blank")
    if probe.has_gps:
        hints.append("contains GPS metadata (stripped on ingest)")
    if re.search(r"whatsapp", probe.name, re.IGNORECASE):
        hints.append("file name looks like a WhatsApp export")
    return hints


def format_report(
    result: SkeletonResult, raw_dir: Path, manifest_path: Path, *, dry_run: bool
) -> str:
    lines = [
        f"manifest_init: {len(scan_raw(raw_dir)[0])} supported file(s) in {raw_dir}",
        f"  manifest: {manifest_path}{'  (dry run: not written)' if dry_run else ''}",
        f"  kept {len(result.rows) - len(result.new_ids)} existing row(s) untouched; "
        f"added {len(result.new_ids)} new",
        "",
    ]
    if result.new_ids:
        lines.append(f"{'doc_id':<9} {'file':<44} {'fmt':<5} {'pg':>3} {'size':>8}  inferred")
        for doc_id in result.new_ids:
            probe = result.probes[doc_id]
            row = next(r for r in result.rows if r.doc_id == doc_id)
            inferred = ", ".join(
                f"{column}={getattr(row, column)}"
                for column in ("capture", "language")
                if getattr(row, column) is not None
            )
            name = probe.name if len(probe.name) <= 44 else probe.name[:41] + "..."
            pages = "-" if probe.page_count is None else str(probe.page_count)
            lines.append(
                f"{doc_id:<9} {name:<44} {probe.extension:<5} {pages:>3} "
                f"{_size(probe.size_bytes):>8}  {inferred or '(nothing inferred)'}"
            )
        lines.append("")
        hint_lines = [(i, h) for i in result.new_ids for h in hints_for(i, result.probes[i])]
        if hint_lines:
            lines.append("Hints (for you to decide):")
            lines.extend(f"  {doc_id}: {hint}" for doc_id, hint in hint_lines)
            lines.append("")
        lines.append("Could not infer (fill in by hand), rows still blank:")
        for column in HAND_FILLED:
            blank = sum(
                1
                for doc_id in result.new_ids
                if getattr(next(r for r in result.rows if r.doc_id == doc_id), column) is None
            )
            lines.append(f"  {column:<16} {blank} of {len(result.new_ids)}")
        for column in ("capture", "language"):
            blank = [
                doc_id
                for doc_id in result.new_ids
                if getattr(next(r for r in result.rows if r.doc_id == doc_id), column) is None
            ]
            lines.append(f"  {column:<16} {len(blank)} of {len(result.new_ids)}")
    for kept, other in result.duplicates:
        lines.append(
            f"NOTE: {other} is byte-identical to {kept}: same content_group? (or remove it)"
        )
    for doc_id, name in result.missing:
        lines.append(f"MISSING: {doc_id} refers to {name!r}, which is not in the raw folder")
    for path, reason in result.ignored:
        lines.append(f"IGNORED: {path}: {reason}")
    return "\n".join(lines)
