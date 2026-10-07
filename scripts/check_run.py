"""Check a run folder against the extraction output contract (the M4 definition of done).

Usage: python -m uv run python scripts/check_run.py RUN_DIR [RUN_DIR ...]

For every `ok` record in <RUN_DIR>/docs/*.json it checks the VALIDATED extraction (what the
validators concluded, not what the model said):

  * every field and line item carries a value (null when absent or abstained), a confidence in
    [0, 1], a status and reasons;
  * an `extracted` field or row: a non-empty evidence quote and a page inside 1..pages sent;
  * an `absent` or `abstained` one: a null value (every cell null for a row);
  * an `abstained` one: at least one reason — a validator code from
    `docextract.validate.codes.REASON_CODES`, or the model's own free-text note;
  * a row's `flagged_cells` are cell names.

The vocabulary is only checked on the reasons the VALIDATORS added: the record holds `parsed`
(what the model said, its own prose reasons included) as well as `validated`, so a reason is
free text exactly when the model already gave it and a code otherwise. Where a record has no
`parsed` output, the vocabulary is not checked (the codes cannot be told apart from the model's
words).

Prints the totals and, when everything is clean, `all fields carry value/confidence/evidence/
status/reasons`. Exit codes: 0 clean, 1 a check failed, 2 bad arguments or an unreadable run.
This is offline: it reads run artifacts only.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docextract.extract.schema import CELL_NAMES, SCALAR_FIELD_NAMES
from docextract.validate.codes import REASON_CODES

EXTRACTED = "extracted"
ABSENT = "absent"
ABSTAINED = "abstained"
STATUSES = (EXTRACTED, ABSENT, ABSTAINED)


@dataclass
class Counts:
    documents: int = 0
    fields: int = 0
    rows: int = 0
    abstained: int = 0
    flagged: int = 0


@dataclass
class CheckReport:
    run_dir: Path
    counts: Counts = field(default_factory=Counts)
    problems: list[str] = field(default_factory=list)
    not_ok: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _is_probability(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 <= float(value) <= 1.0


def _is_page(value: Any, page_count: int) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return 1 <= value <= page_count


def _model_reasons(parsed: Any, name: str, index: int | None) -> set[str] | None:
    """The reasons the model itself gave for this field (index None) or row (1-based), or None
    when they cannot be told (no parsed output at all): None means "do not check the vocabulary"."""
    if not isinstance(parsed, dict):
        return None
    item: Any = parsed.get(name)
    if index is not None:
        rows = parsed.get("line_items")
        item = rows[index - 1] if isinstance(rows, list) and len(rows) >= index else None
    if not isinstance(item, dict):
        return None
    reasons = item.get("reasons")
    return set(reasons) if isinstance(reasons, list) else None


def _reason_problems(
    label: str, item: dict[str, Any], model_reasons: set[str] | None
) -> list[str]:
    """Reasons: a list of strings; on an abstained item at least one; whatever is not a reason
    the model already gave must be a code from the vocabulary."""
    status = item.get("status")
    reasons = item.get("reasons")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        return [f"{label}: reasons must be a list of strings"]
    problems: list[str] = []
    if status == ABSTAINED and not any(reason.strip() for reason in reasons):
        problems.append(f"{label}: abstained without a reason")
    if model_reasons is not None:
        added = sorted({reason for reason in reasons if reason not in model_reasons})
        unknown = [reason for reason in added if reason not in REASON_CODES]
        if unknown:
            problems.append(f"{label}: unknown validator reason code(s) {unknown}")
    return problems


def _status_problems(label: str, item: Any) -> tuple[str | None, list[str]]:
    if not isinstance(item, dict):
        return None, [f"{label}: not an object"]
    status = item.get("status")
    if status not in STATUSES:
        return None, [f"{label}: status {status!r} is not one of {list(STATUSES)}"]
    problems: list[str] = []
    if not _is_probability(item.get("confidence")):
        problems.append(f"{label}: confidence {item.get('confidence')!r} is not in [0, 1]")
    return status, problems


def _check_field(
    label: str, item: Any, page_count: int, model_reasons: set[str] | None
) -> list[str]:
    status, problems = _status_problems(label, item)
    if status is None:
        return problems
    assert isinstance(item, dict)
    value = item.get("value")
    if status == EXTRACTED:
        if value is None:
            problems.append(f"{label}: extracted with a null value")
        evidence = item.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            problems.append(f"{label}: extracted without evidence")
        if not _is_page(item.get("page"), page_count):
            problems.append(f"{label}: extracted without a page in 1..{page_count}")
    elif value is not None:
        problems.append(f"{label}: {status} with a non-null value")
    problems.extend(_reason_problems(label, item, model_reasons))
    return problems


def _check_row(
    label: str, item: Any, page_count: int, model_reasons: set[str] | None
) -> list[str]:
    status, problems = _status_problems(label, item)
    if status is None:
        return problems
    assert isinstance(item, dict)
    cells = {cell: item.get(cell) for cell in CELL_NAMES}
    if status == EXTRACTED:
        if all(cell is None for cell in cells.values()):
            problems.append(f"{label}: extracted with every cell null")
        evidence = item.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            problems.append(f"{label}: extracted without evidence")
        if not _is_page(item.get("page"), page_count):
            problems.append(f"{label}: extracted without a page in 1..{page_count}")
    elif any(cell is not None for cell in cells.values()):
        problems.append(f"{label}: {status} with a non-null cell")
    flagged = item.get("flagged_cells")
    if not isinstance(flagged, list) or any(cell not in CELL_NAMES for cell in flagged):
        problems.append(f"{label}: flagged_cells {flagged!r} are not cell names")
    problems.extend(_reason_problems(label, item, model_reasons))
    return problems


def check_record(record: Any, counts: Counts) -> list[str]:
    """Every problem with one `ok` record's validated output (and the counts it adds up)."""
    if not isinstance(record, dict):
        return ["a record is not an object"]
    doc_id = record.get("doc_id") or "?"
    validated = record.get("validated")
    pages = record.get("pages")
    page_count = len(pages) if isinstance(pages, list) else 0
    if not isinstance(validated, dict):
        return [f"{doc_id}: an ok record without a validated output"]
    problems: list[str] = []
    parsed = record.get("parsed")
    for name in SCALAR_FIELD_NAMES:
        if name not in validated:
            problems.append(f"{doc_id} {name}: missing")
            continue
        counts.fields += 1
        field = validated[name]
        problems.extend(
            _check_field(f"{doc_id} {name}", field, page_count, _model_reasons(parsed, name, None))
        )
        if isinstance(field, dict) and field.get("status") == ABSTAINED:
            counts.abstained += 1
    rows = validated.get("line_items")
    if not isinstance(rows, list):
        problems.append(f"{doc_id} line_items: not a list")
        return problems
    for index, row in enumerate(rows, start=1):
        counts.rows += 1
        problems.extend(
            _check_row(
                f"{doc_id} line_items[{index}]",
                row,
                page_count,
                _model_reasons(parsed, "line_items", index),
            )
        )
        if isinstance(row, dict):
            if row.get("status") == ABSTAINED:
                counts.abstained += 1
            flagged = row.get("flagged_cells")
            if isinstance(flagged, list):
                counts.flagged += sum(1 for cell in flagged if cell in CELL_NAMES)
    return problems


def check_run_dir(run_dir: Path) -> CheckReport:
    """Check every document record the run folder holds."""
    report = CheckReport(run_dir=run_dir)
    docs_dir = run_dir / "docs"
    if not docs_dir.is_dir():
        report.problems.append(f"{run_dir}: no docs/ directory")
        return report
    paths = sorted(docs_dir.glob("*.json"))
    if not paths:
        report.problems.append(f"{docs_dir}: no document records")
        return report
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.problems.append(f"{path.name}: cannot be read ({exc})")
            continue
        if not isinstance(record, dict) or record.get("status") != "ok":
            doc_id = record.get("doc_id") if isinstance(record, dict) else None
            report.not_ok.append(str(doc_id or path.stem))
            continue
        report.counts.documents += 1
        report.problems.extend(check_record(record, report.counts))
    return report


def format_report(report: CheckReport) -> str:
    counts = report.counts
    lines = [
        f"run {report.run_dir.name}: {counts.documents} document(s) checked, "
        f"{len(report.not_ok)} not ok (no validated output to check)"
    ]
    if report.problems:
        lines.extend(f"  FAIL {problem}" for problem in report.problems)
        lines.append(f"{len(report.problems)} problem(s)")
        return "\n".join(lines)
    lines.append(
        f"  {counts.fields} field(s), {counts.rows} row(s), {counts.abstained} abstained, "
        f"{counts.flagged} flagged cell(s)"
    )
    lines.append("all fields carry value/confidence/evidence/status/reasons")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a run folder against the extraction output contract."
    )
    parser.add_argument("run_dir", nargs="+", type=Path, help="outputs/runs/<run_id>")
    args = parser.parse_args(argv)
    status = 0
    for run_dir in args.run_dir:
        if not run_dir.is_dir():
            print(f"{run_dir}: not a directory", file=sys.stderr)
            status = 2
            continue
        report = check_run_dir(run_dir)
        print(format_report(report))
        if not report.ok:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
