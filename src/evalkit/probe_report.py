"""The hallucination probe report (S4): how often a field that was erased is still answered.

The probe set (`evalkit.probes`) erases a field from a copy of a document, so the truth is known
by construction: the field is not on the page, and the expected answer is `absent` (the document
does not state it) or `abstained` (not confident). A value is a hallucination. This module reads a
run of the probe set and reports, for every erased field:

  * what the system returned (the validated field) and what the model said before the validators;
  * for a returned value, whether it is the erased value (read from somewhere else on the page,
    or computed from the rows: still a value the document does not state) or another value;
  * whether the CONTROL (the unmodified document, same run) read that field right: an erased
    case where the control was wrong anyway says less;

and, so that answering nothing is not mistaken for being good, the accuracy of the fields that were
NOT erased and the rows the model invented for a page that was dropped.

The rate comes with an exact (Clopper-Pearson) 95% interval: the set is small. Nothing here gates
or thresholds. Offline and deterministic; changes no record, label or score.
"""

from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from docextract.extract.runner import DocRecord, RunRecord
from evalkit.comparators import compare_field
from evalkit.groundtruth import GroundTruth, is_unlabeled
from evalkit.gt_store import GroundTruthError, label_path, read_label
from evalkit.metrics import SCALAR_FIELD_NAMES, score_field, score_rows
from evalkit.probes import ProbeError, read_probes

ABSENT, ABSTAINED, EXTRACTED = "absent", "abstained", "extracted"
CONFIDENCE = 0.95


class ProbeReportError(Exception):
    """The run or the probe set cannot be analysed (and why)."""


def exact_interval(k: int, n: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """The Clopper-Pearson interval for k successes in n trials (bisection on the tails)."""
    if n <= 0:
        return (0.0, 1.0)
    alpha = 1 - confidence

    def upper_tail(p: float) -> float:  # P(X >= k)
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def lower_tail(p: float) -> float:  # P(X <= k)
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def solve(function: Any, target: float, increasing: bool) -> float:
        low, high = 0.0, 1.0
        for _ in range(80):
            mid = (low + high) / 2
            if (function(mid) < target) == increasing:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    lower = 0.0 if k == 0 else solve(upper_tail, alpha / 2, True)
    upper = 1.0 if k == n else solve(lower_tail, alpha / 2, False)
    return (lower, upper)


@dataclass(frozen=True)
class ErasedCase:
    """One erased field of one probe."""

    probe: str
    source: str
    kind: str  # "erase" or "drop_last_page"
    field: str
    original: str  # the label value that was erased (as the label prints it)
    returned: str  # absent | abstained | extracted: the validated field
    raw: str  # the same for the model's own answer
    value: str  # what was returned, when it was extracted
    recovered: bool | None  # extracted: is it the erased value?
    control_read: bool | None  # did the control read this field right (None: no control)


@dataclass(frozen=True)
class ProbeLine:
    probe: str
    source: str
    layout: str
    kind: str
    erased: tuple[str, ...]
    status: str  # the record's status
    returned: dict[str, int]
    raw: dict[str, int]
    kept_correct: int  # fields that were NOT erased, labeled non-null, answered right
    kept_labeled: int
    rows_labeled: int
    rows_matched: int
    rows_missed: int
    rows_spurious: int


@dataclass(frozen=True)
class ControlLine:
    probe: str
    source: str
    status: str
    correct: int
    labeled: int


@dataclass(frozen=True)
class ProbeReport:
    run_id: str
    model: str
    prompt_sha256: str
    seed: str
    cases: tuple[ErasedCase, ...]
    lines: tuple[ProbeLine, ...]
    controls: tuple[ControlLine, ...]
    not_ok: tuple[str, ...]

    def count(self, which: str, outcome: str, cases: tuple[ErasedCase, ...] | None = None) -> int:
        chosen = self.cases if cases is None else cases
        return sum(1 for case in chosen if getattr(case, which) == outcome)


def _outcome(name: str, field: Any) -> str:
    """absent | abstained | extracted, for a field whose truth is `absent`."""
    result = score_field(name, None, field).outcome
    return {"correct": ABSENT, "abstained": ABSTAINED}.get(result, EXTRACTED)


def _typed(name: str, text: Any) -> Any:
    return getattr(GroundTruth.model_validate({"doc_id": "x", name: text}), name)


def _record(run_dir: Path, doc_id: str) -> DocRecord:
    path = run_dir / "docs" / f"{doc_id}.json"
    try:
        return DocRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProbeReportError(f"{path}: cannot be read ({exc})") from exc
    except ValidationError as exc:
        raise ProbeReportError(f"{path}: not a document record ({exc})") from exc


def _counts(outcomes: list[str]) -> dict[str, int]:
    return {name: outcomes.count(name) for name in (ABSENT, ABSTAINED, EXTRACTED)}


def analyse(run_dir: Path, probes_dir: Path) -> ProbeReport:
    """The report for the run in `run_dir` over the probe set in `probes_dir`."""
    try:
        run = RunRecord.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProbeReportError(f"{run_dir / 'run.json'}: cannot be read ({exc})") from exc
    except ValidationError as exc:
        raise ProbeReportError(f"{run_dir / 'run.json'}: not a run record ({exc})") from exc
    try:
        document = read_probes(probes_dir)
    except ProbeError as exc:
        raise ProbeReportError(str(exc)) from exc
    entries: list[dict[str, Any]] = document["probes"]
    labels: dict[str, GroundTruth] = {}
    for entry in entries:
        try:
            labels[entry["id"]] = read_label(label_path(probes_dir / "labels", entry["id"]))
        except GroundTruthError as exc:
            raise ProbeReportError(str(exc)) from exc

    records: dict[str, DocRecord] = {}
    not_ok: list[str] = []
    for entry in entries:
        probe_id = entry["id"]
        if probe_id not in run.statuses:
            not_ok.append(probe_id)
            continue
        record = _record(run_dir, probe_id)
        if record.status != "ok" or record.validated is None or record.parsed is None:
            not_ok.append(probe_id)
            continue
        records[probe_id] = record

    controls: list[ControlLine] = []
    control_of: dict[str, str] = {}
    for entry in entries:
        if entry["kind"] != "control":
            continue
        control_of[entry["source"]] = entry["id"]
        record = records.get(entry["id"])
        correct = labeled = 0
        if record is not None and record.validated is not None:
            for name in SCALAR_FIELD_NAMES:
                expected = getattr(labels[entry["id"]], name)
                if expected is None or is_unlabeled(expected):
                    continue
                labeled += 1
                correct += score_field(name, expected, getattr(record.validated, name)).correct
        controls.append(
            ControlLine(
                entry["id"], entry["source"], "ok" if record else "not run", correct, labeled
            )
        )

    cases: list[ErasedCase] = []
    lines: list[ProbeLine] = []
    for entry in entries:
        if entry["kind"] == "control":
            continue
        probe_id = entry["id"]
        record = records.get(probe_id)
        if record is None or record.validated is None or record.parsed is None:
            lines.append(
                ProbeLine(
                    probe_id, entry["source"], entry["layout"], entry["kind"],
                    tuple(entry["erased"]), "not run", {}, {}, 0, 0, 0, 0, 0, 0,
                )
            )  # fmt: skip
            continue
        control = records.get(control_of.get(entry["source"], ""))
        returned_outcomes: list[str] = []
        raw_outcomes: list[str] = []
        for name in entry["erased"]:
            original = entry["original"][name]
            validated_field = getattr(record.validated, name)
            returned = _outcome(name, validated_field)
            raw = _outcome(name, getattr(record.parsed, name))
            returned_outcomes.append(returned)
            raw_outcomes.append(raw)
            recovered = None
            if returned == EXTRACTED:
                recovered = compare_field(
                    name, _typed(name, original), validated_field.value
                ).matched
            control_read = None
            if control is not None and control.validated is not None:
                control_read = score_field(
                    name, _typed(name, original), getattr(control.validated, name)
                ).correct
            cases.append(
                ErasedCase(
                    probe=probe_id,
                    source=entry["source"],
                    kind=entry["kind"],
                    field=name,
                    original=str(original),
                    returned=returned,
                    raw=raw,
                    value=str(validated_field.value) if returned == EXTRACTED else "",
                    recovered=recovered,
                    control_read=control_read,
                )
            )
        label = labels[probe_id]
        kept = [
            name
            for name in SCALAR_FIELD_NAMES
            if getattr(label, name) is not None  # an erased field is null in the label
            and not is_unlabeled(getattr(label, name))
        ]
        rows = score_rows(label.line_items, record.validated.line_items)
        lines.append(
            ProbeLine(
                probe=probe_id,
                source=entry["source"],
                layout=entry["layout"],
                kind=entry["kind"],
                erased=tuple(entry["erased"]),
                status="ok",
                returned=_counts(returned_outcomes),
                raw=_counts(raw_outcomes),
                kept_correct=sum(
                    score_field(name, getattr(label, name), getattr(record.validated, name)).correct
                    for name in kept
                ),
                kept_labeled=len(kept),
                rows_labeled=rows.matched + rows.missed,
                rows_matched=rows.matched,
                rows_missed=rows.missed,
                rows_spurious=rows.spurious,
            )
        )
    return ProbeReport(
        run_id=run.run_id,
        model=run.model,
        prompt_sha256=run.prompt_sha256,
        seed=str(document.get("seed", "")),
        cases=tuple(cases),
        lines=tuple(lines),
        controls=tuple(controls),
        not_ok=tuple(not_ok),
    )


# ---------------------------------------------------------------- markdown


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    low, high = exact_interval(k, n)
    return f"{k}/{n} = {_percent(k / n)} (95% CI {_percent(low)} to {_percent(high)})"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _short(text: str, limit: int = 28) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def to_markdown(report: ProbeReport, *, notes: str | None = None) -> str:
    cases = report.cases
    total = len(cases)
    returned_x = report.count("returned", EXTRACTED)
    raw_x = report.count("raw", EXTRACTED)
    recovered = sum(1 for case in cases if case.recovered is True)
    invented = sum(1 for case in cases if case.recovered is False)
    lines = [
        f"# Hallucination probe: `{report.run_id}`",
        "",
        f"Model `{report.model}`, prompt `{report.prompt_sha256[:12]}`, probe seed "
        f"`{report.seed}`. Generated by `scripts/probe_report.py`; the tables are data, the "
        "reading at the end is hand-written. A probe is a copy of a constructed invoice with a "
        "field painted out (or the last page dropped): the field is NOT on the page, so the "
        "right answer is `absent` or `abstained` and any value is a hallucination.",
        "",
        "## Headline",
        "",
        *_table(
            [
                "",
                "erased fields",
                "absent",
                "abstained",
                "answered with a value",
                "hallucination rate",
            ],
            [
                [
                    "**the system** (validated)",
                    str(total),
                    str(report.count("returned", ABSENT)),
                    str(report.count("returned", ABSTAINED)),
                    str(returned_x),
                    _rate(returned_x, total),
                ],
                [
                    "the model alone (before the validators)",
                    str(total),
                    str(report.count("raw", ABSENT)),
                    str(report.count("raw", ABSTAINED)),
                    str(raw_x),
                    _rate(raw_x, total),
                ],
            ],
        ),
        "",
        (
            f"Of the {returned_x} values the system returned for an erased field, {recovered} are "
            "the value that was erased (read from elsewhere on the page or computed from the "
            f"rows: still a value the document does not state) and {invented} are another value."
            if returned_x
            else "The system returned no value for any erased field."
        ),
        "",
    ]
    if report.not_ok:
        lines += [f"Not run or failed, so not counted: {', '.join(report.not_ok)}.", ""]
    lines += ["## By probe", ""]
    rows = []
    for line in report.lines:
        what = "the last page" if line.kind == "drop_last_page" else ", ".join(line.erased)
        if line.status != "ok":
            rows.append([line.probe, line.source, line.layout, what, "not run", "", "", "", "", ""])
            continue
        rows.append(
            [
                line.probe,
                line.source,
                line.layout,
                what,
                str(len(line.erased)),
                str(line.returned[ABSENT]),
                str(line.returned[ABSTAINED]),
                f"**{line.returned[EXTRACTED]}**" if line.returned[EXTRACTED] else "0",
                f"{line.kept_correct}/{line.kept_labeled}",
                f"{line.rows_matched}/{line.rows_labeled} matched, {line.rows_spurious} spurious"
                if line.kind == "drop_last_page"
                else "",
            ]
        )
    lines += _table(
        ["probe", "source", "layout", "taken away", "erased", "absent", "abstained", "answered",
         "other fields right", "rows"],
        rows,
    )  # fmt: skip
    lines += [
        "",
        "`other fields right`: the labeled fields that were NOT erased, answered correctly (a "
        "system that answers nothing would score zero here). `spurious` rows on a dropped page "
        "are returned rows that are not in the label (it keeps only the rows of the remaining "
        "pages).",
        "",
        "## By field",
        "",
    ]
    by_field: dict[str, list[ErasedCase]] = {}
    for case in cases:
        by_field.setdefault(case.field, []).append(case)
    lines += _table(
        ["field", "erased", "answered with a value", "of which the erased value"],
        [
            [
                name,
                str(len(group)),
                str(sum(1 for c in group if c.returned == EXTRACTED)),
                str(sum(1 for c in group if c.recovered is True)),
            ]
            for name, group in sorted(by_field.items())
        ],
    )
    lines += ["", "## Every erased field", ""]
    lines += _table(
        ["probe", "field", "erased value", "system", "model alone", "returned", "control read it"],
        [
            [
                case.probe,
                case.field,
                _short(case.original),
                f"**{case.returned}**" if case.returned == EXTRACTED else case.returned,
                case.raw,
                (
                    _short(case.value)
                    + (" (the erased value)" if case.recovered else " (another value)")
                    if case.returned == EXTRACTED
                    else ""
                ),
                {True: "yes", False: "no", None: ""}[case.control_read],
            ]
            for case in cases
        ],
    )
    lines += ["", "## Controls (the unmodified documents, same run)", ""]
    lines += _table(
        ["control", "source", "status", "labeled fields right"],
        [[c.probe, c.source, c.status, f"{c.correct}/{c.labeled}"] for c in report.controls],
    )
    lines += [
        "",
        "## Limits",
        "",
        "- The sources are clean synthetic invoices, so this is a floor for the hallucination "
        "rate on real documents, not an estimate of it. Real-document probes (boxes measured in "
        "the label app) are the follow-up.",
        f"- {total} erased fields is a small sample: the interval is wide and no difference "
        "between kinds of erasure is called a finding.",
        "- A returned value equal to the erased one is counted as a hallucination on purpose: "
        "`absent` means the document does not state it, and a value computed from the rows is "
        "not stated. Whether that is the behaviour a client wants is a product decision; the "
        "schema says what it says.",
        "",
    ]
    if notes:
        lines += ["## Reading", "", notes.strip(), ""]
    return "\n".join(lines)
