"""Failure analysis of a run: how it fails by kind, and what the model said before withholding.

`score_run` answers "how many"; this answers "which, and why". Every labeled scalar field of every
ok document becomes a `FieldCase` that holds BOTH the field as returned (`validated`) and the
model's own answer (`parsed`), each scored with the scorer's `score_field`. That separates a field
the system got wrong from a field the model got right and the validators then withheld: the price
of a guard, which the headline accuracy (abstentions count as errors) hides inside one number.

Offline and deterministic. Nothing here changes a score: the counterfactual ("had the raw value
been kept") is arithmetic on the cases, labeled as such, never a result.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from docextract.extract.runner import DocRecord, RunRecord
from docextract.ingest.manifest import read_manifest
from docextract.validate.codes import CONFIDENCE_BELOW_THRESHOLD, HARD_CODES, REASON_CODES
from evalkit.groundtruth import is_unlabeled
from evalkit.gt_store import load_labels
from evalkit.metrics import SCALAR_FIELD_NAMES, FieldScore, score_document, score_field

MODEL_ABSTAINED = "model_abstained"  # the model itself said abstained: no validator code on it
COST = "cost"  # a correct value was withheld
SAVED = "saved"  # a wrong value was withheld
NO_ANSWER = "no_answer"  # the model had no usable answer either


class FailureError(Exception):
    """The run folder cannot be analysed (a missing or unreadable record)."""


@dataclass(frozen=True)
class FieldCase:
    """One labeled scalar field of one ok document, as returned and as the model said it."""

    doc_id: str
    name: str
    expected: Any
    returned: FieldScore  # scored on `validated`: what the scorecard counts
    raw: FieldScore  # scored on `parsed`: what the model said before the validators
    confidence: float  # of the returned field (after the caps)
    raw_confidence: float
    codes: tuple[str, ...]  # validator reason codes on the returned field

    @property
    def withheld(self) -> bool:
        return self.returned.outcome == "abstained"

    @property
    def reason(self) -> str:
        """Why it was withheld: the HARD code, the threshold, or the model itself."""
        for code in self.codes:
            if code in HARD_CODES:
                return code
        if CONFIDENCE_BELOW_THRESHOLD in self.codes:
            return CONFIDENCE_BELOW_THRESHOLD
        return MODEL_ABSTAINED

    @property
    def effect(self) -> str:
        """For a withheld field: did the guard cost a correct value or save a wrong one?"""
        if self.raw.outcome == "correct":
            return COST
        if self.raw.outcome in ("value_wrong", "should_be_absent"):
            return SAVED
        return NO_ANSWER


@dataclass(frozen=True)
class RowCount:
    """The line-item result of one labeled document."""

    doc_id: str
    labeled_rows: int  # matched + missed
    matched: int
    missed: int
    spurious: int
    no_itemization: bool
    abstained_rows: int = 0  # predicted rows the system returned as abstained
    raw_matched: int = 0  # the same pairing, scored on the model's own rows (`parsed`)
    raw_missed: int = 0
    raw_spurious: int = 0


@dataclass(frozen=True)
class RunCases:
    run_id: str
    model: str
    prompt_sha256: str
    cases: list[FieldCase]
    rows: list[RowCount]
    not_ok: list[str]  # labeled documents that produced no output (failed, refused, not run)

    @property
    def documents(self) -> int:
        return len({case.doc_id for case in self.cases})


def read_record(run_dir: Path, doc_id: str) -> DocRecord:
    path = run_dir / "docs" / f"{doc_id}.json"
    try:
        record = DocRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FailureError(f"{path}: cannot be read ({exc})") from exc
    except ValidationError as exc:
        raise FailureError(f"{path}: not a document record ({exc})") from exc
    if record.doc_id != doc_id:
        raise FailureError(f"{path}: holds doc_id {record.doc_id!r}, not {doc_id!r}")
    return record


def collect(run_dir: Path, *, manifest_path: Path, labels_dir: Path) -> RunCases:
    """Every labeled field of every ok document of the run in `run_dir`, as returned and as said."""
    run_path = run_dir / "run.json"
    try:
        run = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FailureError(f"{run_path}: cannot be read ({exc})") from exc
    except ValidationError as exc:
        raise FailureError(f"{run_path}: not a run record ({exc})") from exc
    try:
        manifest = read_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        raise FailureError(f"{manifest_path}: cannot be read as a manifest ({exc})") from exc
    labels, _ = load_labels(manifest, labels_dir)

    cases: list[FieldCase] = []
    rows: list[RowCount] = []
    not_ok: list[str] = []
    for doc_id in sorted(run.statuses):
        resolved = labels.get(doc_id)
        if resolved is None:
            continue  # unlabeled documents are not scored, so not analysed
        record = read_record(run_dir, doc_id)
        if record.status != "ok":
            not_ok.append(doc_id)
            continue
        if record.parsed is None or record.validated is None:
            raise FailureError(f"{run_dir / 'docs' / doc_id}.json: ok but holds no output")
        for name in SCALAR_FIELD_NAMES:
            expected = getattr(resolved.label, name)
            if is_unlabeled(expected):
                continue
            returned_field = getattr(record.validated, name)
            raw_field = getattr(record.parsed, name)
            cases.append(
                FieldCase(
                    doc_id=doc_id,
                    name=name,
                    expected=expected,
                    returned=score_field(name, expected, returned_field),
                    raw=score_field(name, expected, raw_field),
                    confidence=returned_field.confidence,
                    raw_confidence=raw_field.confidence,
                    codes=tuple(r for r in returned_field.reasons if r in REASON_CODES),
                )
            )
        score = score_document(
            resolved.label, owner=resolved.owner, status="ok", document=record.validated
        )
        if score.rows.labeled:
            raw = score_document(
                resolved.label, owner=resolved.owner, status="ok", document=record.parsed
            ).rows
            rows.append(
                RowCount(
                    doc_id=doc_id,
                    labeled_rows=score.rows.matched + score.rows.missed,
                    matched=score.rows.matched,
                    missed=score.rows.missed,
                    spurious=score.rows.spurious,
                    no_itemization=score.rows.no_itemization,
                    abstained_rows=sum(
                        r.status == "abstained" for r in record.validated.line_items
                    ),
                    raw_matched=raw.matched,
                    raw_missed=raw.missed,
                    raw_spurious=raw.spurious,
                )
            )
    return RunCases(
        run_id=run.run_id,
        model=run.model,
        prompt_sha256=run.prompt_sha256,
        cases=cases,
        rows=rows,
        not_ok=not_ok,
    )


# ---------------------------------------------------------------- tables


def outcome_table(cases: list[FieldCase]) -> list[tuple[str, str, int]]:
    """(returned outcome, raw outcome, count), most frequent first."""
    counts = Counter((case.returned.outcome, case.raw.outcome) for case in cases)
    return [(a, b, n) for (a, b), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def withheld_cases(cases: list[FieldCase]) -> list[FieldCase]:
    return sorted((c for c in cases if c.withheld), key=lambda c: (c.doc_id, c.name))


def reason_table(cases: list[FieldCase]) -> list[tuple[str, int, int, int, int]]:
    """For each reason a field was withheld: (reason, withheld, cost, saved, no_answer)."""
    by_reason: dict[str, Counter[str]] = {}
    for case in withheld_cases(cases):
        by_reason.setdefault(case.reason, Counter())[case.effect] += 1
    return [
        (reason, sum(c.values()), c[COST], c[SAVED], c[NO_ANSWER])
        for reason, c in sorted(by_reason.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    ]


def wrong_cases(cases: list[FieldCase]) -> list[FieldCase]:
    """Fields returned with a value (or none) that disagrees with the label: not abstentions."""
    bad = ("value_wrong", "should_be_absent", "should_be_value")
    return sorted((c for c in cases if c.returned.outcome in bad), key=lambda c: (c.doc_id, c.name))


def field_table(cases: list[FieldCase]) -> list[tuple[str, int, int, int, int, int]]:
    """Per field: (name, labeled, correct, withheld-but-raw-right, withheld other, wrong)."""
    rows = []
    for name in SCALAR_FIELD_NAMES:
        mine = [c for c in cases if c.name == name]
        if not mine:
            continue
        correct = sum(c.returned.correct for c in mine)
        cost = sum(c.withheld and c.effect == COST for c in mine)
        other = sum(c.withheld and c.effect != COST for c in mine)
        rows.append((name, len(mine), correct, cost, other, len(mine) - correct - cost - other))
    return rows


def headline(cases: list[FieldCase]) -> tuple[int, int, int]:
    """(labeled, correct as returned, correct had every withheld correct raw value been kept)."""
    labeled = len(cases)
    correct = sum(c.returned.correct for c in cases)
    kept = correct + sum(c.withheld and c.effect == COST for c in cases)
    return labeled, correct, kept


# ---------------------------------------------------------------- markdown


def _short(value: Any, limit: int = 34) -> str:
    text = "(none)" if value is None else str(value).replace("\n", " ").replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _pct(part: int, whole: int) -> str:
    return "n/a" if whole == 0 else f"{part / whole * 100:.1f}%"


def to_markdown(result: RunCases, *, notes: str | None = None) -> str:
    cases = result.cases
    labeled, correct, kept = headline(cases)
    lines = [
        f"# Failure analysis: `{result.run_id}`",
        "",
        f"Model `{result.model}`, prompt `{result.prompt_sha256[:12]}`, {result.documents} labeled "
        f"documents, {labeled} labeled scalar fields. Generated by `scripts/analyze_failures.py`; "
        "the tables are data, the reading at the end is hand-written.",
        "",
    ]
    if result.not_ok:
        lines += [f"Labeled documents with no output: {', '.join(result.not_ok)}.", ""]
    lines += [
        "## Returned versus what the model said",
        "",
        "`returned` is the validated field the scorecard counts; `raw` is the model's own answer "
        "before the validators ran, scored with the same comparator.",
        "",
        "| returned | raw | fields |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {a} | {b} | {n} |" for a, b, n in outcome_table(cases)]

    withheld = withheld_cases(cases)
    lines += [
        "",
        "## Withheld fields (abstentions)",
        "",
        f"{len(withheld)} of {labeled} fields were returned as `abstained`. Effect: `cost` = the "
        "raw value was correct and was withheld; `saved` = the raw value was wrong (or should have "
        "been absent); `no_answer` = the model had no usable answer either.",
        "",
        "| reason | withheld | cost | saved | no answer |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [f"| {r} | {w} | {c} | {s} | {n} |" for r, w, c, s, n in reason_table(cases)]
    lines += [
        "",
        f"Counterfactual, not a score: returned {correct}/{labeled} correct "
        f"({_pct(correct, labeled)}); had every withheld correct raw value been kept it would be "
        f"{kept}/{labeled} ({_pct(kept, labeled)}). The wrong values the guards also stopped are "
        "not counted in this arithmetic, so it is the price of the guards, not their net value.",
        "",
        "| document | field | expected | raw said | raw conf | final conf | reason | effect |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {c.doc_id} | {c.name} | {_short(c.expected)} | {_short(c.raw.predicted)} | "
        f"{c.raw_confidence:.2f} | {c.confidence:.2f} | {c.reason} | {c.effect} |"
        for c in withheld
    ]

    wrong = wrong_cases(cases)
    lines += [
        "",
        "## Wrong or missed values (not abstentions)",
        "",
        f"{len(wrong)} fields.",
        "",
        "| document | field | expected | returned | kind | confidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {c.doc_id} | {c.name} | {_short(c.expected)} | {_short(c.returned.predicted)} | "
        f"{c.returned.error_class or c.returned.outcome} | {c.confidence:.2f} |"
        for c in wrong
    ]

    lines += [
        "",
        "## By field",
        "",
        "| field | labeled | correct | withheld, raw right | withheld, other | wrong |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [f"| {n} | {t} | {c} | {w} | {o} | {x} |" for n, t, c, w, o, x in field_table(cases)]

    total_labeled = sum(r.labeled_rows for r in result.rows)
    lines += [
        "",
        "## Line items",
        "",
        f"{total_labeled} labeled rows in {len(result.rows)} documents with itemization. A missed "
        "row is a labeled row nothing paired with; a spurious row is a predicted row nothing "
        "paired with. An abstained row has no values, so it pairs with nothing: it counts as one "
        "missed and one spurious row. `raw` columns score the model's own rows (before the "
        "validators) with the same row scorer.",
        "",
        "| document | labeled rows | matched | missed | spurious | abstained rows | raw matched "
        "| raw missed | raw spurious |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(result.rows, key=lambda r: (-(r.missed + r.spurious), r.doc_id))
    lines += [
        f"| {r.doc_id} | {r.labeled_rows} | {r.matched} | {r.missed} | {r.spurious} | "
        f"{r.abstained_rows} | {r.raw_matched} | {r.raw_missed} | {r.raw_spurious} |"
        for r in ordered
    ]
    totals = [
        total_labeled,
        *(
            sum(getattr(r, name) for r in result.rows)
            for name in (
                "matched",
                "missed",
                "spurious",
                "abstained_rows",
                "raw_matched",
                "raw_missed",
                "raw_spurious",
            )
        ),
    ]
    lines += ["| **all** | " + " | ".join(str(n) for n in totals) + " |"]
    if notes:
        lines += ["", "## Reading", "", notes.strip()]
    return "\n".join(lines) + "\n"
