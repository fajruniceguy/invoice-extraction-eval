"""Scorecard aggregation and reporting.

Turns a run folder into an offline, deterministic verdict: `score_run` scores every document of
`outputs/runs/<run_id>/` that has a ground-truth label, aggregates the field, row and error
numbers overall, per field, per slice (`split`, `doc_type`, `capture` from the manifest) and per
document, and writes

  <out>/manifest.json     the run manifest: git SHA and dirtiness, prompt/schema hashes, model,
                          config and selection from run.json, plus the GT hash, the dataset hash
                          and the comparator settings used
  <out>/scorecard.json    every number, sorted keys (byte-stable for the same inputs)
  <out>/scorecard.md      the same, for a human
  <out>/docs/<doc_id>.json  the per-document result: each labeled field and every row pair

The hashes say exactly what was scored: `gt_hash` is over the OWNER label files actually used
(borrowers add nothing), `dataset_hash` over the manifest bytes plus each scored document's stored
page hashes from its sidecar (ingest.json, or degrade.json for a variant) — metadata only, pages
are never read. `now` and the git probe are injectable so tests can pin an entire scorecard.
"""

import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from docextract.extract.runner import DocRecord, RunRecord
from docextract.ingest.degrade import read_sidecar_named
from docextract.ingest.manifest import read_manifest
from docextract.ingest.pipeline import read_sidecar, sha256_hex
from evalkit.comparators import FUZZY_NAME_MIN, NUMERIC_ABS_TOL, NUMERIC_REL_TOL
from evalkit.gt_store import ResolvedLabel, load_labels
from evalkit.metrics import (
    DocumentScore,
    FieldMetrics,
    FieldScore,
    RowsScore,
    score_document,
)
from evalkit.perf import PerfStats
from evalkit.taxonomy import ErrorClass, FieldOutcome, count_errors

SLICE_COLUMNS: tuple[str, ...] = ("split", "doc_type", "capture")
NO_VALUE = "(none)"
TARGET = "validated"


class ScorecardError(Exception):
    """The run folder cannot be scored (a missing or unreadable record, no manifest)."""


# ---------------------------------------------------------------- hashes


def _text_hash(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def _gt_hash(owners: Iterable[str], labels_dir: Path) -> str:
    """One line per OWNER label file used: its doc_id and the hash of its bytes as stored."""
    lines = []
    for owner in sorted(set(owners)):
        path = labels_dir / f"{owner}.json"
        try:
            digest = sha256_hex(path.read_bytes())
        except OSError as exc:
            raise ScorecardError(
                f"{path}: the label file of {owner} cannot be read ({exc})"
            ) from exc
        lines.append(f"{owner}\t{digest}")
    return _text_hash("\n".join(lines) + "\n")


def _page_hashes(doc_id: str, pages_dir: Path, *, derived: bool) -> str:
    """The page hashes this document's sidecar records, or "(none)" when it has no sidecar."""
    folder = pages_dir / doc_id
    try:
        sidecar = read_sidecar_named(folder) if derived else read_sidecar(folder)
        pages = sidecar["pages"] if sidecar else None
        if not pages:
            return NO_VALUE
        return "\t".join(str(page["sha256"]) for page in pages)
    except (KeyError, TypeError, OSError, ValueError):
        return NO_VALUE


def _dataset_hash(manifest_path: Path, docs: Sequence[tuple[str, bool]], pages_dir: Path) -> str:
    """The manifest's bytes plus the stored-page hashes of the scored documents, in doc_id order."""
    try:
        manifest_digest = sha256_hex(manifest_path.read_bytes())
    except OSError as exc:
        raise ScorecardError(f"{manifest_path}: cannot be read ({exc})") from exc
    lines = [manifest_digest]
    lines.extend(
        f"{doc_id}\t{_page_hashes(doc_id, pages_dir, derived=derived)}"
        for doc_id, derived in sorted(docs)
    )
    return _text_hash("\n".join(lines) + "\n")


def _git() -> tuple[str, bool]:
    """(HEAD sha, working tree dirty), or ("unknown", False) when git cannot say."""

    def run(*arguments: str) -> str:
        try:
            result = subprocess.run(["git", *arguments], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    sha = run("rev-parse", "HEAD")
    if not sha:
        return "unknown", False
    return sha, bool(run("status", "--porcelain"))


# ---------------------------------------------------------------- json views


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "isoformat"):  # date, datetime
        return value.isoformat()
    return str(value)


def metrics_dict(metrics: FieldMetrics) -> dict[str, Any]:
    return {
        "labeled": metrics.labeled,
        "answered": metrics.answered,
        "correct": metrics.correct,
        "value_wrong": metrics.value_wrong,
        "should_be_absent": metrics.should_be_absent,
        "should_be_value": metrics.should_be_value,
        "abstained": metrics.abstained,
        "no_prediction": metrics.no_prediction,
        "accuracy": metrics.accuracy,
        "selective_accuracy": metrics.selective_accuracy,
        "coverage": metrics.coverage,
        "abstention_rate": metrics.abstention_rate,
        "failure_rate": metrics.failure_rate,
    }


def rows_dict(rows: RowsScore) -> dict[str, Any]:
    return {
        "labeled": rows.labeled,
        "no_itemization": rows.no_itemization,
        "matched": rows.matched,
        "missed": rows.missed,
        "spurious": rows.spurious,
        "perfect": rows.perfect,
        "cells_compared": rows.cells_compared,
        "cells_correct": rows.cells_correct,
        "row_precision": rows.row_precision,
        "row_recall": rows.row_recall,
        "row_f1": rows.row_f1,
        "row_accuracy": rows.row_accuracy,
        "cell_accuracy": rows.cell_accuracy,
        "errors": list(rows.errors),
    }


def field_dict(score: FieldScore) -> dict[str, Any]:
    return {
        "name": score.name,
        "expected": _json_value(score.expected),
        "predicted": _json_value(score.predicted),
        "predicted_status": score.predicted_status,
        "outcome": score.outcome,
        "error_class": score.error_class,
        "how": score.how,
        "similarity": score.similarity,
    }


def document_dict(score: DocumentScore) -> dict[str, Any]:
    return {
        "doc_id": score.doc_id,
        "owner": score.owner,
        "status": score.status,
        "metrics": metrics_dict(score.metrics),
        "errors": score.errors,
        "rows": rows_dict(score.rows),
        "fields": [field_dict(field) for field in score.fields],
    }


def _dumps(payload: Any) -> str:
    """Sorted keys, so the same inputs always give the same bytes."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def money(value: Decimal | None, places: int = 4) -> str:
    return "n/a" if value is None else f"${value:.{places}f}"


def seconds(milliseconds: float | None) -> str:
    return "n/a" if milliseconds is None else f"{milliseconds / 1000:.1f} s"


def perf_dict(stats: PerfStats) -> dict[str, Any]:
    return {
        "documents": stats.documents,
        "ok": stats.ok,
        "failed": stats.failed,
        "live_calls": stats.live_calls,
        "cached_documents": stats.cached_documents,
        "cost_usd": str(stats.cost_usd),
        "list_cost_usd": str(stats.list_cost_usd),
        "cost_per_doc": _json_value(stats.cost_per_doc),
        "list_cost_per_doc": _json_value(stats.list_cost_per_doc),
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "repairs": stats.repairs,
        "latency_ms": {
            "p50": stats.latency_p50_ms,
            "p95": stats.latency_p95_ms,
            "mean": stats.latency_mean_ms,
        },
    }


# ---------------------------------------------------------------- the scorecard


@dataclass(frozen=True)
class Scorecard:
    manifest: dict[str, Any]
    overall: FieldMetrics
    fields: dict[str, FieldMetrics]
    rows: RowsScore
    errors: dict[str, int]
    slices: dict[str, dict[str, FieldMetrics]]
    docs: list[DocumentScore]
    unscored: list[str]
    perf: PerfStats = dataclass_field(default_factory=PerfStats)
    perf_slices: dict[str, dict[str, PerfStats]] = dataclass_field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "overall": metrics_dict(self.overall),
            "fields": {name: metrics_dict(m) for name, m in self.fields.items()},
            "rows": rows_dict(self.rows),
            "errors": self.errors,
            "slices": {
                column: {value: metrics_dict(m) for value, m in groups.items()}
                for column, groups in self.slices.items()
            },
            "docs": {
                score.doc_id: {
                    "status": score.status,
                    "metrics": metrics_dict(score.metrics),
                    "errors": score.errors,
                }
                for score in self.docs
            },
            "unscored": self.unscored,
            "perf": perf_dict(self.perf),
            "perf_slices": {
                column: {value: perf_dict(stats) for value, stats in groups.items()}
                for column, groups in self.perf_slices.items()
            },
        }

    def to_json(self) -> str:
        return _dumps(self.payload())

    def to_markdown(self) -> str:
        manifest = self.manifest
        lines = [
            f"# Scorecard: run {manifest['run_id']}",
            "",
            f"- model: `{manifest['model']}`",
            f"- prompt sha256: `{manifest['prompt_sha256']}`",
            f"- git: `{manifest['git_sha']}`{' (dirty)' if manifest['git_dirty'] else ''}",
            f"- gt hash: `{manifest['gt_hash']}`",
            f"- dataset hash: `{manifest['dataset_hash']}`",
            f"- scored: {len(self.docs)} document(s); unscored (no label): {len(self.unscored)}",
            f"- scored at: {manifest['scored_at']}",
            "",
            "## Fields",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| labeled | {self.overall.labeled} |",
            f"| correct | {self.overall.correct} |",
            f"| accuracy (abstentions are errors) | {self.overall.accuracy:.3f} |",
            f"| selective accuracy | {self.overall.selective_accuracy:.3f} |",
            f"| coverage | {self.overall.coverage:.3f} |",
            f"| abstention rate | {self.overall.abstention_rate:.3f} |",
            f"| no prediction rate | {self.overall.failure_rate:.3f} |",
            "",
            "| field | labeled | correct | accuracy | selective | coverage |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, metrics in self.fields.items():
            lines.append(
                f"| {name} | {metrics.labeled} | {metrics.correct} | {metrics.accuracy:.3f} "
                f"| {metrics.selective_accuracy:.3f} | {metrics.coverage:.3f} |"
            )
        lines.extend(
            [
                "",
                "## Rows",
                "",
                f"- matched {self.rows.matched}, missed {self.rows.missed}, "
                f"spurious {self.rows.spurious}, perfect {self.rows.perfect}",
                f"- precision {self.rows.row_precision:.3f}, recall {self.rows.row_recall:.3f}, "
                f"F1 {self.rows.row_f1:.3f}, row accuracy {self.rows.row_accuracy:.3f}, "
                f"cell accuracy {self.rows.cell_accuracy:.3f}",
                "",
                "## Errors",
                "",
            ]
        )
        if self.errors:
            lines.extend(f"- {name}: {count}" for name, count in self.errors.items())
        else:
            lines.append("- none")
        for column, groups in self.slices.items():
            lines.extend(
                [
                    "",
                    f"## Slice: {column}",
                    "",
                    "| value | labeled | accuracy | selective | coverage |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for value, metrics in groups.items():
                lines.append(
                    f"| {value} | {metrics.labeled} | {metrics.accuracy:.3f} "
                    f"| {metrics.selective_accuracy:.3f} | {metrics.coverage:.3f} |"
                )
        lines.extend(self._perf_markdown())
        unscored = ", ".join(self.unscored) if self.unscored else "none"
        lines.extend(["", f"Unscored documents (no ground truth): {unscored}", ""])
        return "\n".join(lines)

    def _perf_markdown(self) -> list[str]:
        """Cost and latency, overall and per slice (see evalkit/perf.py for what is counted)."""
        lines = [
            "",
            "## Cost and latency",
            "",
            "Documents attempted (ok or failed); latency is over live calls only, nearest-rank.",
            "",
            "| scope | docs | live | cached | paid | at list price | per doc (list) "
            "| tokens in/out | latency p50 | p95 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        scopes = [("all", self.perf)]
        for column, groups in self.perf_slices.items():
            scopes.extend((f"{column}={value}", stats) for value, stats in groups.items())
        for name, stats in scopes:
            lines.append(
                f"| {name} | {stats.documents} | {stats.live_calls} | {stats.cached_documents} "
                f"| {money(stats.cost_usd)} | {money(stats.list_cost_usd)} "
                f"| {money(stats.list_cost_per_doc)} "
                f"| {stats.input_tokens}/{stats.output_tokens} "
                f"| {seconds(stats.latency_p50_ms)} | {seconds(stats.latency_p95_ms)} |"
            )
        return lines

    def write(self, out_dir: Path) -> list[Path]:
        """Write the manifest, the scorecard (JSON + markdown) and one file per scored document."""
        out_dir.mkdir(parents=True, exist_ok=True)
        written = [
            _write(out_dir / "manifest.json", _dumps(self.manifest)),
            _write(out_dir / "scorecard.json", self.to_json()),
            _write(out_dir / "scorecard.md", self.to_markdown()),
        ]
        for score in self.docs:
            written.append(
                _write(out_dir / "docs" / f"{score.doc_id}.json", _dumps(document_dict(score)))
            )
        return written


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _read_record(run_dir: Path, doc_id: str) -> DocRecord:
    path = run_dir / "docs" / f"{doc_id}.json"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScorecardError(f"{path}: cannot be read ({exc})") from exc
    try:
        record = DocRecord.model_validate_json(text)
    except ValidationError as exc:
        raise ScorecardError(f"{path}: not a document record ({exc})") from exc
    if record.doc_id != doc_id:
        raise ScorecardError(f"{path}: holds doc_id {record.doc_id!r}, not {doc_id!r}")
    return record


def score_run(
    run_dir: Path,
    *,
    manifest_path: Path,
    labels_dir: Path,
    pages_dir: Path,
    now: datetime | None = None,
    git_sha: str | None = None,
    git_dirty: bool | None = None,
) -> Scorecard:
    """Score the run in `run_dir` against the labels. Deterministic for the same inputs (and the
    same injected `now`/`git_sha`)."""
    run_path = run_dir / "run.json"
    try:
        run = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScorecardError(f"{run_path}: cannot be read ({exc})") from exc
    except ValidationError as exc:
        raise ScorecardError(f"{run_path}: not a run record ({exc})") from exc

    try:
        rows = read_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        raise ScorecardError(f"{manifest_path}: cannot be read as a manifest ({exc})") from exc
    by_id = {row.doc_id: row for row in rows}
    labels, _ = load_labels(rows, labels_dir)

    docs: list[DocumentScore] = []
    unscored: list[str] = []
    by_field: dict[str, list[FieldOutcome]] = {}
    by_slice: dict[str, dict[str, list[FieldOutcome]]] = {column: {} for column in SLICE_COLUMNS}
    all_errors: list[ErrorClass] = []
    owners: list[str] = []
    page_scope: list[tuple[str, bool]] = []
    attempted: list[tuple[str, DocRecord]] = []  # ok or failed: a call was made

    for doc_id in run.statuses:
        resolved: ResolvedLabel | None = labels.get(doc_id)
        if resolved is None:
            unscored.append(doc_id)
            if run.statuses[doc_id] in ("ok", "failed"):
                attempted.append((doc_id, _read_record(run_dir, doc_id)))
            continue
        row = by_id.get(doc_id)
        if row is None:
            raise ScorecardError(f"{doc_id}: the run holds a document that is not in the manifest")
        record = _read_record(run_dir, doc_id)
        if record.status in ("ok", "failed"):
            attempted.append((doc_id, record))
        if record.status == "ok" and record.validated is None:
            raise ScorecardError(f"{run_dir / 'docs' / doc_id}.json: ok but holds no output")
        document = record.validated if record.status == "ok" else None
        score = score_document(
            resolved.label, owner=resolved.owner, status=record.status, document=document
        )
        docs.append(score)
        owners.append(resolved.owner)
        page_scope.append((doc_id, bool(row.is_derived)))
        for field in score.fields:
            by_field.setdefault(field.name, []).append(field.outcome)
            for column in SLICE_COLUMNS:
                value = getattr(row, column) or NO_VALUE
                by_slice[column].setdefault(value, []).append(field.outcome)
            if field.error_class is not None:
                all_errors.append(field.error_class)
        all_errors.extend(score.rows.errors)

    sha, dirty = _git() if git_sha is None else (git_sha, bool(git_dirty))
    scored_at = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    manifest: dict[str, Any] = {
        "run_id": run.run_id,
        "scored_at": scored_at,
        "target": TARGET,
        "model": run.model,
        "endpoint": run.endpoint,
        "prompt_sha256": run.prompt_sha256,
        "schema_sha256": run.schema_sha256,
        "settings": run.settings,
        "selection": run.selection,
        "git_sha": sha,
        "git_dirty": dirty,
        "gt_hash": _gt_hash(owners, labels_dir),
        "dataset_hash": _dataset_hash(manifest_path, page_scope, pages_dir),
        "comparators": {
            "fuzzy_name_min": str(FUZZY_NAME_MIN),
            "numeric_rel_tol": str(NUMERIC_REL_TOL),
            "numeric_abs_tol": str(NUMERIC_ABS_TOL),
        },
    }
    perf_groups: dict[str, dict[str, list[DocRecord]]] = {column: {} for column in SLICE_COLUMNS}
    for doc_id, record in attempted:
        row = by_id.get(doc_id)
        for column in SLICE_COLUMNS:
            value = (getattr(row, column) if row is not None else None) or NO_VALUE
            perf_groups[column].setdefault(value, []).append(record)
    return Scorecard(
        manifest=manifest,
        overall=FieldMetrics.from_outcomes(
            [outcome for outcomes in by_field.values() for outcome in outcomes]
        ),
        fields={name: FieldMetrics.from_outcomes(outcomes) for name, outcomes in by_field.items()},
        rows=RowsScore.merge(score.rows for score in docs),
        errors=count_errors(all_errors),
        slices={
            column: {
                value: FieldMetrics.from_outcomes(outcomes) for value, outcomes in groups.items()
            }
            for column, groups in by_slice.items()
        },
        docs=docs,
        unscored=unscored,
        perf=PerfStats.from_records(record for _, record in attempted),
        perf_slices={
            column: {value: PerfStats.from_records(records) for value, records in groups.items()}
            for column, groups in perf_groups.items()
        },
    )
