import dataclasses
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from docextract.extract.runner import DocRecord, RunRecord, RunTotals
from docextract.extract.schema import SCALAR_FIELD_NAMES, DocumentExtraction, Field, LineItem
from docextract.ingest.manifest import ManifestRow, write_manifest
from evalkit.groundtruth import GroundTruth, GTLineItem
from evalkit.gt_store import write_label
from evalkit.scorecard import ScorecardError, score_run

NOW = datetime(2026, 10, 7, 12, 0, 0, tzinfo=UTC)
PROMPT = "p" * 64
SCHEMA = "s" * 64


def extracted(value: object) -> Field:
    return Field(value=value, confidence=0.9, status="extracted", evidence="quote", page=1)


def absent() -> Field:
    return Field(value=None, confidence=0.9, status="absent")


def abstained() -> Field:
    return Field(value=None, confidence=0.2, status="abstained", reasons=["low_confidence"])


def document(**overrides: object) -> DocumentExtraction:
    fields: dict[str, object] = {name: absent() for name in SCALAR_FIELD_NAMES}
    fields.update(overrides)
    rows = fields.pop("line_items", [])
    return DocumentExtraction(line_items=rows, **fields)


def label(doc_id: str, **fields: object) -> GroundTruth:
    return GroundTruth(doc_id=doc_id, label_mode="prelabel", labeled_at=NOW, **fields)


def dataset(
    tmp_path: Path, rows: list[ManifestRow], labels: list[GroundTruth]
) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "data" / "manifest.csv"
    write_manifest(manifest, rows)
    labels_dir = tmp_path / "ground_truth"
    for one in labels:
        write_label(labels_dir, one)
    pages_dir = tmp_path / "data" / "pages"
    for manifest_row in rows:
        folder = pages_dir / manifest_row.doc_id
        folder.mkdir(parents=True, exist_ok=True)
        sidecar = {"pages": [{"sha256": f"{manifest_row.doc_id}-p1"}]}
        name = "degrade.json" if manifest_row.is_derived else "ingest.json"
        (folder / name).write_text(json.dumps(sidecar), encoding="utf-8")
    return manifest, labels_dir, pages_dir


def run_dir(tmp_path: Path, records: list[DocRecord], *, name: str = "test-run") -> Path:
    folder = tmp_path / "outputs" / "runs" / name
    docs = folder / "docs"
    docs.mkdir(parents=True)
    for record in records:
        (docs / f"{record.doc_id}.json").write_text(
            record.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    run = RunRecord(
        run_id=name,
        started_at="2026-10-07T11:00:00+00:00",
        finished_at="2026-10-07T11:01:00+00:00",
        model="fake-model",
        endpoint="http://example.invalid/v1",
        prompt_sha256=PROMPT,
        schema_sha256=SCHEMA,
        settings={"cache": "off", "max_pages": 15},
        selection={"split": "dev"},
        statuses={record.doc_id: record.status for record in records},
        totals=RunTotals(),
        budget_spent_usd=Decimal(0),
    )
    (folder / "run.json").write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return folder


def two_document_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rows = [
        ManifestRow(
            doc_id="doc_001",
            raw_files=("doc_001.pdf",),
            content_group="doc_001",
            doc_type="invoice",
            capture="born_digital",
            split="dev",
        ),
        ManifestRow(
            doc_id="doc_002",
            raw_files=("doc_002.pdf",),
            content_group="doc_002",
            doc_type="faktur",
            capture="born_digital",
            split="test",
        ),
    ]
    labels = [
        label("doc_001", total=Decimal("10"), vendor_name="Acme"),
        label("doc_002", total=Decimal("20"), vendor_name="Beta"),
    ]
    manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)
    records = [
        DocRecord(
            doc_id="doc_001",
            status="ok",
            validated=document(total=extracted(Decimal("10")), vendor_name=extracted("Acme")),
        ),
        DocRecord(
            doc_id="doc_002",
            status="ok",
            validated=document(total=extracted(Decimal("25")), vendor_name=abstained()),
        ),
    ]
    folder = run_dir(tmp_path, records)
    return folder, manifest, labels_dir, pages_dir


def scored(tmp_path: Path, *, now: datetime = NOW, git_sha: str = "a" * 40):
    folder, manifest, labels_dir, pages_dir = two_document_fixture(tmp_path)
    card = score_run(
        folder,
        manifest_path=manifest,
        labels_dir=labels_dir,
        pages_dir=pages_dir,
        now=now,
        git_sha=git_sha,
        git_dirty=False,
    )
    return card, folder, manifest, labels_dir, pages_dir


class TestScoreRun:
    def test_the_run_is_scored_field_by_field(self, tmp_path: Path) -> None:
        card, *_ = scored(tmp_path)
        assert card.overall.labeled == 4
        assert card.overall.correct == 2
        assert card.overall.accuracy == pytest.approx(0.5)
        assert card.overall.answered == 3
        assert card.overall.selective_accuracy == pytest.approx(2 / 3)
        assert card.overall.coverage == pytest.approx(0.75)
        assert card.overall.abstention_rate == pytest.approx(0.25)

    def test_the_error_taxonomy_is_counted(self, tmp_path: Path) -> None:
        card, *_ = scored(tmp_path)
        assert card.errors == {"numeric_mismatch": 1, "abstained_on_value": 1}

    def test_every_field_gets_its_own_line(self, tmp_path: Path) -> None:
        card, *_ = scored(tmp_path)
        assert set(card.fields) == {"total", "vendor_name"}
        assert card.fields["total"].labeled == 2
        assert card.fields["vendor_name"].correct == 1

    def test_the_manifest_records_the_run_identity(self, tmp_path: Path) -> None:
        card, *_ = scored(tmp_path)
        manifest = card.manifest
        assert manifest["run_id"] == "test-run"
        assert manifest["model"] == "fake-model"
        assert manifest["prompt_sha256"] == PROMPT
        assert manifest["target"] == "validated"
        assert manifest["git_sha"] == "a" * 40 and manifest["git_dirty"] is False
        assert manifest["scored_at"].startswith("2026-10-07T12:00:00")
        assert len(manifest["gt_hash"]) == 64
        assert len(manifest["dataset_hash"]) == 64
        assert manifest["comparators"]["fuzzy_name_min"] == "0.9"

    def test_the_slices_come_from_the_manifest(self, tmp_path: Path) -> None:
        card, *_ = scored(tmp_path)
        assert set(card.slices["split"]) == {"dev", "test"}
        assert card.slices["doc_type"]["faktur"].correct == 0
        assert card.slices["capture"]["born_digital"].labeled == 4

    def test_a_document_without_a_label_is_unscored_not_an_error(self, tmp_path: Path) -> None:
        rows = [
            ManifestRow(
                doc_id="doc_001",
                raw_files=("doc_001.pdf",),
                content_group="doc_001",
                doc_type="invoice",
                capture="born_digital",
                split="dev",
            ),
            ManifestRow(
                doc_id="doc_002",
                raw_files=("doc_002.pdf",),
                content_group="doc_002",
                doc_type="faktur",
                capture="born_digital",
                split="test",
            ),
        ]
        labels = [label("doc_001", total=Decimal("10"))]
        manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)
        folder = run_dir(
            tmp_path,
            [
                DocRecord(
                    doc_id="doc_001",
                    status="ok",
                    validated=document(total=extracted(Decimal("10"))),
                ),
                DocRecord(
                    doc_id="doc_002",
                    status="ok",
                    validated=document(total=extracted(Decimal("20"))),
                ),
            ],
        )
        card = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert card.unscored == ["doc_002"]
        assert [score.doc_id for score in card.docs] == ["doc_001"]

    def test_a_failed_document_scores_every_label_as_no_prediction(self, tmp_path: Path) -> None:
        rows = [
            ManifestRow(
                doc_id="doc_003",
                raw_files=("doc_003.pdf",),
                content_group="doc_003",
                doc_type="invoice",
                capture="born_digital",
                split="dev",
            )
        ]
        labels = [label("doc_003", total=Decimal("10"), vendor_name="Acme")]
        manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)
        folder = run_dir(tmp_path, [DocRecord(doc_id="doc_003", status="failed", error="boom")])
        card = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert card.overall.no_prediction == 2
        assert card.overall.accuracy == 0.0
        assert card.errors == {"no_prediction": 2}

    def test_an_ok_record_without_output_is_a_structural_error(self, tmp_path: Path) -> None:
        folder, manifest, labels_dir, pages_dir = two_document_fixture(tmp_path)
        (folder / "docs" / "doc_001.json").write_text(
            DocRecord(doc_id="doc_001", status="ok").model_dump_json(indent=2),
            encoding="utf-8",
        )
        with pytest.raises(ScorecardError):
            score_run(
                folder,
                manifest_path=manifest,
                labels_dir=labels_dir,
                pages_dir=pages_dir,
                now=NOW,
                git_sha="a" * 40,
            )

    def test_a_document_of_the_run_missing_its_record_is_a_structural_error(
        self, tmp_path: Path
    ) -> None:
        folder, manifest, labels_dir, pages_dir = two_document_fixture(tmp_path)
        (folder / "docs" / "doc_002.json").unlink()
        with pytest.raises(ScorecardError):
            score_run(
                folder,
                manifest_path=manifest,
                labels_dir=labels_dir,
                pages_dir=pages_dir,
                now=NOW,
                git_sha="a" * 40,
            )

    def test_a_variant_takes_its_owners_label(self, tmp_path: Path) -> None:
        rows = [
            ManifestRow(
                doc_id="doc_001",
                raw_files=("doc_001.pdf",),
                content_group="doc_001",
                doc_type="invoice",
                capture="born_digital",
                split="dev",
            ),
            ManifestRow(
                doc_id="doc_001-deg-light",
                capture="synthetic_degraded",
                variant="light",
                gt_from="doc_001",
                content_group="doc_001",
                doc_type="invoice",
                split="dev",
            ),
        ]
        labels = [label("doc_001", total=Decimal("10"))]
        manifest, labels_dir, pages_dir = dataset(tmp_path, rows, labels)
        folder = run_dir(
            tmp_path,
            [
                DocRecord(
                    doc_id="doc_001-deg-light",
                    status="ok",
                    validated=document(total=extracted(Decimal("10"))),
                )
            ],
        )
        card = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert card.overall.correct == 1
        assert card.docs[0].owner == "doc_001"

    def test_a_run_with_no_labels_at_all_scores_nothing(self, tmp_path: Path) -> None:
        rows = [
            ManifestRow(
                doc_id="doc_004",
                raw_files=("doc_004.pdf",),
                content_group="doc_004",
                doc_type="invoice",
                split="dev",
            )
        ]
        manifest, labels_dir, pages_dir = dataset(tmp_path, rows, [])
        folder = run_dir(tmp_path, [DocRecord(doc_id="doc_004", status="ok")])
        card = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert card.docs == [] and card.unscored == ["doc_004"]
        assert "Unscored documents (no ground truth): doc_004" in card.to_markdown()


class TestArtifacts:
    def test_write_creates_the_manifest_scorecard_and_documents(self, tmp_path: Path) -> None:
        card, folder, *_ = scored(tmp_path)
        out = folder / "score"
        written = card.write(out)
        names = {path.relative_to(out).as_posix() for path in written}
        assert names == {
            "manifest.json",
            "scorecard.json",
            "scorecard.md",
            "docs/doc_001.json",
            "docs/doc_002.json",
        }
        assert all(path.is_file() for path in written)
        payload = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
        assert payload["overall"]["labeled"] == 4
        assert payload["errors"] == {"abstained_on_value": 1, "numeric_mismatch": 1}
        assert set(payload["docs"]) == {"doc_001", "doc_002"}
        per_doc = json.loads((out / "docs" / "doc_002.json").read_text(encoding="utf-8"))
        assert {field["name"] for field in per_doc["fields"]} == {"total", "vendor_name"}
        assert per_doc["errors"] == {"abstained_on_value": 1, "numeric_mismatch": 1}

    def test_the_markdown_reports_the_errors(self, tmp_path: Path) -> None:
        card, *_ = scored(tmp_path)
        markdown = card.to_markdown()
        assert "- numeric_mismatch: 1" in markdown
        assert "| accuracy (abstentions are errors) | 0.500 |" in markdown
        assert "## Slice: split" in markdown

    def test_scoring_the_same_inputs_twice_gives_the_same_bytes(self, tmp_path: Path) -> None:
        first, folder, manifest, labels_dir, pages_dir = scored(tmp_path)
        second = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert first.to_json() == second.to_json()
        assert first.to_markdown() == second.to_markdown()

    def test_a_later_scoring_time_changes_only_the_timestamp(self, tmp_path: Path) -> None:
        first, folder, manifest, labels_dir, pages_dir = scored(tmp_path)
        later = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW + timedelta(hours=1),
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert first.to_json() != later.to_json()
        assert first.manifest["gt_hash"] == later.manifest["gt_hash"]

    def test_editing_a_label_changes_the_gt_hash_only(self, tmp_path: Path) -> None:
        card, folder, manifest, labels_dir, pages_dir = scored(tmp_path)
        write_label(labels_dir, label("doc_001", total=Decimal("11"), vendor_name="Acme"))
        edited = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert edited.manifest["gt_hash"] != card.manifest["gt_hash"]
        assert edited.manifest["dataset_hash"] == card.manifest["dataset_hash"]

    def test_editing_the_manifest_changes_the_dataset_hash(self, tmp_path: Path) -> None:
        card, folder, manifest, labels_dir, pages_dir = scored(tmp_path)
        rows = [
            ManifestRow(
                doc_id="doc_001",
                raw_files=("doc_001.pdf",),
                content_group="doc_001",
                doc_type="receipt",
                capture="born_digital",
                split="dev",
            ),
            ManifestRow(
                doc_id="doc_002",
                raw_files=("doc_002.pdf",),
                content_group="doc_002",
                doc_type="faktur",
                capture="born_digital",
                split="test",
            ),
        ]
        write_manifest(manifest, rows)
        edited = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert edited.manifest["dataset_hash"] != card.manifest["dataset_hash"]
        assert edited.manifest["gt_hash"] == card.manifest["gt_hash"]

    def test_a_changed_stored_page_changes_the_dataset_hash(self, tmp_path: Path) -> None:
        card, folder, manifest, labels_dir, pages_dir = scored(tmp_path)
        sidecar = pages_dir / "doc_001" / "ingest.json"
        sidecar.write_text(json.dumps({"pages": [{"sha256": "changed"}]}), encoding="utf-8")
        edited = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert edited.manifest["dataset_hash"] != card.manifest["dataset_hash"]

    def test_the_per_document_file_renders_values_as_strings(self, tmp_path: Path) -> None:
        card, folder, *_ = scored(tmp_path)
        card.write(folder / "score")
        per_doc = json.loads(
            (folder / "score" / "docs" / "doc_002.json").read_text(encoding="utf-8")
        )
        total = next(field for field in per_doc["fields"] if field["name"] == "total")
        assert total["expected"] == "20"
        assert total["predicted"] == "25"
        assert total["outcome"] == "value_wrong"

    def test_a_missing_manifest_is_a_structural_error(self, tmp_path: Path) -> None:
        folder, manifest, labels_dir, pages_dir = two_document_fixture(tmp_path)
        manifest.unlink()
        with pytest.raises(ScorecardError):
            score_run(
                folder,
                manifest_path=manifest,
                labels_dir=labels_dir,
                pages_dir=pages_dir,
                now=NOW,
                git_sha="a" * 40,
            )

    def test_a_run_folder_without_its_run_record_is_an_error(self, tmp_path: Path) -> None:
        folder, manifest, labels_dir, pages_dir = two_document_fixture(tmp_path)
        (folder / "run.json").unlink()
        with pytest.raises(ScorecardError):
            score_run(
                folder,
                manifest_path=manifest,
                labels_dir=labels_dir,
                pages_dir=pages_dir,
                now=NOW,
                git_sha="a" * 40,
            )


# ------------------------------------------------------------------ cost and latency


class TestPerf:
    """doc_001 (dev) 1000 ms and $0.10, doc_002 (test) 3000 ms and $0.20, doc_003 (not in the
    manifest, no label) 5000 ms and $0.30, a refused doc_004 that must not count."""

    def build(self, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        folder, manifest, labels_dir, pages_dir = two_document_fixture(tmp_path)

        def timed(doc_id: str, latency: float, cost: str, **extra: object) -> DocRecord:
            base = {"cost_usd": Decimal(cost), "latency_ms": latency}
            base.update(extra)
            return DocRecord(
                doc_id=doc_id,
                status="ok",
                validated=document(total=extracted(Decimal("10"))),
                usage={"input_tokens": 100, "billable_output_tokens": 50},
                **base,  # type: ignore[arg-type]
            )

        records = [
            timed("doc_001", 1000.0, "0.10"),
            timed("doc_002", 3000.0, "0.20"),
            timed("doc_003", 5000.0, "0.30"),
            DocRecord(
                doc_id="doc_004", status="refused", error="not ingested", cost_usd=Decimal(9)
            ),
        ]
        folder = run_dir(tmp_path, records, name="perf-run")
        return folder, manifest, labels_dir, pages_dir

    def score(self, tmp_path: Path):  # noqa: ANN202
        folder, manifest, labels_dir, pages_dir = self.build(tmp_path)
        return score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )

    def test_every_attempted_document_counts_labeled_or_not_but_not_a_refused_one(
        self, tmp_path: Path
    ) -> None:
        card = self.score(tmp_path)
        assert card.unscored == ["doc_003", "doc_004"]  # no label
        assert card.perf.documents == 3 and card.perf.cost_usd == Decimal("0.60")
        assert (card.perf.latency_p50_ms, card.perf.latency_p95_ms) == (3000.0, 5000.0)
        assert (card.perf.input_tokens, card.perf.output_tokens) == (300, 150)

    def test_perf_by_slice_uses_the_manifest_columns(self, tmp_path: Path) -> None:
        slices = self.score(tmp_path).perf_slices
        assert slices["split"]["dev"].latency_p50_ms == 1000.0
        assert slices["split"]["test"].cost_usd == Decimal("0.20")
        assert slices["doc_type"]["faktur"].documents == 1
        # doc_003 is not in the manifest: it lands in the "(none)" group, not in an error
        assert slices["split"]["(none)"].cost_usd == Decimal("0.30")
        assert sum(stats.documents for stats in slices["split"].values()) == 3

    def test_the_json_carries_the_numbers(self, tmp_path: Path) -> None:
        data = json.loads(self.score(tmp_path).to_json())
        perf = data["perf"]
        assert perf["documents"] == 3 and perf["cost_usd"] == "0.60"
        assert perf["cost_per_doc"] == "0.200000" and perf["list_cost_per_doc"] == "0.200000"
        assert perf["latency_ms"] == {"p50": 3000.0, "p95": 5000.0, "mean": 3000.0}
        assert data["perf_slices"]["split"]["dev"]["latency_ms"]["p50"] == 1000.0

    def test_the_markdown_has_a_cost_and_latency_section(self, tmp_path: Path) -> None:
        markdown = self.score(tmp_path).to_markdown()
        assert "## Cost and latency" in markdown
        assert (
            "| all | 3 | 3 | 0 | $0.6000 | $0.6000 | $0.2000 | 300/150 | 3.0 s | 5.0 s |"
            in markdown
        )
        assert (
            "| split=dev | 1 | 1 | 0 | $0.1000 | $0.1000 | $0.1000 | 100/50 | 1.0 s | 1.0 s |"
            in (markdown)
        )

    def test_a_live_call_with_zero_latency_is_a_real_zero_not_n_a(self, tmp_path: Path) -> None:
        folder, manifest, labels_dir, pages_dir = two_document_fixture(tmp_path)
        card = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert card.perf.live_calls == 2 and card.perf.latency_p50_ms == 0.0

    def test_a_run_answered_from_the_cache_has_no_latency_but_still_a_list_price(
        self, tmp_path: Path
    ) -> None:
        folder, manifest, labels_dir, pages_dir = two_document_fixture(tmp_path)
        for doc_id in ("doc_001", "doc_002"):
            path = folder / "docs" / f"{doc_id}.json"
            record = DocRecord.model_validate_json(path.read_text(encoding="utf-8"))
            cached = record.model_copy(
                update={"cache_hit": True, "avoided_cost_usd": Decimal("0.05")}
            )
            path.write_text(cached.model_dump_json(indent=2) + "\n", encoding="utf-8")
        card = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert card.perf.live_calls == 0 and card.perf.cached_documents == 2
        assert card.perf.latency_p50_ms is None and card.perf.cost_usd == Decimal(0)
        assert card.perf.list_cost_per_doc == Decimal("0.05")
        assert "| all | 2 | 0 | 2 | $0.0000 | $0.1000 | $0.0500 | 0/0 | n/a | n/a |" in (
            card.to_markdown()
        )

    def test_the_output_is_byte_identical_twice(self, tmp_path: Path) -> None:
        first = self.score(tmp_path)
        second = score_run(
            tmp_path / "outputs" / "runs" / "perf-run",
            manifest_path=tmp_path / "data" / "manifest.csv",
            labels_dir=tmp_path / "ground_truth",
            pages_dir=tmp_path / "data" / "pages",
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert first.to_json() == second.to_json() and first.to_markdown() == second.to_markdown()


# ------------------------------------------------------------------ before / after


def metric(before: object, after: object) -> dict[str, object]:
    delta = None if before is None or after is None else after - before  # type: ignore[operator]
    return {"before": before, "after": after, "delta": delta}


def before_after_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "baseline": "base",
        "candidate": "cand",
        "documents_paired": 10,
        "fields_paired": 130,
        "metrics": {
            "accuracy": metric(0.9, 0.85),
            "selective_accuracy": metric(0.95, 0.95),
            "coverage": metric(0.9, 0.85),
            "row_f1": metric(0.8, 0.8),
            "list_cost_per_doc_usd": metric(0.1, 0.05),
            "latency_p50_ms": metric(100_000.0, 50_000.0),
            "latency_p95_ms": metric(150_000.0, 60_000.0),
            "input_tokens_per_doc": metric(3000.0, 1500.0),
            "output_tokens_per_doc": metric(8000.0, 4000.0),
        },
        "mcnemar": {"regressions": 6, "improvements": 1, "p_worse": 0.0625},
        "bootstrap": {"delta": -0.05, "low": -0.09, "high": -0.01, "documents": 10},
        "verdict": "ok",
        "differs": [{"key": "settings.reasoning_effort", "baseline": None, "candidate": "low"}],
    }
    data.update(overrides)
    return data


class TestBeforeAfter:
    def card(self, tmp_path: Path, **extra: object):  # noqa: ANN202
        folder, manifest, labels_dir, pages_dir = two_document_fixture(tmp_path)
        card = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        return dataclasses.replace(card, **extra)

    def test_without_a_baseline_there_is_no_section_and_no_key(self, tmp_path: Path) -> None:
        card = self.card(tmp_path)
        assert "Before / after" not in card.to_markdown()
        assert "before_after" not in json.loads(card.to_json())

    def test_the_section_shows_before_after_and_the_change(self, tmp_path: Path) -> None:
        text = self.card(tmp_path, before_after=before_after_data()).to_markdown()
        assert "## Before / after" in text
        assert "Baseline `base` against this run `cand`: 10 paired document(s), 130 labeled" in text
        for line in (
            "| accuracy | 0.900 | 0.850 | -5.0 pts |",
            "| selective accuracy | 0.950 | 0.950 | +0.0 pts |",
            "| coverage | 0.900 | 0.850 | -5.0 pts |",
            "| cost per doc (list price) | $0.1000 | $0.0500 | -50% |",
            "| latency p50 | 100.0 s | 50.0 s | -50% |",
            "| latency p95 | 150.0 s | 60.0 s | -60% |",
            "| input tokens per doc | 3000 | 1500 | -50% |",
            "| output tokens per doc | 8000 | 4000 | -50% |",
            "- McNemar (exact): 6 regression(s), 1 improvement(s), p_worse 0.0625; "
            "regression verdict: ok",
            "- paired bootstrap over 10 document(s): accuracy change -5.0 pts, "
            "interval [-9.0, -1.0] pts",
            "  - settings.reasoning_effort: None -> low",
        ):
            assert line in text, line

    def test_the_section_sits_after_cost_and_latency_and_before_the_unscored_line(
        self, tmp_path: Path
    ) -> None:
        text = self.card(tmp_path, before_after=before_after_data()).to_markdown()
        assert (
            text.index("## Cost and latency")
            < text.index("## Before / after")
            < text.index("Unscored documents")
        )

    def test_unknown_numbers_read_n_a_and_an_equal_run_has_no_differences(
        self, tmp_path: Path
    ) -> None:
        metrics = before_after_data()["metrics"]
        assert isinstance(metrics, dict)
        metrics["list_cost_per_doc_usd"] = metric(None, None)
        metrics["latency_p50_ms"] = metric(None, None)
        text = self.card(
            tmp_path, before_after=before_after_data(metrics=metrics, differs=[])
        ).to_markdown()
        assert "| cost per doc (list price) | n/a | n/a | n/a |" in text
        assert "| latency p50 | n/a | n/a | n/a |" in text
        assert "what differs" not in text

    def test_a_before_cost_of_zero_has_no_percentage(self, tmp_path: Path) -> None:
        metrics = before_after_data()["metrics"]
        assert isinstance(metrics, dict)
        metrics["list_cost_per_doc_usd"] = metric(0.0, 0.05)
        text = self.card(tmp_path, before_after=before_after_data(metrics=metrics)).to_markdown()
        assert "| cost per doc (list price) | $0.0000 | $0.0500 | n/a |" in text

    def test_the_json_carries_the_data_as_given(self, tmp_path: Path) -> None:
        data = before_after_data()
        payload = json.loads(self.card(tmp_path, before_after=data).to_json())
        assert payload["before_after"] == data

    def test_row_accuracy_is_n_a_when_no_row_matched_and_a_number_when_some_did(
        self, tmp_path: Path
    ) -> None:
        # the two-document fixture labels no rows: nothing matched, so no "perfect 1.000"
        text = self.card(tmp_path).to_markdown()
        assert "row accuracy n/a, cell accuracy n/a" in text

        tea = GTLineItem(
            description="Tea", quantity=Decimal(1), unit_price=Decimal(5), amount=Decimal(5)
        )
        rows = [
            ManifestRow(
                doc_id="doc_001",
                raw_files=("doc_001.pdf",),
                content_group="doc_001",
                doc_type="invoice",
                capture="born_digital",
                split="dev",
            )
        ]
        manifest, labels_dir, pages_dir = dataset(
            tmp_path / "w", rows, [label("doc_001", total=Decimal("5"), line_items=[tea])]
        )
        predicted = LineItem(
            description="Tea",
            quantity=Decimal(1),
            unit_price=Decimal(5),
            amount=Decimal(5),
            status="extracted",
            confidence=1.0,
        )
        folder = run_dir(
            tmp_path / "w",
            [
                DocRecord(
                    doc_id="doc_001",
                    status="ok",
                    validated=document(total=extracted(Decimal("5")), line_items=[predicted]),
                )
            ],
            name="rows-run",
        )
        card = score_run(
            folder,
            manifest_path=manifest,
            labels_dir=labels_dir,
            pages_dir=pages_dir,
            now=NOW,
            git_sha="a" * 40,
            git_dirty=False,
        )
        assert "row accuracy 1.000, cell accuracy 1.000" in card.to_markdown()
