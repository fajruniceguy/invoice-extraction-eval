import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from docextract.extract.runner import DocRecord, RunRecord, RunTotals
from docextract.extract.schema import SCALAR_FIELD_NAMES, DocumentExtraction, Field
from docextract.ingest.manifest import ManifestRow, write_manifest
from evalkit.groundtruth import GroundTruth
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
