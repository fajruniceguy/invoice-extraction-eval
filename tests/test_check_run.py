import importlib.util
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from docextract.config import Settings
from docextract.extract.runner import DocRecord, run_extraction
from docextract.extract.schema import DocumentExtraction, Field, LineItem
from docextract.ingest.manifest import ManifestRow
from docextract.providers.openai_compat import OpenAICompatClient
from docextract.validate import validate_document
from docextract.validate.codes import CONFIDENCE_BELOW_THRESHOLD, DESCRIPTION_NOT_VERBATIM
from fakes.fake_llm import FakeLLMServer, canned_answer, chat_reply
from ingest_support import Env, gray_png

ROOT = Path(__file__).resolve().parents[1]


def load_check_run() -> Any:
    """scripts/check_run.py is a script, not an installed module: load it by path."""
    spec = importlib.util.spec_from_file_location("check_run", ROOT / "scripts" / "check_run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_run = load_check_run()


def scalar(value: Any, evidence: str = "quote", **extra: Any) -> Field[Any]:
    return Field.model_validate(
        {
            "value": value,
            "status": "extracted",
            "confidence": 0.9,
            "evidence": evidence,
            "page": 1,
            **extra,
        }
    )


def row() -> LineItem:
    return LineItem.model_validate(
        {
            "description": "Widget",
            "quantity": "2",
            "unit_price": "4.50",
            "amount": "9.00",
            "status": "extracted",
            "confidence": 0.9,
            "evidence": "2 x 4.50 = 9.00 Widget",
            "page": 1,
        }
    )


def document() -> DocumentExtraction:
    """A clean, grounded, consistent extraction."""
    return DocumentExtraction(
        vendor_name=scalar("PT Contoh", "PT Contoh"),
        vendor_address=Field.absent(0.9),
        vendor_tax_id=Field.absent(0.9),
        buyer_name=Field.absent(0.9),
        document_number=scalar("INV-1", "INV-1"),
        document_date=scalar(date(2024, 3, 5), "5 Maret 2024"),
        due_date=Field.absent(0.9),
        currency=scalar("IDR", "Rp"),
        payment_method=Field.absent(0.9),
        line_items=[row()],
        subtotal=scalar(Decimal("9.00"), "9.00"),
        tax=scalar(Decimal("0.99"), "0.99"),
        discount=Field.absent(0.9),
        total=scalar(Decimal("9.99"), "9.99"),
    )


def record(
    document_value: DocumentExtraction | None = None,
    parsed_value: DocumentExtraction | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """One record as the runner writes it, with optional JSON-level damage."""
    record = DocRecord(
        doc_id="doc_a",
        status="ok",
        pages=[{"file": "p1.png", "sha256": "0" * 64}],
        parsed=parsed_value if parsed_value is not None else document(),
        validated=document_value if document_value is not None else document(),
    )
    data: dict[str, Any] = json.loads(record.model_dump_json())
    for key, value in overrides.items():
        data[key] = value
    return data


def run_dir_with(tmp_path: Path, data: dict[str, Any], *, doc_id: str = "doc_a") -> Path:
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "docs").mkdir(parents=True)
    (run_dir / "docs" / f"{doc_id}.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir


def problems_of(tmp_path: Path, data: dict[str, Any]) -> list[str]:
    report = check_run.check_run_dir(run_dir_with(tmp_path, data))
    return report.problems


class TestACleanRecord:
    def test_a_clean_record_passes(self, tmp_path: Path) -> None:
        run_dir = run_dir_with(tmp_path, record())
        report = check_run.check_run_dir(run_dir)
        assert report.ok, report.problems
        assert report.counts.fields == 13
        assert report.counts.rows == 1

    def test_the_report_names_the_definition_of_done(self, tmp_path: Path) -> None:
        text = check_run.format_report(check_run.check_run_dir(run_dir_with(tmp_path, record())))
        assert "all fields carry value/confidence/evidence/status/reasons" in text
        assert "13 field(s), 1 row(s), 0 abstained, 0 flagged cell(s)" in text

    def test_an_abstained_field_and_a_flagged_row_are_counted(self, tmp_path: Path) -> None:
        damaged = document()
        damaged.vendor_name = Field.abstain(CONFIDENCE_BELOW_THRESHOLD, confidence=0.1)
        damaged.line_items = [
            row().model_copy(
                update={
                    "confidence": 0.5,
                    "flagged_cells": ["amount"],
                    "reasons": [DESCRIPTION_NOT_VERBATIM],
                }
            )
        ]
        report = check_run.check_run_dir(run_dir_with(tmp_path, record(damaged)))
        assert report.ok, report.problems
        assert report.counts.abstained == 1
        assert report.counts.flagged == 1

    def test_records_that_are_not_ok_are_skipped(self, tmp_path: Path) -> None:
        run_dir = run_dir_with(tmp_path, record(status="failed", validated=None, parsed=None))
        report = check_run.check_run_dir(run_dir)
        assert report.ok and report.counts.documents == 0
        assert report.not_ok == ["doc_a"]


class TestWhatItRejects:
    def test_an_extracted_field_without_evidence(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["total"]["evidence"] = None
        assert problems_of(tmp_path, data) == ["doc_a total: extracted without evidence"]

    def test_a_page_outside_the_document(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["tax"]["page"] = 2  # the record has one page
        assert problems_of(tmp_path, data) == ["doc_a tax: extracted without a page in 1..1"]

    def test_an_absent_field_with_a_value(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["discount"]["value"] = "9.00"
        assert problems_of(tmp_path, data) == ["doc_a discount: absent with a non-null value"]

    def test_an_abstained_field_without_a_reason(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["vendor_name"] = {
            "value": None,
            "confidence": 0.1,
            "status": "abstained",
            "evidence": None,
            "page": None,
            "reasons": [],
        }
        assert "abstained without a reason" in problems_of(tmp_path, data)[0]

    def test_an_unknown_reason_code_on_an_extracted_field(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["total"]["reasons"] = ["made_up_code"]
        assert "unknown validator reason code(s) ['made_up_code']" in problems_of(tmp_path, data)[0]

    def test_prose_a_validator_added_is_flagged_too(self, tmp_path: Path) -> None:
        # the model did not give this reason, so it is a validator addition and must be a code
        data = record()
        data["validated"]["total"]["reasons"] = ["the total looks wrong"]
        assert "unknown validator reason code(s)" in problems_of(tmp_path, data)[0]

    def test_the_model_own_prose_reasons_are_allowed(self, tmp_path: Path) -> None:
        # the model abstained this field with its own words (in parsed and validated alike)
        parsed = document()
        parsed.vendor_name = Field.abstain("the photo is blurry", confidence=0.2)
        validated = document()
        validated.vendor_name = Field.abstain("the photo is blurry", confidence=0.2)
        assert problems_of(tmp_path, record(validated, parsed_value=parsed)) == []

    def test_the_model_prose_reasons_on_an_absent_field_are_allowed(self, tmp_path: Path) -> None:
        parsed = document()
        parsed.discount = Field.absent(0.9, "not stated on the document")
        validated = document()
        validated.discount = Field.absent(0.9, "not stated on the document")
        assert problems_of(tmp_path, record(validated, parsed_value=parsed)) == []

    def test_the_vocabulary_is_not_checked_without_a_parsed_output(self, tmp_path: Path) -> None:
        # nothing to compare with, so a prose reason cannot be told from a missing code
        validated = document()
        validated.vendor_name = Field.absent(0.9, "not stated on the document")
        assert problems_of(tmp_path, record(validated, parsed=None)) == []

    def test_a_reason_prose_on_an_abstained_field(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["vendor_name"] = {
            "value": None,
            "confidence": 0.2,
            "status": "abstained",
            "evidence": "It is blurry",
            "page": 1,
            "reasons": ["the photo is blurry"],
        }
        data["parsed"]["vendor_name"] = {
            "value": None,
            "confidence": 0.2,
            "status": "abstained",
            "evidence": "It is blurry",
            "page": 1,
            "reasons": ["the photo is blurry"],
        }
        assert problems_of(tmp_path, data) == []

    def test_a_confidence_outside_the_range(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["vendor_name"]["confidence"] = 1.5
        assert "is not in [0, 1]" in problems_of(tmp_path, data)[0]

    def test_an_unknown_status(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["vendor_name"]["status"] = "maybe"
        assert "is not one of" in problems_of(tmp_path, data)[0]

    def test_an_extracted_row_with_every_cell_null(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["line_items"][0].update(
            {"description": None, "quantity": None, "unit_price": None, "amount": None}
        )
        assert "extracted with every cell null" in problems_of(tmp_path, data)[0]

    def test_an_abstained_row_with_a_non_null_cell(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["line_items"][0].update({"status": "abstained", "amount": "9.00"})
        assert "abstained with a non-null cell" in problems_of(tmp_path, data)[0]

    def test_a_flagged_cell_that_is_not_a_cell(self, tmp_path: Path) -> None:
        data = record()
        data["validated"]["line_items"][0]["flagged_cells"] = ["vendor_name"]
        assert "are not cell names" in problems_of(tmp_path, data)[0]

    def test_a_missing_validated_output(self, tmp_path: Path) -> None:
        assert problems_of(tmp_path, record(validated=None)) == [
            "doc_a: an ok record without a validated output"
        ]

    def test_a_missing_field(self, tmp_path: Path) -> None:
        data = record()
        del data["validated"]["total"]
        assert problems_of(tmp_path, data) == ["doc_a total: missing"]

    def test_a_run_without_records(self, tmp_path: Path) -> None:
        (tmp_path / "runs" / "r1" / "docs").mkdir(parents=True)
        report = check_run.check_run_dir(tmp_path / "runs" / "r1")
        assert not report.ok and "no document records" in report.problems[0]

    def test_a_directory_that_is_not_a_run(self, tmp_path: Path) -> None:
        report = check_run.check_run_dir(tmp_path)
        assert not report.ok and "no docs/ directory" in report.problems[0]


class TestTheCommand:
    def test_main_exits_zero_and_prints_the_definition_of_done(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = run_dir_with(tmp_path, record())
        assert check_run.main([str(run_dir)]) == 0
        out = capsys.readouterr().out
        assert "all fields carry value/confidence/evidence/status/reasons" in out

    def test_main_exits_one_on_a_problem(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = record()
        data["validated"]["total"]["evidence"] = None
        assert check_run.main([str(run_dir_with(tmp_path, data))]) == 1
        assert "FAIL doc_a total: extracted without evidence" in capsys.readouterr().out

    def test_main_exits_two_for_a_missing_directory(self, tmp_path: Path) -> None:
        assert check_run.main([str(tmp_path / "nope")]) == 2

    def test_main_checks_every_directory_given(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        clean = run_dir_with(tmp_path, record())
        damaged = tmp_path / "runs" / "r2"
        (damaged / "docs").mkdir(parents=True)
        data = record()
        data["validated"]["total"]["evidence"] = None
        (damaged / "docs" / "doc_a.json").write_text(json.dumps(data), encoding="utf-8")
        assert check_run.main([str(clean), str(damaged)]) == 1
        assert capsys.readouterr().out.count("run r") == 2


class TestOverARealRunFolder:
    """check_run over what the runner writes for a fake model (the wiring the live run uses)."""

    @pytest.fixture
    async def run_dir(self, env: Env, fake_llm: FakeLLMServer, tmp_path: Path) -> Path:
        gray_png(env, "doc_a.png")
        row = ManifestRow.model_validate(
            {
                "doc_id": "doc_a",
                "raw_files": "doc_a.png",
                "content_group": "doc_a",
                "doc_type": "invoice",
                "capture": "born_digital",
                "split": "dev",
                "needs_redaction": "n",
            }
        )
        env.ingest(row)
        labels = tmp_path / "ground_truth"
        labels.mkdir()
        fake_llm.enqueue(chat_reply(canned_answer()))
        settings = Settings(
            _env_file=None,
            llm_base_url=fake_llm.base_url,
            llm_model="fake-extractor",
            llm_api_key="sk-test-key",
            llm_max_retries=0,
        )
        async with OpenAICompatClient(settings) as client:
            await run_extraction(
                [row],
                rows=[row],
                client=client,
                settings=settings,
                pages_dir=env.pages,
                labels_dir=labels,
                run_dir=tmp_path / "runs" / "r1",
                selection={"docs": ["doc_a"]},
            )
        return tmp_path / "runs" / "r1"

    async def test_the_validated_output_of_a_real_run_passes(self, run_dir: Path) -> None:
        report = check_run.check_run_dir(run_dir)
        assert report.ok, report.problems
        assert report.counts.documents == 1
        assert report.counts.fields == 13 and report.counts.rows == 3

    async def test_it_passes_the_chain_the_runner_applied(self, run_dir: Path) -> None:
        """The record's validated output really is the post-validation copy (a shape-capped
        tax id from the fake's canned answer), and the check agrees with validate_document."""
        data = json.loads((run_dir / "docs" / "doc_a.json").read_text(encoding="utf-8"))
        assert data["validated"]["vendor_tax_id"]["confidence"] == 0.7
        assert data["parsed"]["vendor_tax_id"]["confidence"] == 0.95
        document_value = DocumentExtraction.model_validate(data["validated"])
        assert document_value == validate_document(
            DocumentExtraction.model_validate(data["parsed"]),
            abstain_below=0.3,
            normalizations=data["normalizations"],
            today=date.today(),
        )
