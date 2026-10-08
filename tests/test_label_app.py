"""The labeling app, driven headlessly with Streamlit's AppTest against a temporary dataset.

The rules are tested in test_labeling.py; these tests prove the wiring: what is shown for a
pre-labeled and for a blind document, that every decision goes through a button, that a refresh
loses nothing, that saving is blocked while anything is pending, and that the label and its log
line are what end up on disk."""

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from docextract.ingest.manifest import ManifestRow, write_manifest
from docextract.ingest.pipeline import ingest_document
from docextract.paths import DatasetPaths
from evalkit.groundtruth import SCALAR_FIELD_NAMES, UNLABELED, GroundTruth
from evalkit.gt_store import label_path, read_label, write_label
from evalkit.labeling_ui import wip_dir
from ingest_support import settings

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.slow

APP = Path(__file__).resolve().parents[1] / "tools" / "label_app.py"
DRAFT: dict[str, Any] = {
    "doc_id": "doc_001",
    "label_mode": "prelabel",
    "notes": "pre-label draft by test-model; unverified",
    "vendor_name": "PT Contoh Makmur",
    "vendor_address": "Jl. Merdeka 12, Bandung",
    "vendor_tax_id": "01.234.567.8-901.000",
    "buyer_name": "CV Pembeli",
    "document_number": "INV/2024/0457",
    "document_date": "2024-03-05",
    "due_date": None,
    "currency": "IDR",
    "payment_method": UNLABELED,
    "subtotal": "450000.00",
    "tax": "49500.00",
    "discount": None,
    "total": "499500.00",
    "line_items": [
        {
            "description": "Kertas A4",
            "quantity": "10",
            "unit_price": "45000.00",
            "amount": "450000.00",
        }
    ],
    "field_notes": {"payment_method": "the model did not answer this field"},
}


def row(doc_id: str, **overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "doc_id": doc_id,
        "raw_files": f"{doc_id}.png",
        "content_group": doc_id,
        "doc_type": "faktur",
        "split": "dev",
        "label_mode": "prelabel",
        "needs_redaction": "n",
    }
    values.update(overrides)
    return ManifestRow.model_validate(values)


class World:
    """A temporary dataset root with a manifest, ingested pages and (optionally) a draft."""

    def __init__(self, root: Path) -> None:
        self.paths = DatasetPaths(root)
        self.rows = [
            row("doc_001"),
            row("doc_002", label_mode="blind", split="test"),
            row("doc_003", gt_from="doc_001", label_mode=None, content_group="doc_001"),
            row("cons_001", capture="constructed", label_mode=None),
        ]
        write_manifest(self.paths.manifest, self.rows)
        for name in ("doc_001", "doc_002"):
            self.paths.raw_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (120, 90), (200, 200, 200)).save(self.paths.raw_dir / f"{name}.png")
            ingest_document(
                row(name),
                raw_dir=self.paths.raw_dir,
                out_dir=self.paths.pages_dir / name,
                settings=settings(),
            )
        write_label(self.paths.drafts_dir, GroundTruth.model_validate(DRAFT))

    def app(self) -> AppTest:
        at = AppTest.from_file(str(APP), default_timeout=60)
        return at.run()

    def label(self, doc_id: str = "doc_001") -> GroundTruth:
        return read_label(label_path(self.paths.ground_truth_dir, doc_id))

    def changelog(self) -> str:
        path = self.paths.changelog
        return path.read_text(encoding="utf-8") if path.is_file() else ""


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    monkeypatch.setenv("DOCEXTRACT_ROOT", str(tmp_path))
    return World(tmp_path)


def select(at: AppTest, doc_id: str) -> AppTest:
    at.sidebar.selectbox(key="doc").select(doc_id)
    return at.run()


def text(at: AppTest, doc: str, name: str) -> str:
    return str(at.text_input(key=f"{doc}:{name}:text").value)


def click(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click()
    return at.run()


def decide_like_the_draft(at: AppTest, doc: str = "doc_001") -> AppTest:
    """Verify every field the draft proposed a value for, accept its absent / unlabeled."""
    for name in SCALAR_FIELD_NAMES:
        proposal = DRAFT[name]
        action = (
            "absent" if proposal is None else "unlabeled" if proposal == UNLABELED else "verify"
        )
        at = click(at, f"{doc}:{name}:{action}")
    return click(at, f"{doc}:rows:verify")


def markdown(at: AppTest) -> str:
    return "\n".join(str(item.value) for item in at.markdown)


# ------------------------------------------------------------------ what is shown


class TestOpening:
    def test_the_app_starts_cleanly_and_lists_only_documents_a_person_labels(
        self, world: World
    ) -> None:
        at = world.app()
        assert not at.exception
        assert at.title[0].value == "Label documents"
        box = at.sidebar.selectbox(key="doc")
        assert list(box.options) == [
            "doc_001  ·  faktur  ·  dev  ·  prelabel  ·  draft",
            "doc_002  ·  faktur  ·  test  ·  blind  ·  to do",
        ]  # not the borrower (doc_003), not the constructed invoice

    def test_a_prelabeled_document_starts_from_the_draft_with_everything_pending(
        self, world: World
    ) -> None:
        at = world.app()
        assert "Pre-label draft loaded" in at.info[0].value
        assert text(at, "doc_001", "vendor_name") == "PT Contoh Makmur"
        assert text(at, "doc_001", "document_date") == "2024-03-05"
        assert text(at, "doc_001", "total") == "499500.00"
        assert text(at, "doc_001", "due_date") == ""  # the draft proposed "absent"
        assert markdown(at).count(":orange[pending]") == len(SCALAR_FIELD_NAMES) + 1
        assert at.button(key="save").disabled

    def test_the_drafts_proposals_and_remarks_are_shown_but_decide_nothing(
        self, world: World
    ) -> None:
        captions = "\n".join(str(c.value) for c in at_captions(world.app()))
        assert "draft says: absent" in captions
        assert "draft says: unlabeled (the model did not answer this field)" in captions
        assert "draft says: 1 row(s)" in captions

    def test_a_blind_document_starts_empty_and_says_so(self, world: World) -> None:
        at = select(world.app(), "doc_002")
        assert not at.exception
        assert "BLIND document" in at.warning[0].value
        assert all(text(at, "doc_002", name) == "" for name in SCALAR_FIELD_NAMES)
        assert not any("Pre-label draft" in str(i.value) for i in at.info)

    def test_a_blind_document_with_a_draft_is_refused_loudly(self, world: World) -> None:
        write_label(
            world.paths.drafts_dir, GroundTruth.model_validate({**DRAFT, "doc_id": "doc_002"})
        )
        at = select(world.app(), "doc_002")
        errors = " ".join(str(e.value) for e in at.error)
        assert "BLIND document but has a model draft" in errors
        keys = [b.key for b in at.button]
        assert "save" not in keys  # no form at all: nothing can be labeled or saved
        assert not [k for k in keys if k and k.startswith("doc_002:")]

    def test_merely_looking_at_a_document_saves_no_work(self, world: World) -> None:
        select(world.app(), "doc_002")
        assert not wip_dir(world.paths).exists() or not list(wip_dir(world.paths).iterdir())

    def test_a_missing_manifest_is_an_error_not_a_crash(self, world: World) -> None:
        world.paths.manifest.unlink()
        at = world.app()
        assert "The manifest cannot be read" in at.error[0].value

    def test_pages_that_were_not_ingested_are_reported_and_the_form_still_works(
        self, world: World
    ) -> None:
        (world.paths.pages_dir / "doc_001" / "ingest.json").unlink()
        at = world.app()
        assert any("not ingested yet" in str(w.value) for w in at.warning)
        assert text(at, "doc_001", "vendor_name") == "PT Contoh Makmur"


def at_captions(at: AppTest) -> list[Any]:
    return list(at.caption)


# ------------------------------------------------------------------ deciding


class TestDeciding:
    def test_verifying_normalizes_what_is_in_the_box(self, world: World) -> None:
        at = world.app()
        at.text_input(key="doc_001:total:text").set_value("Rp 499.500,00")
        at = click(at, "doc_001:total:verify")
        assert text(at, "doc_001", "total") == "499500.00"
        assert ":green[verified]" in markdown(at)

    def test_an_ambiguous_amount_shows_an_inline_error_and_stays_pending(
        self, world: World
    ) -> None:
        at = world.app()
        at.text_input(key="doc_001:currency:text").set_value("")
        at.text_input(key="doc_001:total:text").set_value("1.250")
        at = click(at, "doc_001:total:verify")
        assert any("ambiguous" in str(e.value) for e in at.error)
        assert text(at, "doc_001", "total") == "1.250"
        assert markdown(at).count(":orange[pending]") == len(SCALAR_FIELD_NAMES) + 1
        assert ":green[verified]" not in markdown(at)

    def test_absent_and_unlabeled_clear_the_box(self, world: World) -> None:
        at = click(world.app(), "doc_001:vendor_name:absent")
        assert text(at, "doc_001", "vendor_name") == ""
        assert ":gray[absent (null)]" in markdown(at)
        at = click(at, "doc_001:tax:unlabeled")
        assert text(at, "doc_001", "tax") == "" and ":blue[unlabeled]" in markdown(at)

    def test_progress_follows_the_decisions(self, world: World) -> None:
        at = world.app()
        assert any("0 of 14 fields decided" in str(p) for p in progress_texts(at))
        at = click(at, "doc_001:due_date:absent")
        assert any("1 of 14 fields decided" in str(p) for p in progress_texts(at))

    def test_a_note_is_kept(self, world: World) -> None:
        at = world.app()
        at.text_input(key="doc_001:total:note").set_value("total not printed")
        at = at.run()
        saved = json.loads((wip_dir(world.paths) / "doc_001.json").read_text(encoding="utf-8"))
        assert saved["fields"]["total"]["note"] == "total not printed"


def progress_texts(at: AppTest) -> list[str]:
    return [str(element.proto.text) for element in at.get("progress")]


# ------------------------------------------------------------------ saving


class TestSaving:
    def test_saving_is_blocked_while_anything_is_pending(self, world: World) -> None:
        at = click(world.app(), "doc_001:vendor_name:verify")
        assert at.button(key="save").disabled
        assert not label_path(world.paths.ground_truth_dir, "doc_001").exists()

    def test_a_fully_decided_draft_is_saved_with_its_mode_time_and_log_line(
        self, world: World
    ) -> None:
        at = decide_like_the_draft(world.app())
        assert not at.button(key="save").disabled
        at = click(at, "save")
        assert any("Saved" in str(s.value) for s in at.success)
        label = world.label()
        assert label.label_mode == "prelabel" and label.labeled_at is not None
        assert label.vendor_name == "PT Contoh Makmur" and label.total == 499500
        assert label.due_date is None and label.payment_method == UNLABELED
        assert label.notes is None  # the draft's own remark is not part of the label
        log = world.changelog()
        assert "doc_001.json | (new file) | - -> label_mode=prelabel | labeled by hand" in log

    def test_saved_work_is_cleared_once_the_label_is_saved(self, world: World) -> None:
        at = decide_like_the_draft(world.app())
        assert (wip_dir(world.paths) / "doc_001.json").exists()
        click(at, "save")
        assert not (wip_dir(world.paths) / "doc_001.json").exists()

    def test_a_blind_document_is_labeled_from_scratch_and_saved_as_blind(
        self, world: World
    ) -> None:
        at = select(world.app(), "doc_002")
        answers = {
            "vendor_name": "PT Contoh", "vendor_address": "Jl. A", "vendor_tax_id": "01.234",
            "buyer_name": "CV B", "document_number": "N-1", "document_date": "5 Maret 2024",
            "currency": "Rp", "subtotal": "100.000", "tax": "11.000", "total": "111.000",
        }  # fmt: skip
        for name in SCALAR_FIELD_NAMES:
            if name in answers:
                at.text_input(key=f"doc_002:{name}:text").set_value(answers[name])
                at = click(at, f"doc_002:{name}:verify")
            else:
                at = click(at, f"doc_002:{name}:absent")
        at = click(at, "doc_002:rows:absent")
        assert not at.button(key="save").disabled
        click(at, "save")
        label = world.label("doc_002")
        assert label.label_mode == "blind" and label.line_items is None
        assert str(label.document_date) == "2024-03-05" and label.total == 111000
        assert "doc_002.json | (new file) | - -> label_mode=blind" in world.changelog()

    def test_replacing_a_finished_label_asks_for_a_reason_and_logs_it(self, world: World) -> None:
        at = click(decide_like_the_draft(world.app()), "save")
        before = (world.paths.ground_truth_dir / "doc_001.json").read_bytes()
        at.text_input(key="doc_001:vendor_name:text").set_value("PT Baru")
        at = click(at, "doc_001:vendor_name:verify")
        at = click(at, "save")
        assert any("say why" in str(e.value) for e in at.error)
        assert (world.paths.ground_truth_dir / "doc_001.json").read_bytes() == before
        at.text_input(key="doc_001:reason").set_value("misread the letterhead")
        at = click(at, "save")
        assert world.label().vendor_name == "PT Baru"
        assert (
            "vendor_name | PT Contoh Makmur -> PT Baru | misread the letterhead"
            in world.changelog()
        )

    def test_saving_the_same_label_again_changes_nothing(self, world: World) -> None:
        at = click(decide_like_the_draft(world.app()), "save")
        before, log = (
            (world.paths.ground_truth_dir / "doc_001.json").read_bytes(),
            world.changelog(),
        )
        at = click(at, "save")
        assert any("Nothing to save" in str(s.value) for s in at.success)
        assert (world.paths.ground_truth_dir / "doc_001.json").read_bytes() == before
        assert world.changelog() == log


# ------------------------------------------------------------------ keeping work


class TestKeepingWork:
    def test_a_refresh_loses_nothing(self, world: World) -> None:
        at = world.app()
        at.text_input(key="doc_001:vendor_name:text").set_value("Edited by hand")
        at = click(at, "doc_001:vendor_name:verify")
        at = click(at, "doc_001:due_date:absent")
        again = world.app()  # a new session: nothing in memory
        assert text(again, "doc_001", "vendor_name") == "Edited by hand"
        assert again.markdown and ":green[verified]" in markdown(again)
        assert ":gray[absent (null)]" in markdown(again)
        assert any("saved work" in str(c.value) for c in again.caption)

    def test_saved_work_wins_over_the_draft_and_the_overview_says_so(self, world: World) -> None:
        click(world.app(), "doc_001:due_date:absent")
        labels = list(world.app().sidebar.selectbox(key="doc").options)
        assert labels[0].endswith("in progress")

    def test_discard_goes_back_to_the_draft(self, world: World) -> None:
        at = world.app()
        at.text_input(key="doc_001:vendor_name:text").set_value("Edited")
        at = click(at, "doc_001:vendor_name:verify")
        at = click(at, "discard")
        assert text(at, "doc_001", "vendor_name") == "PT Contoh Makmur"
        assert not (wip_dir(world.paths) / "doc_001.json").exists()
        assert ":green[verified]" not in markdown(at)

    def test_unusable_saved_work_is_ignored_with_a_warning(self, world: World) -> None:
        wip_dir(world.paths).mkdir(parents=True)
        (wip_dir(world.paths) / "doc_001.json").write_text("{not json", encoding="utf-8")
        at = world.app()
        assert any("saved work could not be used" in str(w.value) for w in at.warning)
        assert text(at, "doc_001", "vendor_name") == "PT Contoh Makmur"  # fell back to the draft

    def test_a_finished_label_reopens_fully_decided(self, world: World) -> None:
        click(decide_like_the_draft(world.app()), "save")
        again = world.app()
        assert any("finished label" in str(c.value) for c in again.caption)
        assert not again.button(key="save").disabled
        assert ":orange[pending]" not in markdown(again)

    def test_integrity_problems_are_shown_in_the_sidebar(self, world: World) -> None:
        write_label(world.paths.ground_truth_dir, GroundTruth.model_validate({"doc_id": "doc_999"}))
        at = world.app()
        assert any("no document 'doc_999'" in str(e.value) for e in at.sidebar.error)


def form_rows(at: AppTest, doc: str) -> list[dict[str, str]]:
    return list(at.session_state["forms"][doc].form.rows)


def paste(at: AppTest, doc: str, text: str, how: str = "replace") -> AppTest:
    at.text_area(key=f"{doc}:rows_paste").set_value(text)
    return click(at, f"{doc}:rows:paste_{how}")


class TestPasteRows:
    def test_pasted_cells_fill_the_form_and_the_grid_and_the_box_empties(
        self, world: World
    ) -> None:
        at = select(world.app(), "doc_002")
        at = paste(at, "doc_002", "Tea\t2\t12500\t25000\nCake\t1\t3000\t3000")
        assert [r["description"] for r in form_rows(at, "doc_002")] == ["Tea", "Cake"]
        assert at.text_area(key="doc_002:rows_paste").value == ""
        assert at.session_state["doc_002:rows_version"] == 1
        saved = (wip_dir(world.paths) / "doc_002.json").read_text(encoding="utf-8")
        assert "Tea" in saved and "Cake" in saved  # work in progress is written

    def test_the_pasted_table_verifies_like_a_typed_one(self, world: World) -> None:
        at = select(world.app(), "doc_002")
        at = paste(at, "doc_002", "Tea\t2\t12500\t25000")
        at = click(at, "doc_002:rows:verify")
        assert at.session_state["forms"]["doc_002"].form.rows_status == "verified"

    def test_add_after_the_last_row_keeps_what_is_there(self, world: World) -> None:
        at = select(world.app(), "doc_002")
        at = paste(at, "doc_002", "Tea\t1\t5\t5")
        at = paste(at, "doc_002", "Cake\t1\t3\t3", how="add")
        assert [r["description"] for r in form_rows(at, "doc_002")] == ["Tea", "Cake"]
        assert at.session_state["doc_002:rows_version"] == 2

    def test_replace_discards_the_rows_that_were_there(self, world: World) -> None:
        at = select(world.app(), "doc_002")
        at = paste(at, "doc_002", "Tea\t1\t5\t5\nCake\t1\t3\t3")
        at = paste(at, "doc_002", "Bun\t1\t2\t2", how="replace")
        assert [r["description"] for r in form_rows(at, "doc_002")] == ["Bun"]

    def test_a_bad_paste_changes_nothing_and_says_why(self, world: World) -> None:
        at = select(world.app(), "doc_002")
        at = paste(at, "doc_002", "Tea\t1\t5\t5")
        at = paste(at, "doc_002", "just typed words", how="add")
        assert [r["description"] for r in form_rows(at, "doc_002")] == ["Tea"]
        assert any("no tab found" in str(e.value) for e in at.error)
        assert at.text_area(key="doc_002:rows_paste").value == "just typed words"  # kept to fix
        assert at.session_state["doc_002:rows_version"] == 1

    def test_a_good_paste_after_a_bad_one_clears_the_message(self, world: World) -> None:
        at = select(world.app(), "doc_002")
        at = paste(at, "doc_002", "words")
        assert any("no tab found" in str(e.value) for e in at.error)
        at = paste(at, "doc_002", "Tea\t1\t5\t5")
        assert not any("no tab found" in str(e.value) for e in at.error)

    def test_a_blind_document_stays_blind_after_a_paste(self, world: World) -> None:
        at = select(world.app(), "doc_002")
        at = paste(at, "doc_002", "Tea\t1\t5\t5")
        opened = at.session_state["forms"]["doc_002"]
        assert opened.form.label_mode == "blind" and opened.source == "empty"

    def test_saved_work_from_before_this_change_still_loads(self, world: World) -> None:
        from evalkit.labeling import LabelForm
        from evalkit.labeling_ui import save_wip

        form = LabelForm.start(world.rows[1])
        form.set_rows([{"description": "Saved earlier", "quantity": "1"}])
        save_wip(form, world.paths)
        at = select(world.app(), "doc_002")
        assert [r["description"] for r in form_rows(at, "doc_002")] == ["Saved earlier"]
        assert at.session_state["forms"]["doc_002"].source == "saved work"

    def test_the_grid_is_given_the_same_input_on_every_rerun_until_the_version_moves(
        self, world: World
    ) -> None:
        at = select(world.app(), "doc_002")
        at = paste(at, "doc_002", "Tea\t1\t5\t5")
        held = at.session_state["doc_002:rows_source"]
        assert held[0] == 1 and list(held[1]["description"]) == ["Tea"]
        # the grid's own output is written back into the form on every run; that must not change
        # what the grid is given, or the next edit is applied to a different table
        at.session_state["forms"]["doc_002"].form.set_rows([{"description": "Tea (edited)"}])
        at = at.run()
        again = at.session_state["doc_002:rows_source"]
        assert again[0] == 1 and list(again[1]["description"]) == ["Tea"]
        # the version moving is the only way to show something new: the grid output (not the
        # simulated rewrite above) is what the form holds, and the new input is built from it
        at = paste(at, "doc_002", "Cake\t1\t3\t3", how="add")
        moved = at.session_state["doc_002:rows_source"]
        assert moved[0] == 2 and list(moved[1]["description"]) == ["Tea", "Cake"]

    def test_discarding_unsaved_work_also_forgets_the_frozen_grid_input(self, world: World) -> None:
        at = select(world.app(), "doc_002")
        at = paste(at, "doc_002", "Tea\t1\t5\t5")
        assert "doc_002:rows_source" in at.session_state
        at = click(at, "discard")
        # rebuilt from the reloaded (empty) form at version 0, not the pasted table
        reloaded = at.session_state["doc_002:rows_source"]
        assert reloaded[0] == 0 and len(reloaded[1]) == 0
