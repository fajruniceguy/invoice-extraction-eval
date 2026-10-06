import json
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from docextract.ingest.manifest import ManifestRow
from evalkit.groundtruth import SCALAR_FIELD_NAMES, UNLABELED, GroundTruth
from evalkit.gt_store import dumps_label, label_path, parse_label, read_label, write_label
from evalkit.labeling import (
    FIELD_KINDS,
    ROW_CELLS,
    BlindDraftError,
    FieldState,
    LabelForm,
    LabelingError,
    changelog_lines,
    labelable_mode,
    save_label,
)

NOW = datetime(2026, 10, 6, 9, 15, tzinfo=UTC)
TODAY = date(2026, 10, 6)


def row(doc_id: str = "doc_001", **overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "doc_id": doc_id,
        "raw_files": f"{doc_id}.png",
        "content_group": doc_id,
        "label_mode": "prelabel",
        "needs_redaction": "n",
    }
    values.update(overrides)
    return ManifestRow.model_validate(values)


def draft(**changes: Any) -> GroundTruth:
    values: dict[str, Any] = {
        "doc_id": "doc_001",
        "label_mode": "prelabel",
        "notes": "pre-label draft by m; unverified",
        "vendor_name": "PT Contoh",
        "vendor_address": "Jl. Merdeka 12",
        "vendor_tax_id": "01.234.567.8-901.000",
        "buyer_name": "CV Pembeli",
        "document_number": "INV/1",
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
                "description": "Kertas",
                "quantity": "10",
                "unit_price": "45000.00",
                "amount": "450000.00",
            }
        ],
        "field_notes": {"payment_method": "the model did not answer this field"},
    }
    values.update(changes)
    return GroundTruth.model_validate(values)


ANSWERS: dict[str, str] = {
    "vendor_name": " PT Contoh Makmur ",
    "vendor_address": "Jl. Merdeka 12, Bandung",
    "vendor_tax_id": "01.234.567.8-901.000",
    "buyer_name": "CV Pembeli",
    "document_number": "INV/2024/0457",
    "document_date": "5 Maret 2024",
    "due_date": "",
    "currency": "Rp",
    "payment_method": "",
    "subtotal": "Rp 450.000,00",
    "tax": "49.500,00",
    "discount": "",
    "total": "499.500,00",
}


def complete(form: LabelForm) -> LabelForm:
    """Decide every field: answers where there is one, absent where there is none."""
    for name in SCALAR_FIELD_NAMES:
        if ANSWERS[name]:
            form.set_text(name, ANSWERS[name])
            assert form.verify(name), (name, form.fields[name].error)
        else:
            form.mark_absent(name)
    form.set_rows(
        [{"description": "Kertas", "quantity": "10", "unit_price": "45.000,00", "amount": ""}]
    )
    assert form.verify_rows(), form.rows_errors
    return form


# ------------------------------------------------------------------ starting a form


class TestStart:
    def test_every_field_starts_pending(self) -> None:
        form = LabelForm.start(row())
        assert form.pending() == [*SCALAR_FIELD_NAMES, "line_items"]
        assert form.progress() == (0, len(SCALAR_FIELD_NAMES) + 1)
        assert not form.can_save
        assert all(state.text == "" and state.proposal is None for state in form.fields.values())

    def test_every_field_has_a_kind(self) -> None:
        assert set(FIELD_KINDS) == set(SCALAR_FIELD_NAMES)

    def test_a_blind_document_starts_empty_and_is_never_given_a_draft(self) -> None:
        blind = row(label_mode="blind")
        assert LabelForm.start(blind).label_mode == "blind"
        with pytest.raises(BlindDraftError, match="BLIND document"):
            LabelForm.start(blind, draft())
        assert isinstance(BlindDraftError("x"), LabelingError)

    def test_a_draft_prefills_the_form_but_decides_nothing(self) -> None:
        form = LabelForm.start(row(), draft())
        assert form.fields["vendor_name"].text == "PT Contoh"
        assert form.fields["total"].text == "499500.00"
        assert form.fields["document_date"].text == "2024-03-05"
        assert form.fields["document_date"].proposal == "2024-03-05"
        assert all(state.status == "pending" for state in form.fields.values())
        assert form.rows[0] == {
            "description": "Kertas",
            "quantity": "10",
            "unit_price": "45000.00",
            "amount": "450000.00",
        }
        assert form.rows_proposal == "1 row(s)" and form.rows_status == "pending"
        assert not form.can_save

    def test_a_drafts_absent_and_unlabeled_are_shown_as_proposals_not_decisions(self) -> None:
        form = LabelForm.start(row(), draft())
        assert form.fields["due_date"].proposal == "absent" and form.fields["due_date"].text == ""
        assert form.fields["payment_method"].proposal == "unlabeled"
        assert form.fields["due_date"].status == "pending"

    def test_the_drafts_remarks_are_shown_but_not_made_the_labels_notes(self) -> None:
        form = LabelForm.start(row(), draft())
        assert form.fields["payment_method"].hint == "the model did not answer this field"
        assert form.fields["payment_method"].note == ""
        assert form.draft_notes == "pre-label draft by m; unverified" and form.notes == ""

    @pytest.mark.parametrize(("items", "proposal"), [(None, "absent"), (UNLABELED, "unlabeled")])
    def test_a_drafts_missing_table_is_a_proposal(self, items: Any, proposal: str) -> None:
        form = LabelForm.start(row(), draft(line_items=items))
        assert form.rows == [] and form.rows_proposal == proposal

    def test_draft_cells_that_are_null_or_unlabeled_become_empty_or_question_mark(self) -> None:
        form = LabelForm.start(
            row(),
            draft(line_items=[{"description": "A", "quantity": None, "unit_price": UNLABELED}]),
        )
        assert form.rows[0] == {
            "description": "A",
            "quantity": "",
            "unit_price": "?",
            "amount": "?",
        }

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"gt_from": "doc_009"}, "takes its labels from doc_009"),
            ({"capture": "constructed"}, "is constructed"),
            ({"label_mode": None}, "has no label_mode: run scripts/assign_split.py"),
        ],
    )
    def test_documents_a_person_does_not_label_are_refused(
        self, overrides: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(LabelingError, match=message):
            LabelForm.start(row("doc_002", **overrides))
        with pytest.raises(LabelingError, match=message):
            labelable_mode(row("doc_002", **overrides))


# ------------------------------------------------------------------ verifying a field


class TestVerify:
    def form(self) -> LabelForm:
        return LabelForm.start(row())

    def test_a_text_field_is_stripped_and_verified(self) -> None:
        form = self.form()
        form.set_text("vendor_name", "  PT  Contoh ")
        assert form.verify("vendor_name")
        assert (form.fields["vendor_name"].text, form.fields["vendor_name"].status) == (
            "PT  Contoh",
            "verified",
        )

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_an_empty_box_cannot_be_verified(self, blank: str) -> None:
        form = self.form()
        form.set_text("vendor_name", blank)
        assert not form.verify("vendor_name")
        state = form.fields["vendor_name"]
        assert state.status == "pending" and "mark the field absent" in (state.error or "")

    @pytest.mark.parametrize(
        ("text", "stored"),
        [
            ("Rp 1.250.000,00", "1250000.00"),
            ("1.250.000", "1250000"),
            ("9.116,8", "9116.8"),
            ("(1.500,00)", "-1500.00"),
            ("1250000.00", "1250000.00"),
        ],
    )
    def test_money_is_normalized_when_verified(self, text: str, stored: str) -> None:
        form = self.form()
        form.set_text("total", text)
        assert form.verify("total")
        assert form.fields["total"].text == stored  # the box now shows exactly what is stored

    def test_an_ambiguous_amount_is_an_inline_error_never_a_guess(self) -> None:
        form = self.form()
        form.set_text("total", "1.250")
        assert not form.verify("total")
        state = form.fields["total"]
        assert state.status == "pending" and state.text == "1.250"
        assert "ambiguous" in (state.error or "") and "Set the currency first" in (
            state.error or ""
        )

    def test_the_currency_settles_it(self) -> None:
        form = self.form()
        form.set_text("currency", "Rp")
        assert form.verify("currency") and form.fields["currency"].text == "IDR"
        form.set_text("total", "1.250")
        assert form.verify("total") and form.fields["total"].text == "1250"
        usd = self.form()
        usd.set_text("currency", "USD")
        usd.verify("currency")
        usd.set_text("total", "1.250")
        assert usd.verify("total") and usd.fields["total"].text == "1.250"

    def test_the_currency_box_counts_even_before_it_is_verified(self) -> None:
        form = self.form()
        form.set_text("currency", "Rp")  # typed, not yet verified
        assert form.currency_context() == "IDR"
        form.set_text("total", "1.250")
        assert form.verify("total") and form.fields["total"].text == "1250"

    @pytest.mark.parametrize("decide", ["mark_absent", "mark_unlabeled"])
    def test_an_absent_or_unlabeled_currency_gives_no_context(self, decide: str) -> None:
        form = self.form()
        form.set_text("currency", "Rp")
        getattr(form, decide)("currency")
        assert form.currency_context() is None

    def test_an_unreadable_amount_is_invalid(self) -> None:
        form = self.form()
        form.set_text("tax", "abc")
        assert not form.verify("tax")
        assert "invalid" in (form.fields["tax"].error or "")

    @pytest.mark.parametrize(
        ("text", "code"), [("Rp", "IDR"), ("rupiah", "IDR"), ("usd", "USD"), ("CHF", "CHF")]
    )
    def test_currencies(self, text: str, code: str) -> None:
        form = self.form()
        form.set_text("currency", text)
        assert form.verify("currency") and form.fields["currency"].text == code

    def test_an_unknown_currency_is_refused(self) -> None:
        form = self.form()
        form.set_text("currency", "rupee-ish")
        assert not form.verify("currency")
        assert "not a currency I know" in (form.fields["currency"].error or "")

    @pytest.mark.parametrize(
        ("text", "iso"),
        [
            ("5 Maret 2024", "2024-03-05"),
            ("2024-03-05", "2024-03-05"),
            ("13/03/2024", "2024-03-13"),
        ],
    )
    def test_dates_become_iso(self, text: str, iso: str) -> None:
        form = self.form()
        form.set_text("document_date", text)
        assert form.verify("document_date") and form.fields["document_date"].text == iso

    def test_a_slash_date_needs_a_convention(self) -> None:
        form = self.form()
        form.set_text("document_date", "05/03/2024")
        assert not form.verify("document_date")
        assert "ambiguous" in (form.fields["document_date"].error or "")
        assert "YYYY-MM-DD" in (form.fields["document_date"].error or "")
        form.set_text("currency", "Rp")
        assert form.verify("document_date") and form.fields["document_date"].text == "2024-03-05"

    def test_an_impossible_date_is_invalid(self) -> None:
        form = self.form()
        form.set_text("due_date", "31/02/2024")
        form.set_text("currency", "Rp")
        assert not form.verify("due_date")
        assert "invalid" in (form.fields["due_date"].error or "")

    def test_a_failed_verification_leaves_the_field_as_it_was(self) -> None:
        form = self.form()
        form.set_text("total", "1.250000")
        form.set_text("currency", "Rp")
        assert form.verify("total")
        form.set_text("total", "x")
        assert not form.verify("total")
        assert form.fields["total"].status == "pending" and form.fields["total"].text == "x"

    def test_a_later_success_clears_the_error(self) -> None:
        form = self.form()
        form.set_text("total", "1.250")
        form.verify("total")
        form.set_text("total", "1250")
        assert form.fields["total"].error is None
        assert form.verify("total") and form.fields["total"].error is None


class TestDecisions:
    def test_editing_the_box_undoes_the_decision(self) -> None:
        form = LabelForm.start(row())
        form.set_text("vendor_name", "A")
        form.verify("vendor_name")
        form.set_text("vendor_name", "B")
        assert form.fields["vendor_name"].status == "pending"

    def test_typing_the_same_text_keeps_the_decision(self) -> None:
        form = LabelForm.start(row())
        form.set_text("vendor_name", "A")
        form.verify("vendor_name")
        form.set_text("vendor_name", "A")
        assert form.fields["vendor_name"].status == "verified"

    def test_absent_and_unlabeled_clear_the_box(self) -> None:
        form = LabelForm.start(row())
        form.set_text("due_date", "5 Maret 2024")
        form.mark_absent("due_date")
        assert form.fields["due_date"] == FieldState(status="absent")
        form.set_text("discount", "10")
        form.mark_unlabeled("discount")
        assert form.fields["discount"] == FieldState(status="unlabeled")

    def test_editing_after_absent_makes_it_pending_again(self) -> None:
        form = LabelForm.start(row())
        form.mark_absent("due_date")
        form.set_text("due_date", "5 Maret 2024")
        assert form.fields["due_date"].status == "pending"

    def test_reset_makes_a_decided_field_pending_without_touching_its_text(self) -> None:
        form = LabelForm.start(row())
        form.set_text("vendor_name", "A")
        form.verify("vendor_name")
        form.reset("vendor_name")
        assert (form.fields["vendor_name"].status, form.fields["vendor_name"].text) == (
            "pending",
            "A",
        )

    def test_a_note_does_not_change_the_decision(self) -> None:
        form = LabelForm.start(row())
        form.mark_absent("total")
        form.set_note("total", "total not printed")
        assert (
            form.fields["total"].status == "absent"
            and form.fields["total"].note == "total not printed"
        )
        form.set_note("line_items", "two pages")
        assert form.rows_note == "two pages"

    @pytest.mark.parametrize("action", ["verify", "mark_absent", "mark_unlabeled", "reset"])
    def test_an_unknown_field_is_an_error(self, action: str) -> None:
        with pytest.raises(LabelingError, match="no such field: 'totl'"):
            getattr(LabelForm.start(row()), action)("totl")

    def test_progress_counts_the_decided_fields_including_the_table(self) -> None:
        form = LabelForm.start(row())
        form.mark_absent("due_date")
        form.mark_unlabeled("discount")
        form.set_text("vendor_name", "A")
        form.verify("vendor_name")
        form.mark_rows_absent()
        assert form.progress() == (4, len(SCALAR_FIELD_NAMES) + 1)
        assert "due_date" not in form.pending() and "line_items" not in form.pending()


# ------------------------------------------------------------------ the table


class TestRows:
    def test_cells_are_cleaned_and_a_change_makes_the_table_pending(self) -> None:
        form = LabelForm.start(row())
        form.set_rows([{"description": " A ", "quantity": None, "unit_price": "5", "amount": 7}])
        assert form.rows == [{"description": "A", "quantity": "", "unit_price": "5", "amount": "7"}]
        assert form.verify_rows() and form.rows_status == "verified"
        form.set_rows(form.rows)  # nothing changed
        assert form.rows_status == "verified"
        form.set_rows([{"description": "B"}])
        assert form.rows_status == "pending" and form.rows_values == []

    def test_missing_keys_are_empty_cells(self) -> None:
        form = LabelForm.start(row())
        form.set_rows([{"description": "A"}])
        assert form.rows == [{"description": "A", "quantity": "", "unit_price": "", "amount": ""}]

    def test_verifying_normalizes_every_cell_and_shows_what_is_stored(self) -> None:
        form = LabelForm.start(row())
        form.set_text("currency", "Rp")
        form.set_rows(
            [
                {
                    "description": "Kertas A4",
                    "quantity": "2,5",
                    "unit_price": "1.250,00",
                    "amount": "3.125",
                }
            ]
        )
        assert form.verify_rows()
        assert form.rows == [
            {
                "description": "Kertas A4",
                "quantity": "2.5",
                "unit_price": "1250.00",
                "amount": "3125",
            }
        ]
        assert form.rows_values == [
            {
                "description": "Kertas A4",
                "quantity": "2.5",
                "unit_price": "1250.00",
                "amount": "3125",
            }
        ]

    def test_an_empty_cell_is_absent_and_a_question_mark_is_unlabeled(self) -> None:
        form = LabelForm.start(row())
        form.set_rows(
            [{"description": "Ongkos kirim", "quantity": "", "unit_price": "?", "amount": "5.000"}]
        )
        form.set_text("currency", "Rp")
        assert form.verify_rows()
        assert form.rows_values == [
            {
                "description": "Ongkos kirim",
                "quantity": None,
                "unit_price": UNLABELED,
                "amount": "5000",
            }
        ]
        assert form.rows[0]["quantity"] == "" and form.rows[0]["unit_price"] == "?"

    def test_the_unlabeled_sentinel_itself_also_works_in_a_cell(self) -> None:
        form = LabelForm.start(row())
        form.set_rows(
            [{"description": UNLABELED, "quantity": "1", "unit_price": "1", "amount": "1"}]
        )
        assert form.verify_rows() and form.rows_values[0]["description"] == UNLABELED

    def test_ambiguous_cells_are_listed_by_row_and_column_and_nothing_changes(self) -> None:
        form = LabelForm.start(row())
        form.set_rows(
            [
                {"description": "A", "quantity": "1", "unit_price": "1.250", "amount": "1.250"},
                {"description": "B", "quantity": "x", "unit_price": "10", "amount": "10"},
            ]
        )
        assert not form.verify_rows()
        assert form.rows_status == "pending" and form.rows_values == []
        assert len(form.rows_errors) == 3
        assert form.rows_errors[0].startswith("row 1 unit_price: ambiguous")
        assert form.rows_errors[1].startswith("row 1 amount: ambiguous")
        assert form.rows_errors[2].startswith("row 2 quantity: invalid")
        assert form.rows[0]["unit_price"] == "1.250"  # the text is left for the person to fix

    def test_the_currency_resolves_the_ambiguity(self) -> None:
        form = LabelForm.start(row())
        form.set_rows(
            [{"description": "A", "quantity": "1", "unit_price": "1.250", "amount": "1.250"}]
        )
        assert not form.verify_rows()
        form.set_text("currency", "Rp")
        assert form.verify_rows() and form.rows_errors == []

    def test_an_empty_table_cannot_be_verified(self) -> None:
        form = LabelForm.start(row())
        assert not form.verify_rows()
        assert "mark the table absent" in form.rows_errors[0]

    def test_a_table_can_be_absent_or_unlabeled(self) -> None:
        form = LabelForm.start(row())
        form.set_rows([{"description": "A"}])
        form.mark_rows_absent()
        assert form.rows_status == "absent" and form.rows_values == []
        form.mark_rows_unlabeled()
        assert form.rows_status == "unlabeled"

    @pytest.mark.parametrize("decide", ["mark_rows_absent", "mark_rows_unlabeled"])
    def test_deciding_the_table_discards_what_the_rows_had_produced(self, decide: str) -> None:
        verified = LabelForm.start(row())
        verified.set_rows([{"description": "A", "quantity": "1", "unit_price": "2", "amount": "2"}])
        assert verified.verify_rows() and verified.rows_values
        getattr(verified, decide)()
        assert verified.rows_values == [] and verified.rows_errors == []
        failed = LabelForm.start(row())
        failed.set_rows([{"description": "A", "quantity": "x"}])
        assert not failed.verify_rows() and failed.rows_errors
        getattr(failed, decide)()
        assert failed.rows_errors == []  # the errors were about rows that no longer count

    def test_rows_can_be_added_and_removed_and_each_change_needs_a_fresh_verification(self) -> None:
        form = LabelForm.start(row())
        form.add_row()
        assert form.rows == [dict.fromkeys(ROW_CELLS, "")]
        form.set_rows([{"description": "A"}, {"description": "B"}])
        form.verify_rows()
        form.remove_row(0)
        assert [r["description"] for r in form.rows] == ["B"] and form.rows_status == "pending"
        form.verify_rows()
        form.add_row()
        assert form.rows_status == "pending" and len(form.rows) == 2

    @pytest.mark.parametrize("index", [-1, 1, 5])
    def test_removing_a_row_that_is_not_there_is_an_error(self, index: int) -> None:
        form = LabelForm.start(row())
        form.add_row()
        with pytest.raises(LabelingError, match=f"no row {index + 1}"):
            form.remove_row(index)


# ------------------------------------------------------------------ the finished label


class TestBuildLabel:
    def test_it_is_refused_while_anything_is_pending(self) -> None:
        form = LabelForm.start(row())
        with pytest.raises(LabelingError, match=r"14 field\(s\) still pending: vendor_name"):
            form.build_label(NOW)
        form.mark_absent("vendor_name")
        with pytest.raises(LabelingError, match=r"13 field\(s\) still pending: vendor_address"):
            form.build_label(NOW)

    def test_the_table_has_to_be_decided_too(self) -> None:
        form = complete(LabelForm.start(row()))
        form.set_rows([{"description": "Changed"}])
        with pytest.raises(LabelingError, match=r"1 field\(s\) still pending: line_items"):
            form.build_label(NOW)

    def test_a_completed_form_becomes_a_label(self) -> None:
        label = complete(LabelForm.start(row())).build_label(NOW)
        assert label.doc_id == "doc_001" and label.label_mode == "prelabel"
        assert label.labeled_at == NOW
        assert label.vendor_name == "PT Contoh Makmur"
        assert str(label.document_date) == "2024-03-05" and label.currency == "IDR"
        assert label.subtotal == Decimal("450000.00") and label.total == Decimal("499500.00")
        assert label.due_date is None and label.discount is None  # decided absent
        assert [item.model_dump() for item in label.line_items] == [
            {
                "description": "Kertas",
                "quantity": Decimal(10),
                "unit_price": Decimal("45000.00"),
                "amount": None,
            }
        ]

    def test_unlabeled_fields_are_stored_as_the_sentinel(self) -> None:
        form = complete(LabelForm.start(row()))
        form.mark_unlabeled("tax")
        form.mark_rows_unlabeled()
        label = form.build_label(NOW)
        assert label.tax == UNLABELED and label.line_items == UNLABELED

    def test_a_table_marked_absent_is_null(self) -> None:
        form = complete(LabelForm.start(row()))
        form.mark_rows_absent()
        assert form.build_label(NOW).line_items is None

    def test_notes_are_kept_only_when_there_is_something_in_them(self) -> None:
        form = complete(LabelForm.start(row()))
        form.set_note("total", " total not printed ")
        form.set_note("discount", "   ")
        form.set_note("line_items", "continued on page 2")
        form.notes = "  a bundle page  "
        label = form.build_label(NOW)
        assert label.field_notes == {
            "total": "total not printed",
            "line_items": "continued on page 2",
        }
        assert label.notes == "a bundle page"
        assert complete(LabelForm.start(row())).build_label(NOW).notes is None

    def test_the_mode_comes_from_the_manifest(self) -> None:
        assert (
            complete(LabelForm.start(row(label_mode="blind"))).build_label(NOW).label_mode
            == "blind"
        )

    def test_the_timestamp_defaults_to_now_in_utc_and_must_have_a_timezone(self) -> None:
        stamp = complete(LabelForm.start(row())).build_label().labeled_at
        assert stamp is not None and stamp.utcoffset() == timedelta(0)
        with pytest.raises(LabelingError, match="needs a timezone"):
            complete(LabelForm.start(row())).build_label(datetime(2026, 10, 6, 9, 15))
        local = datetime(2026, 10, 6, 9, 15, tzinfo=timezone(timedelta(hours=7)))
        assert complete(LabelForm.start(row())).build_label(local).labeled_at == local

    def test_the_label_survives_the_file_format(self) -> None:
        label = complete(LabelForm.start(row())).build_label(NOW)
        assert parse_label(dumps_label(label), source="x") == label
        data = json.loads(dumps_label(label))
        assert data["labeled_at"] == "2026-10-06T09:15:00Z" and data["due_date"] is None
        assert data["subtotal"] == "450000.00"  # the printed scale, as a string

    def test_a_blind_label_built_from_an_empty_form_has_no_draft_in_it(self) -> None:
        label = complete(LabelForm.start(row(label_mode="blind"))).build_label(NOW)
        assert "draft" not in (label.notes or "")

    def test_a_draft_that_is_verified_unchanged_becomes_the_label(self) -> None:
        form = LabelForm.start(row(), draft())
        for name in SCALAR_FIELD_NAMES:
            proposal = form.fields[name].proposal
            if proposal == "absent":
                form.mark_absent(name)
            elif proposal == "unlabeled":
                form.mark_unlabeled(name)
            else:
                assert form.verify(name), (name, form.fields[name].error)
        assert form.verify_rows()
        label = form.build_label(NOW)
        assert label.vendor_name == "PT Contoh" and label.total == Decimal("499500.00")
        assert label.payment_method == UNLABELED and label.due_date is None
        assert label.notes is None  # the draft's own remark is not part of the label


class TestReopen:
    def test_a_finished_label_reopens_fully_decided(self) -> None:
        label = complete(LabelForm.start(row())).build_label(NOW)
        form = LabelForm.from_label(row(), label)
        assert form.pending() == [] and form.can_save
        assert form.fields["vendor_name"].status == "verified"
        assert form.fields["due_date"].status == "absent"
        assert form.rows_status == "verified" and form.rows[0]["description"] == "Kertas"
        assert form.build_label(NOW) == label

    def test_unlabeled_and_absent_survive_reopening(self) -> None:
        source = complete(LabelForm.start(row()))
        source.mark_unlabeled("tax")
        source.mark_rows_absent()
        form = LabelForm.from_label(row(), source.build_label(NOW))
        assert form.fields["tax"].status == "unlabeled" and form.rows_status == "absent"

    def test_notes_survive_reopening(self) -> None:
        source = complete(LabelForm.start(row()))
        source.set_note("total", "n")
        source.set_note("line_items", "m")
        source.notes = "doc"
        form = LabelForm.from_label(row(), source.build_label(NOW))
        assert (form.fields["total"].note, form.rows_note, form.notes) == ("n", "m", "doc")

    def test_editing_a_reopened_field_makes_it_pending(self) -> None:
        form = LabelForm.from_label(row(), complete(LabelForm.start(row())).build_label(NOW))
        form.set_text("vendor_name", "Someone else")
        assert form.pending() == ["vendor_name"] and not form.can_save

    def test_a_label_whose_mode_disagrees_with_the_manifest_is_refused(self) -> None:
        label = complete(LabelForm.start(row())).build_label(NOW)
        with pytest.raises(
            LabelingError, match="label_mode='prelabel' but the manifest says 'blind'"
        ):
            LabelForm.from_label(row(label_mode="blind"), label)


class TestPersistence:
    def test_work_in_progress_round_trips(self) -> None:
        form = LabelForm.start(row(), draft())
        form.set_text("vendor_name", "Edited")
        form.verify("vendor_name")
        form.mark_absent("due_date")
        form.set_note("total", "n")
        form.notes = "doc note"
        again = LabelForm.from_json(json.loads(json.dumps(form.to_json())))
        assert again == form

    def test_a_verified_table_is_verified_again_on_load(self) -> None:
        form = complete(LabelForm.start(row()))
        again = LabelForm.from_json(json.loads(json.dumps(form.to_json())))
        assert again.rows_status == "verified" and again.rows == form.rows
        assert again.rows_values == form.rows_values
        assert again.build_label(NOW) == form.build_label(NOW)

    def test_a_completed_form_round_trips_to_the_same_label(self) -> None:
        form = complete(LabelForm.start(row()))
        form.set_note("total", "n")
        again = LabelForm.from_json(json.loads(json.dumps(form.to_json())))
        assert again.can_save and again.build_label(NOW) == form.build_label(NOW)

    def test_absent_unlabeled_and_pending_tables_keep_their_status(self) -> None:
        for decide, status in (
            ("mark_rows_absent", "absent"),
            ("mark_rows_unlabeled", "unlabeled"),
        ):
            form = LabelForm.start(row())
            getattr(form, decide)()
            assert LabelForm.from_json(form.to_json()).rows_status == status
        assert LabelForm.from_json(LabelForm.start(row()).to_json()).rows_status == "pending"

    def test_a_field_claimed_verified_must_verify_again(self) -> None:
        data = LabelForm.start(row()).to_json()
        data["fields"]["total"] = {"text": "1.250", "status": "verified"}  # ambiguous, no currency
        data["fields"]["vendor_name"] = {"text": "PT Contoh", "status": "verified"}
        data["fields"]["tax"] = {"text": "", "status": "verified"}  # nothing to verify
        form = LabelForm.from_json(data)
        assert form.fields["total"].status == "pending" and "ambiguous" in (
            form.fields["total"].error or ""
        )
        assert form.fields["tax"].status == "pending"
        assert form.fields["vendor_name"].status == "verified"

    def test_a_table_claimed_verified_must_verify_again(self) -> None:
        data = LabelForm.start(row()).to_json()
        data["rows"] = [
            {"description": "A", "quantity": "1", "unit_price": "1.250", "amount": "1.250"}
        ]
        data["rows_status"] = "verified"
        form = LabelForm.from_json(data)
        assert form.rows_status == "pending" and form.rows_errors

    def test_a_form_loaded_from_tampered_work_still_cannot_save_what_was_never_decided(
        self,
    ) -> None:
        data = LabelForm.start(row()).to_json()
        data["fields"]["total"] = {"text": "not a number", "status": "verified"}
        assert not LabelForm.from_json(data).can_save

    def test_it_is_plain_json(self) -> None:
        text = json.dumps(complete(LabelForm.start(row())).to_json())
        assert "doc_001" in text

    @pytest.mark.parametrize(
        "broken",
        [
            {},
            {"doc_id": "d"},
            {"doc_id": "d", "label_mode": "bogus"},
            {"doc_id": "d", "label_mode": "blind", "fields": 3},
            {"doc_id": "d", "label_mode": "blind", "fields": {"total": 3}},
            {"doc_id": "d", "label_mode": "blind", "fields": {"total": {"status": "weird"}}},
            {"doc_id": "d", "label_mode": "blind", "fields": {"total": {"text": 5}}},
            {"doc_id": "d", "label_mode": "blind", "fields": {"total": {"colour": "red"}}},
            {"doc_id": "d", "label_mode": "blind", "rows": 7},
            {"doc_id": "d", "label_mode": "blind", "rows": [3]},
            {"doc_id": "d", "label_mode": "blind", "rows": [{"amount": 5}]},
            {"doc_id": "d", "label_mode": "blind", "rows_status": "weird"},
            {"doc_id": "d", "label_mode": "blind", "notes": ["x"]},
        ],
    )
    def test_unusable_saved_work_is_a_labeling_error(self, broken: dict[str, Any]) -> None:
        with pytest.raises(LabelingError, match="unusable saved work"):
            LabelForm.from_json(broken)

    def test_minimal_saved_work_is_fine(self) -> None:
        form = LabelForm.from_json({"doc_id": "d", "label_mode": "blind"})
        assert form.pending() == [*SCALAR_FIELD_NAMES, "line_items"]


# ------------------------------------------------------------------ the changelog lines


class TestChangelogLines:
    def label(self, **changes: Any) -> GroundTruth:
        form = complete(LabelForm.start(row()))
        built = form.build_label(NOW)
        return built.model_copy(update=changes) if changes else built

    def test_a_new_label_is_one_line(self) -> None:
        assert changelog_lines(None, self.label(), today=TODAY, reason=None) == [
            "2026-10-06 | doc_001.json | (new file) | - -> label_mode=prelabel | labeled by hand"
        ]

    def test_a_replaced_label_logs_every_changed_field_with_the_reason(self) -> None:
        old = self.label()
        new = self.label(vendor_name="PT Baru", total=Decimal("1"))
        assert changelog_lines(old, new, today=TODAY, reason="read the stamp wrong") == [
            "2026-10-06 | doc_001.json | vendor_name | PT Contoh Makmur -> PT Baru"
            " | read the stamp wrong",
            "2026-10-06 | doc_001.json | total | 499500.00 -> 1 | read the stamp wrong",
        ]

    def test_the_timestamp_alone_is_not_a_change(self) -> None:
        old = self.label()
        new = self.label(labeled_at=NOW + timedelta(days=3))
        assert changelog_lines(old, new, today=TODAY, reason="r") == []

    def test_absent_unlabeled_and_values_are_shown_plainly(self) -> None:
        old = self.label()
        new = self.label(due_date=date(2024, 4, 1), discount=UNLABELED, tax=None)
        lines = changelog_lines(old, new, today=TODAY, reason="r")
        assert any("due_date | null -> 2024-04-01" in line for line in lines)
        assert any("discount | null -> __unlabeled__" in line for line in lines)
        assert any("tax | 49500.00 -> null" in line for line in lines)

    def test_a_table_change_is_summarized_by_row_count(self) -> None:
        old = self.label()
        new = self.label(line_items=[])
        (line,) = changelog_lines(old, new, today=TODAY, reason="r")
        assert "line_items | 1 rows -> 0 rows" in line

    def test_long_values_are_truncated_and_newlines_flattened(self) -> None:
        old = self.label(vendor_address="x" * 200)
        new = self.label(vendor_address="line one\nline two")
        (line,) = changelog_lines(old, new, today=TODAY, reason="r")
        assert "x" * 57 + "..." in line and "line one line two" in line

    def test_notes_and_field_notes_are_logged_too(self) -> None:
        old = self.label()
        new = self.label(notes="new note", field_notes={"total": "n"})
        fields = {
            line.split(" | ")[2] for line in changelog_lines(old, new, today=TODAY, reason="r")
        }
        assert fields == {"notes", "field_notes"}


# ------------------------------------------------------------------ saving


class Place:
    def __init__(self, root: Path) -> None:
        self.gt = root / "ground_truth"
        self.drafts = self.gt / "drafts"
        self.changelog = self.gt / "CHANGELOG.md"

    def save(self, label: GroundTruth, r: ManifestRow | None = None, **kwargs: Any) -> Any:
        return save_label(
            label,
            r or row(label_mode=label.label_mode),
            ground_truth_dir=self.gt,
            changelog=self.changelog,
            drafts_dir=self.drafts,
            today=kwargs.pop("today", TODAY),
            **kwargs,
        )

    def log(self) -> list[str]:
        return (
            self.changelog.read_text(encoding="utf-8").splitlines()
            if self.changelog.is_file()
            else []
        )


@pytest.fixture
def place(tmp_path: Path) -> Place:
    return Place(tmp_path)


def finished(**changes: Any) -> GroundTruth:
    label = complete(LabelForm.start(row())).build_label(NOW)
    return label.model_copy(update=changes) if changes else label


class TestSave:
    def test_a_new_label_is_written_and_logged(self, place: Place) -> None:
        result = place.save(finished())
        assert result.status == "created" and result.path == place.gt / "doc_001.json"
        assert read_label(result.path) == finished()
        assert result.changelog == [
            "2026-10-06 | doc_001.json | (new file) | - -> label_mode=prelabel | labeled by hand"
        ]
        assert place.log()[-1] == result.changelog[0]

    def test_the_file_records_how_and_when_it_was_made(self, place: Place) -> None:
        place.save(finished())
        data = json.loads((place.gt / "doc_001.json").read_text(encoding="utf-8"))
        assert data["label_mode"] == "prelabel" and data["labeled_at"] == "2026-10-06T09:15:00Z"

    def test_saving_the_same_label_again_changes_nothing(self, place: Place) -> None:
        place.save(finished())
        before, log = (place.gt / "doc_001.json").read_bytes(), place.log()
        result = place.save(finished(labeled_at=NOW + timedelta(days=2)))
        assert result.status == "unchanged" and result.changelog == []
        assert (place.gt / "doc_001.json").read_bytes() == before  # the old timestamp stays
        assert place.log() == log

    def test_replacing_a_finished_label_needs_a_reason(self, place: Place) -> None:
        place.save(finished())
        before, log = (place.gt / "doc_001.json").read_bytes(), place.log()
        for reason in (None, "", "   "):
            with pytest.raises(LabelingError, match="already exists: say why"):
                place.save(finished(vendor_name="Changed"), reason=reason)
        assert (place.gt / "doc_001.json").read_bytes() == before and place.log() == log

    def test_a_replacement_is_logged_field_by_field_with_its_reason(self, place: Place) -> None:
        place.save(finished())
        result = place.save(
            finished(vendor_name="PT Baru", total=Decimal(5)), reason="  checked the stamp  "
        )
        assert result.status == "replaced"
        assert read_label(place.gt / "doc_001.json").vendor_name == "PT Baru"
        assert result.changelog == [
            "2026-10-06 | doc_001.json | vendor_name | PT Contoh Makmur -> PT Baru"
            " | checked the stamp",
            "2026-10-06 | doc_001.json | total | 499500.00 -> 5 | checked the stamp",
        ]
        assert place.log()[-2:] == result.changelog

    def test_the_log_date_defaults_to_the_day_it_was_labeled(self, place: Place) -> None:
        result = save_label(
            finished(),
            row(),
            ground_truth_dir=place.gt,
            changelog=place.changelog,
        )
        assert result.changelog[0].startswith("2026-10-06 | ")


class TestSaveBlind:
    def blind(self) -> tuple[ManifestRow, GroundTruth]:
        r = row(label_mode="blind")
        return r, complete(LabelForm.start(r)).build_label(NOW)

    def test_a_blind_label_is_saved_when_no_draft_exists(self, place: Place) -> None:
        r, label = self.blind()
        assert place.save(label, r).status == "created"

    def test_a_blind_document_with_a_draft_is_refused_and_nothing_is_written(
        self, place: Place
    ) -> None:
        r, label = self.blind()
        write_label(place.drafts, draft())
        with pytest.raises(BlindDraftError, match="BLIND document but has a model draft"):
            place.save(label, r)
        assert not label_path(place.gt, "doc_001").exists() and place.log() == []

    def test_a_draft_does_not_stop_a_prelabel_document(self, place: Place) -> None:
        write_label(place.drafts, draft())
        assert place.save(finished()).status == "created"


class TestSaveRefusals:
    def test_the_label_must_be_for_the_document(self, place: Place) -> None:
        with pytest.raises(LabelingError, match="the label is for doc_001, not doc_002"):
            place.save(finished(), row("doc_002"))

    def test_the_label_mode_must_match_the_manifest(self, place: Place) -> None:
        with pytest.raises(
            LabelingError, match="label_mode='prelabel' but the manifest says 'blind'"
        ):
            place.save(finished(), row(label_mode="blind"))
        assert not place.gt.exists()

    @pytest.mark.parametrize(
        "overrides", [{"gt_from": "doc_009"}, {"capture": "constructed"}, {"label_mode": None}]
    )
    def test_documents_that_are_not_hand_labeled_cannot_be_saved(
        self, place: Place, overrides: dict[str, Any]
    ) -> None:
        with pytest.raises(LabelingError):
            place.save(finished(), row(**overrides))
        assert not place.gt.exists()


class TestSaveIsAtomicWithItsLog:
    def failing_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> None:
            raise OSError("log is locked")

        monkeypatch.setattr("evalkit.labeling.append_changelog", boom)

    def test_a_new_label_is_removed_if_it_cannot_be_logged(
        self, place: Place, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.failing_log(monkeypatch)
        with pytest.raises(OSError, match="log is locked"):
            place.save(finished())
        assert not label_path(place.gt, "doc_001").exists()  # never a label without its log line

    def test_a_replaced_label_is_put_back_if_it_cannot_be_logged(
        self, place: Place, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        place.save(finished())
        before = (place.gt / "doc_001.json").read_bytes()
        self.failing_log(monkeypatch)
        with pytest.raises(OSError, match="log is locked"):
            place.save(finished(vendor_name="Changed"), reason="why")
        assert (place.gt / "doc_001.json").read_bytes() == before
