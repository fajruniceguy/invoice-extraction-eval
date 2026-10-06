import json
import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pytest
from pydantic import ValidationError

from docextract.config import IngestSettings
from docextract.ingest.manifest import (
    ManifestRow,
    check_manifest,
    missing_fields,
    read_manifest,
    write_manifest,
)
from docextract.ingest.pipeline import ingest_document
from docextract.ingest.splits import assign_splits
from docextract.normalize import parse_date, parse_number
from evalkit.construct import (
    ITEM_NAMES,
    LAYOUTS,
    MIN_LAST_PAGE_ROWS,
    Invoice,
    format_date,
    format_money,
    format_quantity,
    generate_invoice,
    ground_truth,
    gt_money,
    manifest_row,
    page_plan,
    paginate,
    render_invoice,
    row_capacities,
)
from evalkit.construct_io import (
    LAYOUT_ORDER,
    ConstructError,
    build_invoices,
    construct,
    doc_ids,
)
from evalkit.groundtruth import SCALAR_FIELD_NAMES, GroundTruth
from evalkit.gt_store import PLACEHOLDER
from ingest_support import run_script

TODAY = date(2026, 10, 6)


@pytest.fixture(scope="module")
def invoices() -> list[Invoice]:
    return build_invoices()


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory, invoices: list[Invoice]) -> dict[str, Any]:
    """Each invoice rendered once: {doc_id: (path, rows_per_page, text_per_page)}."""
    folder = tmp_path_factory.mktemp("constructed")
    result: dict[str, Any] = {}
    for invoice in invoices:
        path = folder / f"{invoice.doc_id}.pdf"
        plan = render_invoice(invoice, path)
        pdf = pdfium.PdfDocument(str(path))
        texts = [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
        pdf.close()
        result[invoice.doc_id] = (path, plan, texts)
    return result


def by_layout(invoices: list[Invoice]) -> dict[str, Invoice]:
    return {invoice.layout: invoice for invoice in invoices}


# ------------------------------------------------------------------ formatting


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "decimals", "printed"),
        [
            ("1250000", True, "1.250.000,00"),
            ("1250000", False, "1.250.000"),
            ("0", True, "0,00"),
            ("999", False, "999"),
            ("1000", False, "1.000"),
            ("12345678.5", True, "12.345.678,50"),
            ("-1500", False, "-1.500"),
            ("1000000000", True, "1.000.000.000,00"),
        ],
    )
    def test_indonesian_money(self, value: str, decimals: bool, printed: str) -> None:
        assert format_money(Decimal(value), decimals) == printed

    @pytest.mark.parametrize(
        ("value", "decimals"),
        [("1250000", True), ("1250000", False), ("12345678", True), ("500", False)],
    )
    def test_the_normalizer_reads_back_what_was_printed(self, value: str, decimals: bool) -> None:
        printed = format_money(Decimal(value), decimals)
        result = parse_number(printed, currency="IDR")
        assert result.status == "ok"
        assert str(result.value) == gt_money(Decimal(value), decimals)

    def test_ground_truth_money_keeps_the_printed_scale(self) -> None:
        assert gt_money(Decimal(250000), True) == "250000.00"
        assert gt_money(Decimal(250000), False) == "250000"

    @pytest.mark.parametrize(
        ("value", "printed", "gt"),
        [("10", "10", "10"), ("2.5", "2,5", "2.5"), ("0.5", "0,5", "0.5")],
    )
    def test_quantities(self, value: str, printed: str, gt: str) -> None:
        assert format_quantity(Decimal(value)) == printed
        assert f"{Decimal(value):f}" == gt
        assert str(parse_number(printed, currency="IDR").value) == gt

    @pytest.mark.parametrize(
        ("style", "printed"),
        [("long", "5 Maret 2024"), ("slash", "05/03/2024"), ("iso", "2024-03-05")],
    )
    def test_dates_read_back_as_iso(self, style: str, printed: str) -> None:
        assert format_date(date(2024, 3, 5), style) == printed
        assert parse_date(printed, day_first=True).value == date(2024, 3, 5)

    def test_every_month_name_is_one_the_normalizer_knows(self) -> None:
        for month in range(1, 13):
            printed = format_date(date(2024, month, 17), "long")
            assert parse_date(printed).value == date(2024, month, 17), printed


# ------------------------------------------------------------------ generation


class TestGeneration:
    def test_three_layouts_in_three_documents(self, invoices: list[Invoice]) -> None:
        assert [i.doc_id for i in invoices] == ["cons_001", "cons_002", "cons_003"]
        assert (
            [i.layout for i in invoices] == list(LAYOUT_ORDER) == ["classic", "modern", "compact"]
        )

    def test_thirty_to_sixty_items_over_two_or_three_pages(self, invoices: list[Invoice]) -> None:
        for invoice in invoices:
            assert 30 <= len(invoice.items) <= 60, invoice.doc_id
            assert 2 <= len(page_plan(invoice)) <= 3, invoice.doc_id
        pages = {i.layout: len(page_plan(i)) for i in invoices}
        assert pages == {"classic": 2, "modern": 3, "compact": 2}

    def test_it_is_deterministic_and_the_inputs_matter(self) -> None:
        assert generate_invoice("cons_001", "classic") == generate_invoice("cons_001", "classic")
        base = generate_invoice("cons_001", "classic")
        # compare the CONTENT: the doc_id field alone would make any two invoices differ
        assert base.items != generate_invoice("cons_001", "classic", seed="other").items
        assert base.items != generate_invoice("cons_009", "classic").items
        assert base.items != generate_invoice("cons_001", "compact").items

    def test_the_printed_arithmetic_is_exact(self, invoices: list[Invoice]) -> None:
        for invoice in invoices:
            for item in invoice.items:
                assert item.amount == item.quantity * item.unit_price
                assert item.amount == item.amount.to_integral_value()
                assert item.unit_price % 500 == 0
            assert invoice.subtotal == sum((i.amount for i in invoice.items), Decimal(0))
            taxable = invoice.subtotal - (invoice.discount or 0)
            assert invoice.tax == (taxable * Decimal("0.11")).quantize(Decimal(1), ROUND_HALF_UP)
            assert invoice.total == taxable + invoice.tax

    def test_no_catalogue_name_contains_another(self) -> None:
        # a description that is a substring of another would make "where is this row?" ambiguous
        assert len(ITEM_NAMES) >= 60 and len(set(ITEM_NAMES)) == len(ITEM_NAMES)
        clashes = [(a, b) for a in ITEM_NAMES for b in ITEM_NAMES if a != b and a in b]
        assert clashes == []

    def test_descriptions_are_unique_within_an_invoice(self, invoices: list[Invoice]) -> None:
        for invoice in invoices:
            descriptions = [i.description for i in invoice.items]
            assert len(set(descriptions)) == len(descriptions)

    def test_only_the_compact_layout_has_fractional_quantities(
        self, invoices: list[Invoice]
    ) -> None:
        fractional = {
            i.layout: any(item.quantity != item.quantity.to_integral_value() for item in i.items)
            for i in invoices
        }
        assert fractional == {"classic": False, "modern": False, "compact": True}

    def test_layouts_differ_in_what_they_print(self, invoices: list[Invoice]) -> None:
        classic, modern, compact = (by_layout(invoices)[n] for n in LAYOUT_ORDER)
        assert modern.discount is None and classic.discount and compact.discount
        assert modern.due is None and classic.due and compact.due
        assert classic.discount == (classic.subtotal * Decimal("0.05")).quantize(
            Decimal(1), ROUND_HALF_UP
        )

    def test_all_amounts_are_whole_rupiah(self, invoices: list[Invoice]) -> None:
        for invoice in invoices:
            for value in (invoice.subtotal, invoice.tax, invoice.total, invoice.discount or 0):
                assert value == Decimal(value).to_integral_value()


class TestPagination:
    def test_every_allowed_count_paginates_into_two_or_three_pages(self) -> None:
        for layout in LAYOUTS.values():
            first, middle, last = row_capacities(layout)
            for count in range(layout.items[0], layout.items[1] + 1):
                sizes = paginate(count, first, middle, last)
                assert sum(sizes) == count
                assert 2 <= len(sizes) <= 3, (layout.name, count)
                assert MIN_LAST_PAGE_ROWS <= sizes[-1] <= last
                caps = [first, middle, middle][: len(sizes) - 1]
                assert all(size <= cap for size, cap in zip(sizes[:-1], caps, strict=True))
                assert all(size >= MIN_LAST_PAGE_ROWS for size in sizes)

    def test_rows_fill_pages_in_order_and_the_last_page_leaves_room_for_the_totals(self) -> None:
        assert paginate(40, 25, 30, 20) == [25, 15]  # two pages are enough
        assert paginate(45, 25, 30, 20) == [25, 20]  # the last page exactly full (with totals)
        assert paginate(60, 25, 30, 20) == [25, 30, 5]  # a third page

    @pytest.mark.parametrize(
        ("count", "sizes"),
        [
            (46, [25, 17, 4]),  # one row would spill: 25 + 21 + 0, so the last page borrows 4
            (51, [25, 22, 4]),
            (52, [25, 23, 4]),
            (56, [25, 27, 4]),  # 25 + 30 + 1 -> borrows 3
            (57, [25, 28, 4]),
            (58, [25, 29, 4]),
            (59, [25, 30, 4]),
            (60, [25, 30, 5]),
        ],
    )
    def test_a_last_page_with_too_few_rows_borrows_from_the_page_before(
        self, count: int, sizes: list[int]
    ) -> None:
        assert paginate(count, 25, 30, 20) == sizes

    def test_it_never_returns_fewer_than_two_pages(self) -> None:
        assert len(paginate(30, 40, 50, 40)) == 2  # fits on one page, still two: 26 + 4
        assert paginate(30, 40, 50, 40) == [26, 4]

    def test_a_count_too_small_to_fill_a_last_page_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot paginate"):
            paginate(5, 25, 30, 20)

    def test_the_capacities_follow_the_layout_geometry(self) -> None:
        classic = LAYOUTS["classic"]
        first, middle, last = row_capacities(classic)
        assert first == int((590 - 17 - 55) // 17)
        assert middle == int((780 - 17 - 55) // 17)
        assert last == middle - 7  # ceil(110 / 17) rows are reserved for the totals


# ------------------------------------------------------------------ the PDFs


class TestRendering:
    def test_the_page_count_matches_the_plan(
        self, invoices: list[Invoice], rendered: dict[str, Any]
    ) -> None:
        for invoice in invoices:
            _, plan, texts = rendered[invoice.doc_id]
            assert plan == page_plan(invoice)
            assert len(texts) == len(plan)

    def test_every_row_is_printed_on_its_page_exactly_as_labeled(
        self, invoices: list[Invoice], rendered: dict[str, Any]
    ) -> None:
        for invoice in invoices:
            layout = LAYOUTS[invoice.layout]
            _, plan, texts = rendered[invoice.doc_id]
            cursor = 0
            for page, rows in enumerate(plan):
                for item in invoice.items[cursor : cursor + rows]:
                    for printed in (
                        item.description,
                        format_quantity(item.quantity),
                        format_money(item.unit_price, layout.money_decimals),
                        format_money(item.amount, layout.money_decimals),
                    ):
                        assert printed in texts[page], (invoice.doc_id, page, printed)
                cursor += rows
            assert cursor == len(invoice.items)

    def test_each_row_is_printed_once_and_only_on_its_page(
        self, invoices: list[Invoice], rendered: dict[str, Any]
    ) -> None:
        for invoice in invoices:
            _, plan, texts = rendered[invoice.doc_id]
            cursor = 0
            for page, rows in enumerate(plan):
                for item in invoice.items[cursor : cursor + rows]:
                    for other, text in enumerate(texts):
                        count = text.count(item.description)
                        assert count == (1 if other == page else 0), (
                            invoice.doc_id,
                            item.description,
                        )
                cursor += rows

    def test_the_totals_are_on_the_last_page_only(
        self, invoices: list[Invoice], rendered: dict[str, Any]
    ) -> None:
        for invoice in invoices:
            layout = LAYOUTS[invoice.layout]
            _, _, texts = rendered[invoice.doc_id]
            last = len(texts) - 1
            figures = [invoice.subtotal, invoice.tax, invoice.total]
            labels = [layout.labels["subtotal"], layout.labels["tax"], layout.labels["total"]]
            if invoice.discount is not None:
                figures.append(invoice.discount)
                labels.append(layout.labels["discount"])
            for value in figures:
                printed = format_money(value, layout.money_decimals)
                assert printed in texts[last], (invoice.doc_id, printed)
                for page in range(last):
                    assert printed not in texts[page], (invoice.doc_id, page, printed)
            for label in labels:
                assert label in texts[last]
                assert all(label not in texts[page] for page in range(last)), (
                    invoice.doc_id,
                    label,
                )

    def test_a_layout_without_a_discount_prints_no_discount_line(
        self, invoices: list[Invoice], rendered: dict[str, Any]
    ) -> None:
        modern = by_layout(invoices)["modern"]
        _, _, texts = rendered[modern.doc_id]
        assert not any("Potongan" in t or "Diskon" in t or "Discount" in t for t in texts)

    def test_header_facts_are_printed_on_the_first_page(
        self, invoices: list[Invoice], rendered: dict[str, Any]
    ) -> None:
        for invoice in invoices:
            layout = LAYOUTS[invoice.layout]
            _, _, texts = rendered[invoice.doc_id]
            first = texts[0]
            assert invoice.vendor_name in first and invoice.vendor_address in first
            assert invoice.buyer_name in first and invoice.number in first
            assert format_date(invoice.issued, layout.date_style) in first
            assert (invoice.vendor_tax_id in first) is layout.prints_tax_id
            assert ("NPWP" in first) is layout.prints_tax_id
            assert ("Jatuh Tempo" in first) is layout.prints_due_date
            assert ("Pembayaran" in first) is layout.prints_payment_method
            if invoice.due is not None:
                assert format_date(invoice.due, layout.date_style) in first

    def test_only_one_layout_repeats_a_lanjutan_header_on_continuation_pages(
        self, invoices: list[Invoice], rendered: dict[str, Any]
    ) -> None:
        for invoice in invoices:
            _, _, texts = rendered[invoice.doc_id]
            continued = [("(lanjutan)" in t) for t in texts]
            if invoice.layout == "modern":
                assert continued == [False, True, True]
                assert all(f"Invoice No. {invoice.number}" in t for t in texts[1:])
            else:
                assert not any(continued)

    def test_every_page_says_which_page_it_is(
        self, invoices: list[Invoice], rendered: dict[str, Any]
    ) -> None:
        for invoice in invoices:
            _, _, texts = rendered[invoice.doc_id]
            for number, text in enumerate(texts, start=1):
                assert f"Halaman {number} dari {len(texts)}" in text

    def test_the_column_order_differs_between_layouts(self) -> None:
        orders = {name: tuple(c.key for c in layout.columns) for name, layout in LAYOUTS.items()}
        assert len(set(orders.values())) == 3
        assert orders["modern"].index("unit_price") < orders["modern"].index("qty")
        assert orders["classic"].index("qty") < orders["classic"].index("unit_price")

    def test_the_pdfs_are_byte_for_byte_reproducible(
        self, tmp_path: Path, invoices: list[Invoice]
    ) -> None:
        for invoice in invoices:
            render_invoice(invoice, tmp_path / "a.pdf")
            render_invoice(invoice, tmp_path / "b.pdf")
            assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()

    def test_a_description_that_does_not_fit_its_column_is_refused(self, tmp_path: Path) -> None:
        invoice = generate_invoice("cons_001", "classic")
        item = invoice.items[0]
        wide = type(item)("X" * 120, item.quantity, item.unit_price, item.amount)
        broken = Invoice(**{**invoice.__dict__, "items": (wide, *invoice.items[1:])})
        with pytest.raises(ValueError, match="does not fit its column"):
            render_invoice(broken, tmp_path / "x.pdf")


# ------------------------------------------------------------------ the ground truth


class TestGroundTruth:
    def test_it_loads_and_labels_every_scalar_field(self, invoices: list[Invoice]) -> None:
        for invoice in invoices:
            label = GroundTruth.model_validate(ground_truth(invoice))
            assert set(label.labeled_fields()) == set(SCALAR_FIELD_NAMES)  # none left unlabeled
            assert label.line_items not in (None, "__unlabeled__")

    def test_what_is_not_printed_is_explicit_null(self, invoices: list[Invoice]) -> None:
        modern = ground_truth(by_layout(invoices)["modern"])
        for field in ("due_date", "vendor_tax_id", "payment_method", "discount"):
            assert field in modern and modern[field] is None, field
        classic = ground_truth(by_layout(invoices)["classic"])
        assert all(
            classic[f] is not None
            for f in ("due_date", "vendor_tax_id", "payment_method", "discount")
        )

    def test_values_are_the_printed_ones_as_decimal_strings_and_iso_dates(
        self, invoices: list[Invoice], rendered: dict[str, Any]
    ) -> None:
        for invoice in invoices:
            layout = LAYOUTS[invoice.layout]
            label = ground_truth(invoice)
            _, _, texts = rendered[invoice.doc_id]
            joined = "\n".join(texts)
            for field in ("subtotal", "tax", "total", "discount"):
                if label[field] is None:
                    continue
                printed = format_money(Decimal(label[field]), layout.money_decimals)
                assert printed in joined
                assert (
                    str(parse_number(printed, currency="IDR").value) == label[field]
                )  # same scale
            assert label["document_date"] == invoice.issued.isoformat()
            printed_date = format_date(invoice.issued, layout.date_style)
            assert (
                parse_date(printed_date, day_first=True).value.isoformat() == label["document_date"]
            )
            assert label["currency"] == "IDR"
            for row in label["line_items"]:
                for key in ("unit_price", "amount"):
                    printed = format_money(Decimal(row[key]), layout.money_decimals)
                    assert str(parse_number(printed, currency="IDR").value) == row[key]
                assert row["quantity"] == f"{Decimal(row['quantity']):f}"

    def test_the_money_scale_is_the_printed_scale(self, invoices: list[Invoice]) -> None:
        scales = {i.layout: ground_truth(i)["total"].partition(".")[2] for i in invoices}
        assert scales == {"classic": "00", "modern": "", "compact": "00"}

    def test_line_items_are_in_printed_order(self, invoices: list[Invoice]) -> None:
        for invoice in invoices:
            rows = ground_truth(invoice)["line_items"]
            assert [r["description"] for r in rows] == [i.description for i in invoice.items]

    def test_the_notes_describe_the_construction(self, invoices: list[Invoice]) -> None:
        modern = ground_truth(by_layout(invoices)["modern"])["notes"]
        assert "modern layout" in modern and "3 pages" in modern and "(lanjutan)" in modern
        assert "not printed: due_date, vendor_tax_id, payment_method, discount" in modern
        classic = ground_truth(by_layout(invoices)["classic"])["notes"]
        assert "not printed" not in classic and "(lanjutan)" not in classic

    def test_the_manifest_row(self, invoices: list[Invoice]) -> None:
        for invoice in invoices:
            row = manifest_row(invoice)
            assert row.raw_files == (f"constructed/{invoice.doc_id}.pdf",)
            assert (row.capture, row.doc_type, row.content_group) == (
                "constructed",
                "invoice",
                invoice.doc_id,
            )
            assert row.template_group == f"constructed_{invoice.layout}"
            assert (row.language, row.original_format, row.needs_redaction) == ("id", "pdf", False)
            assert row.split is None and row.label_mode is None
            assert missing_fields(row, "ingest") == []


# ------------------------------------------------------------------ writing the files


class Dirs:
    def __init__(self, root: Path) -> None:
        self.raw = root / "raw"
        self.gt = root / "ground_truth"
        self.manifest = root / "manifest.csv"
        self.changelog = self.gt / "CHANGELOG.md"

    def run(self, **kwargs: Any) -> Any:
        kwargs.setdefault("today", TODAY)
        return construct(
            raw_dir=self.raw,
            ground_truth_dir=self.gt,
            manifest=self.manifest,
            changelog=self.changelog,
            **kwargs,
        )


@pytest.fixture
def dirs(tmp_path: Path) -> Dirs:
    return Dirs(tmp_path)


class TestConstruct:
    def test_it_writes_pdfs_labels_changelog_lines_and_manifest_rows(self, dirs: Dirs) -> None:
        actions = dirs.run()
        assert [(a.doc_id, a.pdf, a.ground_truth, a.manifest) for a in actions] == [
            (doc_id, "written", "created", "added") for doc_id in doc_ids()
        ]
        for doc_id in doc_ids():
            assert (dirs.raw / "constructed" / f"{doc_id}.pdf").read_bytes().startswith(b"%PDF")
            label = json.loads((dirs.gt / f"{doc_id}.json").read_text(encoding="utf-8"))
            assert label["doc_id"] == doc_id
            GroundTruth.model_validate(label)
        rows = read_manifest(dirs.manifest)
        assert [r.doc_id for r in rows] == doc_ids()
        assert check_manifest(rows) == []

    def test_each_new_label_is_logged_and_the_placeholder_is_replaced(self, dirs: Dirs) -> None:
        dirs.gt.mkdir(parents=True)
        dirs.changelog.write_text(
            "# Ground-truth changelog\n\n"
            "Format: `YYYY-MM-DD | file | field | old -> new | reason`\n\n"
            f"---\n\n{PLACEHOLDER}\n",
            encoding="utf-8",
        )
        dirs.run()
        text = dirs.changelog.read_text(encoding="utf-8")
        assert PLACEHOLDER not in text
        lines = [line for line in text.splitlines() if line.startswith("2026-10-06 | ")]
        assert len(lines) == 3
        for line, doc_id in zip(lines, doc_ids(), strict=True):
            assert re.fullmatch(
                rf"2026-10-06 \| {doc_id}\.json \| \(new file\) \| - -> \d+ pages, \d+ items \| .+",
                line,
            ), line
        for line, layout in zip(lines, LAYOUT_ORDER, strict=True):
            assert len(line.split(" | ")) == 5
            assert f"{layout} layout" in line and "scripts/construct_invoices.py" in line
        assert text.startswith("# Ground-truth changelog")
        assert "---" in text

    def test_running_again_changes_nothing_and_logs_nothing(self, dirs: Dirs) -> None:
        dirs.run()
        snapshot = {
            p: p.read_bytes() for p in [*dirs.gt.glob("*"), dirs.manifest, *dirs.raw.rglob("*.pdf")]
        }
        actions = dirs.run()
        assert all(
            (a.pdf, a.ground_truth, a.manifest) == ("unchanged", "unchanged", "present")
            for a in actions
        )
        assert snapshot == {p: p.read_bytes() for p in snapshot}

    def test_a_label_that_would_not_load_stops_everything_before_any_write(
        self, dirs: Dirs, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def typo(invoice: Invoice) -> dict[str, Any]:
            label = ground_truth(invoice)
            label["vendor_nme"] = label.pop("vendor_name")
            return label

        monkeypatch.setattr("evalkit.construct_io.ground_truth", typo)
        with pytest.raises(ValidationError, match="vendor_nme"):
            dirs.run()
        assert not dirs.raw.exists() and not dirs.gt.exists() and not dirs.manifest.exists()

    def test_check_writes_nothing(self, dirs: Dirs) -> None:
        actions = dirs.run(check=True)
        assert [a.ground_truth for a in actions] == ["created"] * 3
        assert not dirs.raw.exists() and not dirs.gt.exists() and not dirs.manifest.exists()

    def test_a_label_that_was_edited_is_never_overwritten_silently(self, dirs: Dirs) -> None:
        dirs.run()
        path = dirs.gt / "cons_002.json"
        label = json.loads(path.read_text(encoding="utf-8"))
        label["vendor_name"] = "Hand edited"
        path.write_text(json.dumps(label), encoding="utf-8")
        changelog_before = dirs.changelog.read_bytes()
        with pytest.raises(ConstructError, match=r"cons_002.json differs .*vendor_name.*refusing"):
            dirs.run()
        assert json.loads(path.read_text(encoding="utf-8"))["vendor_name"] == "Hand edited"
        assert dirs.changelog.read_bytes() == changelog_before

    def test_a_conflict_blocks_every_write_not_just_the_conflicting_file(self, dirs: Dirs) -> None:
        dirs.run()
        (dirs.raw / "constructed" / "cons_001.pdf").write_bytes(b"stale")
        path = dirs.gt / "cons_003.json"
        path.write_text(json.dumps({"doc_id": "cons_003"}), encoding="utf-8")
        with pytest.raises(ConstructError):
            dirs.run()
        assert (dirs.raw / "constructed" / "cons_001.pdf").read_bytes() == b"stale"

    def test_force_replaces_the_label_and_logs_what_changed(self, dirs: Dirs) -> None:
        dirs.run()
        path = dirs.gt / "cons_002.json"
        label = json.loads(path.read_text(encoding="utf-8"))
        original = label["vendor_name"]
        label["vendor_name"] = "Hand edited"
        path.write_text(json.dumps(label), encoding="utf-8")
        actions = dirs.run(force=True, today=date(2026, 11, 1))
        assert [a.ground_truth for a in actions] == ["unchanged", "replaced", "unchanged"]
        assert json.loads(path.read_text(encoding="utf-8"))["vendor_name"] == original
        last = dirs.changelog.read_text(encoding="utf-8").strip().splitlines()[-1]
        assert last.startswith("2026-11-01 | cons_002.json | vendor_name | (see git diff) | ")
        assert "--force" in last

    def test_another_seed_is_refused_without_force_and_logged_with_it(self, dirs: Dirs) -> None:
        dirs.run()
        with pytest.raises(ConstructError, match="seed 'other'"):
            dirs.run(seed="other")
        actions = dirs.run(seed="other", force=True)
        assert [a.ground_truth for a in actions] == ["replaced"] * 3
        assert dirs.changelog.read_text(encoding="utf-8").count("regenerated with --force") == 3

    def test_existing_manifest_rows_are_kept_and_new_ones_appended(self, dirs: Dirs) -> None:
        dirs.run()
        rows = read_manifest(dirs.manifest)
        other = rows[0].model_copy(
            update={
                "doc_id": "doc_001",
                "content_group": "doc_001",
                "capture": "phone_photo",
                "raw_files": ("x.jpg",),
            }
        )
        write_manifest(dirs.manifest, [other, *rows[1:]])  # drop cons_001's row, add a real doc
        dirs.run()
        assert [r.doc_id for r in read_manifest(dirs.manifest)] == [
            "doc_001",
            "cons_002",
            "cons_003",
            "cons_001",
        ]

    def test_a_manifest_row_that_is_not_constructed_is_a_conflict(self, dirs: Dirs) -> None:
        clash = ManifestRow(doc_id="cons_001", raw_files=("photo.jpg",), capture="phone_photo")
        write_manifest(dirs.manifest, [clash])
        with pytest.raises(
            ConstructError, match="cons_001 is in the manifest with capture=phone_photo"
        ):
            dirs.run()
        assert not dirs.gt.exists()

    def test_the_generated_documents_go_through_ingestion_and_splitting(
        self, dirs: Dirs, tmp_path: Path
    ) -> None:
        dirs.run()
        rows = read_manifest(dirs.manifest)
        settings = IngestSettings(_env_file=None, dpi=40)
        for row in rows:
            result = ingest_document(
                row, raw_dir=dirs.raw, out_dir=tmp_path / "pages" / row.doc_id, settings=settings
            )
            assert len(result.pages) == len(
                page_plan(next(i for i in build_invoices() if i.doc_id == row.doc_id))
            )
        plan = assign_splits(rows)
        assert [(s.stratum, s.groups, s.dev, s.test) for s in plan.strata] == [
            (("constructed", "invoice"), 3, 1, 2)
        ]
        assert all(r.label_mode is None for r in plan.rows)  # generated labels: nobody labels them
        assert plan.blind == []


# ------------------------------------------------------------------ the command line


@pytest.mark.slow
class TestCommandLine:
    def args(self, tmp_path: Path, *extra: str) -> list[str]:
        gt = tmp_path / "gt"
        return [
            "--raw-dir", str(tmp_path / "raw"),
            "--ground-truth-dir", str(gt),
            "--manifest", str(tmp_path / "manifest.csv"),
            "--changelog", str(gt / "CHANGELOG.md"),
            *extra,
        ]  # fmt: skip

    def test_it_constructs_then_reports_unchanged(self, tmp_path: Path) -> None:
        code, out, _ = run_script("construct_invoices.py", *self.args(tmp_path))
        assert code == 0, out
        assert re.search(
            r"cons_002  \d+ items, 3 pages \| pdf written \| ground truth created"
            r" \| manifest added",
            out,
        )
        code, out, _ = run_script("construct_invoices.py", *self.args(tmp_path))
        assert code == 0
        assert out.count("pdf unchanged | ground truth unchanged | manifest present") == 3

    def test_check_writes_nothing(self, tmp_path: Path) -> None:
        code, out, _ = run_script("construct_invoices.py", *self.args(tmp_path, "--check"))
        assert code == 0 and "(check only)" in out
        assert not (tmp_path / "gt").exists() and not (tmp_path / "raw").exists()

    def test_a_conflict_exits_1(self, tmp_path: Path) -> None:
        assert run_script("construct_invoices.py", *self.args(tmp_path))[0] == 0
        code, _, err = run_script("construct_invoices.py", *self.args(tmp_path, "--seed", "other"))
        assert code == 1 and "refusing to overwrite a label" in err
        code, out, _ = run_script(
            "construct_invoices.py", *self.args(tmp_path, "--seed", "other", "--force")
        )
        assert code == 0 and "ground truth replaced" in out

    def test_an_unreadable_manifest_exits_2(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.csv").write_text("nonsense\n", encoding="utf-8")
        code, _, err = run_script("construct_invoices.py", *self.args(tmp_path))
        assert code == 2 and "cannot be read" in err
