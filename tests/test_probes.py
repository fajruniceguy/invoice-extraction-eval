"""The hallucination probe set: erasure, labels, byte identity with the dataset, and the report.

The report world is worked out by hand (see `report_world`):

  c1 (control of source s1) reads everything right: vendor, number, subtotal, total.
  p1 erased document_number; the system answers INV-9 (another value); everything else right.
  p2 dropped the last page: subtotal and total erased. The system abstains on subtotal (the
     model alone said 9), returns total 10 (the erased value, computed or read elsewhere), gets the
     vendor wrong, the number right, and returns the labeled row plus one invented row.

  erased cases: p1.document_number, p2.subtotal, p2.total = 3
  system : absent 0, abstained 1, extracted 2  -> 2 of 3 (one recovered: total; one other: number)
  model  : extracted 3 of 3 (its own answer had a value in all three)
  kept fields right: p1 3/3 (vendor, subtotal, total); p2 1/2 (vendor wrong, number right)
  p2 rows: 1 labeled, 1 matched, 0 missed, 1 spurious
"""

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw

import evalkit.probes as probes
from docextract.extract.runner import DocRecord
from docextract.extract.schema import LineItem
from evalkit.groundtruth import UNLABELED, GroundTruth, GTLineItem
from evalkit.gt_store import write_label
from evalkit.probe_report import ABSTAINED, EXTRACTED, analyse, exact_interval, to_markdown
from evalkit.probes import (
    PROBES,
    ProbeError,
    ProbeSpec,
    build_probe,
    check_sources,
    erase_box,
    tree_hashes,
    write_probe_set,
)
from ingest_support import ROOT
from test_failures import abst, ext
from test_scorecard import document, run_dir

spec = importlib.util.spec_from_file_location(
    "probe_report_script", ROOT / "scripts" / "probe_report.py"
)
assert spec is not None and spec.loader is not None
report_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_script)

D = Decimal
SMALL = (
    ProbeSpec("t1", "cons_003", erase=("buyer_name",)),
    ProbeSpec("t2", "cons_003", drop_last_page=True),
    ProbeSpec("t3", "cons_003"),
)


@pytest.fixture(scope="module")
def built() -> dict[str, probes.BuiltProbe]:
    return {s.probe_id: build_probe(s) for s in SMALL}


# ---------------------------------------------------------------- painting


def text_image(background: tuple[int, int, int], ink: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (200, 60), background)
    ImageDraw.Draw(image).text((40, 20), "HELLO 123", fill=ink)
    return image


class TestEraseBox:
    def test_the_text_is_painted_out_and_the_rest_is_untouched(self) -> None:
        image = text_image((255, 255, 255), (0, 0, 0))
        before = image.copy()
        erase_box(image, (35, 15, 120, 40))
        assert image.crop((32, 12, 124, 44)).getextrema() == ((255, 255), (255, 255), (255, 255))
        assert image.crop((0, 0, 200, 10)).tobytes() == before.crop((0, 0, 200, 10)).tobytes()
        assert image.crop((130, 0, 200, 60)).tobytes() == before.crop((130, 0, 200, 60)).tobytes()

    def test_it_paints_with_the_colour_around_it_not_white(self) -> None:
        blue = (28, 78, 154)
        image = text_image(blue, (255, 255, 255))  # white text on a dark band, like the header
        erase_box(image, (35, 15, 120, 40))
        assert image.crop((36, 16, 118, 38)).getextrema() == ((28, 28), (78, 78), (154, 154))

    def test_a_box_a_little_tighter_than_the_text_still_removes_all_of_it(self) -> None:
        image = text_image((255, 255, 255), (0, 0, 0))
        left, top, right, bottom = ImageDraw.Draw(image).textbbox((40, 20), "HELLO 123")
        erase_box(image, (left + 2, top + 2, right - 2, bottom - 2))  # the padding covers the rest
        assert image.getextrema() == ((255, 255), (255, 255), (255, 255))

    def test_a_box_with_nothing_in_it_is_refused(self) -> None:
        image = Image.new("RGB", (200, 60), (255, 255, 255))
        with pytest.raises(ProbeError, match="nothing is printed"):
            erase_box(image, (35, 15, 120, 40))

    def test_a_box_at_the_edge_is_clipped_not_an_error(self) -> None:
        image = Image.new("RGB", (60, 30), (255, 255, 255))
        ImageDraw.Draw(image).text((2, 2), "EDGE", fill=(0, 0, 0))
        erase_box(image, (-5, -5, 40, 20))
        assert image.crop((0, 0, 44, 24)).getextrema() == ((255, 255), (255, 255), (255, 255))

    def test_points_become_pixels_with_the_origin_at_the_top(self) -> None:
        image = Image.new("RGB", (200, 400))
        # a 100 x 200 pt page drawn at 2 px per point: the box (10,20)-(30,40) pt (bottom-left
        # origin) is x 20..60 and, from the top, y = (200 - 40) * 2 .. (200 - 20) * 2
        assert probes._pixel_box((10, 20, 30, 40), (100, 200), image) == (20, 320, 60, 360)


# ---------------------------------------------------------------- building a probe


class TestBuild:
    def test_the_erased_field_is_null_and_everything_else_is_the_sources_label(
        self, built: dict[str, probes.BuiltProbe]
    ) -> None:
        label, source = built["t1"].label, built["t3"].label
        assert label["buyer_name"] is None and source["buyer_name"] is not None
        differing = {k for k in label if label[k] != source[k]}
        assert differing == {"doc_id", "notes", "buyer_name"}
        assert built["t1"].erased == ("buyer_name",)

    def test_a_dropped_last_page_nulls_the_totals_and_keeps_the_rows_of_the_pages_left(
        self, built: dict[str, probes.BuiltProbe]
    ) -> None:
        drop, whole = built["t2"], built["t3"]
        for name in ("subtotal", "discount", "tax", "total"):
            assert drop.label[name] is None, name
        assert drop.erased == ("subtotal", "discount", "tax", "total")  # compact prints a discount
        first_page_rows = len(drop.label["line_items"])
        assert 0 < first_page_rows < len(whole.label["line_items"])
        assert drop.label["line_items"] == whole.label["line_items"][:first_page_rows]
        assert drop.label["currency"] == UNLABELED  # "Rp" is only in the totals block
        assert len(drop.images) == whole.source_pages - 1 == drop.source_pages - 1

    def test_a_dropped_page_keeps_the_earlier_pages_pixel_for_pixel(
        self, built: dict[str, probes.BuiltProbe]
    ) -> None:
        drop, whole = built["t2"], built["t3"]
        for kept, original in zip(drop.images, whole.images, strict=False):
            assert kept.tobytes() == original.tobytes()

    def test_a_control_has_the_sources_label_and_nothing_erased(
        self, built: dict[str, probes.BuiltProbe]
    ) -> None:
        control = built["t3"]
        assert control.erased == () and control.erasures == [] and control.spec.is_control
        assert GroundTruth.model_validate(control.label).buyer_name is not None

    def test_the_erased_value_is_recorded_with_the_pages_it_was_found_on(
        self, built: dict[str, probes.BuiltProbe]
    ) -> None:
        (erasure,) = built["t1"].erasures
        assert erasure.field == "buyer_name" and erasure.pages == (0,)
        assert erasure.value == built["t3"].original["buyer_name"]

    def test_the_erased_page_differs_from_the_original_only_where_the_value_was(
        self, built: dict[str, probes.BuiltProbe]
    ) -> None:
        erased, whole = built["t1"].images[0], built["t3"].images[0]
        assert erased.size == whole.size
        box = ImageChops.difference(erased, whole).getbbox()
        assert box is not None
        assert (box[2] - box[0]) < 0.35 * erased.width and (box[3] - box[1]) < 0.02 * erased.height
        assert built["t1"].images[1].tobytes() == built["t3"].images[1].tobytes()

    def test_a_number_repeated_on_every_page_is_erased_on_every_page(self) -> None:
        classic = build_probe(ProbeSpec("t4", "cons_001", erase=("document_number",)))
        (erasure,) = classic.erasures
        assert erasure.pages == (0, 1)  # the header and the continuation header

    def test_a_field_the_layout_does_not_print_is_refused(self) -> None:
        with pytest.raises(ProbeError, match="does not print vendor_tax_id"):
            build_probe(ProbeSpec("t5", "cons_002", erase=("vendor_tax_id",)))

    def test_a_value_found_the_wrong_number_of_times_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(probes, "_find", lambda textpage, value: [])
        with pytest.raises(ProbeError, match="found 0 time"):
            build_probe(ProbeSpec("t6", "cons_003", erase=("buyer_name",)))
        box = (50.0, 700.0, 120.0, 712.0)
        monkeypatch.setattr(probes, "_find", lambda textpage, value: [box, box])
        with pytest.raises(ProbeError, match="found 2 time"):
            build_probe(ProbeSpec("t6", "cons_003", erase=("buyer_name",)))

    def test_text_that_is_not_where_the_search_says_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(probes, "_find", lambda textpage, value: [(500.0, 20.0, 560.0, 30.0)])
        with pytest.raises(ProbeError, match="nothing is printed"):
            build_probe(ProbeSpec("t7", "cons_003", erase=("buyer_name",)))

    def test_every_committed_probe_names_a_constructed_source(self) -> None:
        assert [s.probe_id for s in PROBES] == [
            *(f"p0{i}" for i in range(1, 10)),
            "c01",
            "c02",
            "c03",
        ]
        assert sum(1 for s in PROBES if s.is_control) == 3


# ---------------------------------------------------------------- writing the set


class TestWrite:
    def test_the_set_is_written_the_way_the_pipeline_reads_it(self, tmp_path: Path) -> None:
        document = write_probe_set(
            tmp_path / "p", specs=SMALL, dataset_pages=ROOT / "data" / "pages"
        )
        root = tmp_path / "p"
        from docextract.ingest.manifest import read_manifest
        from docextract.ingest.stored import stored_pages
        from evalkit.gt_store import read_label

        rows = read_manifest(root / "manifest.csv")
        assert [r.doc_id for r in rows] == ["t1", "t2", "t3"]
        assert {r.split for r in rows} == {"test"}
        assert len(stored_pages("t1", root / "pages")) == 2  # stored like any ingested document
        assert len(stored_pages("t2", root / "pages")) == 1
        assert read_label(root / "labels" / "t1.json").buyer_name is None
        entry = {e["id"]: e for e in document["probes"]}["t2"]
        assert entry["kind"] == "drop_last_page" and entry["source_pages"] == 2
        assert entry["original"]["total"] is not None
        assert json.loads((root / "probes.json").read_text(encoding="utf-8")) == document

    def test_a_second_generation_is_byte_identical(self, tmp_path: Path) -> None:
        write_probe_set(tmp_path / "a", specs=SMALL)
        write_probe_set(tmp_path / "b", specs=SMALL)
        assert tree_hashes(tmp_path / "a") == tree_hashes(tmp_path / "b")

    def test_a_control_must_equal_the_datasets_own_pages(self, tmp_path: Path) -> None:
        fake = tmp_path / "pages"
        (fake / "cons_003").mkdir(parents=True)
        (fake / "cons_003" / "ingest.json").write_text(
            json.dumps({"pages": [{"n": 1, "sha256": "0" * 64}, {"n": 2, "sha256": "1" * 64}]}),
            encoding="utf-8",
        )
        with pytest.raises(ProbeError, match="not byte-identical"):
            write_probe_set(tmp_path / "p", specs=(SMALL[2],), dataset_pages=fake)

    def test_the_pages_a_drop_keeps_must_equal_the_datasets_first_pages(
        self, tmp_path: Path
    ) -> None:
        fake = tmp_path / "pages"
        (fake / "cons_003").mkdir(parents=True)
        (fake / "cons_003" / "ingest.json").write_text(
            json.dumps({"pages": [{"n": 1, "sha256": "0" * 64}, {"n": 2, "sha256": "1" * 64}]}),
            encoding="utf-8",
        )
        with pytest.raises(ProbeError, match="not byte-identical"):
            write_probe_set(tmp_path / "p", specs=(SMALL[1],), dataset_pages=fake)

    def test_the_sources_must_equal_the_datasets_raw_pdfs(self, tmp_path: Path) -> None:
        from evalkit.construct import render_invoice

        raw = tmp_path / "raw"
        (raw / "constructed").mkdir(parents=True)
        assert check_sources(raw) == []  # no raw files: nothing to compare
        for source in ("cons_001", "cons_002", "cons_003"):
            render_invoice(probes.invoice_for(source), raw / "constructed" / f"{source}.pdf")
        assert check_sources(raw) == []
        path = raw / "constructed" / "cons_002.pdf"
        path.write_bytes(path.read_bytes() + b"\n%changed")
        (problem,) = check_sources(raw)
        assert problem.startswith("cons_002: the regenerated PDF differs")

    @pytest.mark.slow
    def test_the_committed_probe_set_is_what_a_regeneration_gives(self, tmp_path: Path) -> None:
        write_probe_set(tmp_path / "p", dataset_pages=ROOT / "data" / "pages")
        assert tree_hashes(tmp_path / "p") == tree_hashes(ROOT / "data" / "probes")


# ---------------------------------------------------------------- the interval


class TestInterval:
    def test_known_values(self) -> None:
        low, high = exact_interval(5, 20)
        assert (round(low, 3), round(high, 3)) == (0.087, 0.491)
        low, high = exact_interval(0, 19)
        assert low == 0.0 and round(high, 4) == round(1 - 0.025 ** (1 / 19), 4)
        low, high = exact_interval(19, 19)
        assert high == 1.0 and round(low, 4) == round(0.025 ** (1 / 19), 4)

    def test_no_trials_say_nothing(self) -> None:
        assert exact_interval(0, 0) == (0.0, 1.0)


# ---------------------------------------------------------------- the report


def tea(quantity: int = 1, description: str = "Tea") -> LineItem:
    return LineItem(
        description=description,
        quantity=D(quantity),
        unit_price=D(5),
        amount=D(5),
        status="extracted",
        confidence=0.9,
        evidence="q",
        page=1,
    )


def label_of(doc_id: str, **fields: object) -> GroundTruth:
    values: dict[str, object] = {
        "vendor_name": "Acme",
        "document_number": "INV-1",
        "subtotal": D(9),
        "total": D(10),
        "line_items": [GTLineItem(description="Tea", quantity=D(1), unit_price=D(5), amount=D(5))],
    }
    values.update(fields)
    return GroundTruth(doc_id=doc_id, **values)  # type: ignore[arg-type]


def entry(pid: str, kind: str, erased: list[str], original: dict[str, str]) -> dict[str, object]:
    return {
        "id": pid,
        "source": "s1",
        "layout": "classic",
        "kind": kind,
        "erased": erased,
        "erasures": [],
        "original": original,
        "source_pages": 2,
        "pages": [],
    }


def report_world(
    tmp_path: Path, *, run_p2: bool = True, control_number: str = "INV-1"
) -> tuple[Path, Path]:
    pdir = tmp_path / "probes"
    for label in (
        label_of("c1"),
        label_of("p1", document_number=None),
        label_of("p2", subtotal=None, total=None),
    ):
        write_label(pdir / "labels", label)
    (pdir / "probes.json").write_text(
        json.dumps(
            {
                "seed": "test-seed",
                "probes": [
                    entry("c1", "control", [], {}),
                    entry("p1", "erase", ["document_number"], {"document_number": "INV-1"}),
                    entry(
                        "p2",
                        "drop_last_page",
                        ["subtotal", "total"],
                        {"subtotal": "9", "total": "10"},
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    c1 = document(
        vendor_name=ext("Acme"),
        document_number=ext(control_number),
        subtotal=ext(D(9)),
        total=ext(D(10)),
        line_items=[tea()],
    )
    p1 = document(
        vendor_name=ext("Acme"),
        document_number=ext("INV-9"),
        subtotal=ext(D(9)),
        total=ext(D(10)),
        line_items=[tea()],
    )
    p2_raw = document(
        vendor_name=ext("Other"),
        document_number=ext("INV-1"),
        subtotal=ext(D(9)),
        total=ext(D(10)),
        line_items=[tea(), tea(2, "Ghost")],
    )
    p2 = p2_raw.model_copy(update={"subtotal": abst(0.5, "evidence_not_grounded")})
    records = [
        DocRecord(doc_id="c1", status="ok", parsed=c1, validated=c1),
        DocRecord(doc_id="p1", status="ok", parsed=p1, validated=p1),
    ]
    if run_p2:
        records.append(DocRecord(doc_id="p2", status="ok", parsed=p2_raw, validated=p2))
    else:
        records.append(DocRecord(doc_id="p2", status="failed", error="boom"))
    return run_dir(tmp_path, records, name="r1"), pdir


class TestReport:
    def test_the_counts_are_the_hand_worked_ones(self, tmp_path: Path) -> None:
        run, pdir = report_world(tmp_path)
        report = analyse(run, pdir)
        assert [(c.probe, c.field) for c in report.cases] == [
            ("p1", "document_number"),
            ("p2", "subtotal"),
            ("p2", "total"),
        ]
        assert [c.returned for c in report.cases] == [EXTRACTED, ABSTAINED, EXTRACTED]
        assert [c.raw for c in report.cases] == [EXTRACTED] * 3
        assert [c.recovered for c in report.cases] == [False, None, True]
        assert [c.value for c in report.cases] == ["INV-9", "", "10"]
        assert [c.control_read for c in report.cases] == [True, True, True]

    def test_the_fields_that_were_not_erased_and_the_rows(self, tmp_path: Path) -> None:
        report = analyse(*report_world(tmp_path))
        p1, p2 = report.lines
        assert (p1.kept_correct, p1.kept_labeled) == (3, 3)
        assert (p2.kept_correct, p2.kept_labeled) == (1, 2)
        assert (p2.rows_labeled, p2.rows_matched, p2.rows_missed, p2.rows_spurious) == (1, 1, 0, 1)
        assert p2.returned == {"absent": 0, "abstained": 1, "extracted": 1}

    def test_a_control_that_misread_the_field_says_so(self, tmp_path: Path) -> None:
        report = analyse(*report_world(tmp_path, control_number="INV-7"))
        assert [c.control_read for c in report.cases] == [False, True, True]
        (control,) = report.controls
        assert (control.correct, control.labeled) == (3, 4)
        assert (
            "| p1 | document_number | INV-1 | **extracted** | extracted | INV-9 (another value) "
            "| no |" in to_markdown(report)
        )

    def test_the_control_line(self, tmp_path: Path) -> None:
        (control,) = analyse(*report_world(tmp_path)).controls
        assert (control.probe, control.correct, control.labeled) == ("c1", 4, 4)

    def test_a_probe_that_did_not_run_is_left_out_and_said(self, tmp_path: Path) -> None:
        report = analyse(*report_world(tmp_path, run_p2=False))
        assert report.not_ok == ("p2",)
        assert [c.probe for c in report.cases] == ["p1"]
        assert "Not run or failed, so not counted: p2." in to_markdown(report)

    def test_the_markdown_headline(self, tmp_path: Path) -> None:
        text = to_markdown(analyse(*report_world(tmp_path)), notes="  My reading.  ")
        assert "| **the system** (validated) | 3 | 0 | 1 | 2 | 2/3 = 66.7% (95% CI" in text
        assert (
            "| the model alone (before the validators) | 3 | 0 | 0 | 3 | 3/3 = 100.0% (95% CI"
            in text
        )
        assert "1 are the value that was erased" in text and "1 are another value" in text
        assert (
            "| p2 | total | 10 | **extracted** | extracted | 10 (the erased value) | yes |" in text
        )
        assert "| p2 | subtotal | 9 | abstained | extracted |  | yes |" in text
        assert "1/2 | 1/1 matched, 1 spurious" in text
        assert "| c1 | s1 | ok | 4/4 |" in text
        assert text.rstrip().endswith("## Reading\n\nMy reading.")

    def test_a_system_that_answers_no_erased_field_says_so(self, tmp_path: Path) -> None:
        import dataclasses

        report = analyse(*report_world(tmp_path))
        clean = dataclasses.replace(
            report,
            cases=tuple(
                dataclasses.replace(c, returned=ABSTAINED, value="", recovered=None)
                for c in report.cases
            ),
        )
        text = to_markdown(clean)
        assert "The system returned no value for any erased field." in text
        assert "values the system returned" not in text
        assert "values the system returned" in to_markdown(report)

    def test_an_unreadable_run_is_an_error(self, tmp_path: Path) -> None:
        from evalkit.probe_report import ProbeReportError

        with pytest.raises(ProbeReportError, match="cannot be read"):
            analyse(tmp_path / "nope", tmp_path)

    def test_the_script(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run, pdir = report_world(tmp_path)
        out = tmp_path / "out" / "r.md"
        code = report_script.main([str(run), "--probes-dir", str(pdir), "--out", str(out)])
        assert code == 0 and "3 erased field(s)" in capsys.readouterr().out
        assert out.read_text(encoding="utf-8").startswith("# Hallucination probe: `r1`")
        assert report_script.main([str(tmp_path / "missing")]) == 2
        assert report_script.main([str(run), "--probes-dir", str(tmp_path / "none")]) == 2

    def test_the_report_is_reproducible(self, tmp_path: Path) -> None:
        a = to_markdown(analyse(*report_world(tmp_path / "a")))
        b = to_markdown(analyse(*report_world(tmp_path / "b")))
        assert a == b
