import base64
import json
import math
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from docextract.ingest.manifest import ManifestRow
from docextract.ingest.pipeline import ingest_document
from docextract.ingest.redact import apply_boxes, validate_boxes
from docextract.ingest.redactions import parse_redactions
from docextract.paths import DatasetPaths
from evalkit.groundtruth import UNLABELED, GroundTruth
from evalkit.gt_store import write_label
from evalkit.labeling import BlindDraftError, LabelForm, LabelingError
from evalkit.labeling_ui import (
    GEOMETRY_JS,
    PagesUnavailable,
    clear_wip,
    labelable_rows,
    load_form,
    overview,
    page_data_url,
    page_files,
    save_wip,
    viewer_height,
    viewer_html,
    wip_dir,
)
from ingest_support import settings

DRAFT: dict[str, Any] = {
    "doc_id": "doc_001",
    "label_mode": "prelabel",
    "notes": "pre-label draft by m; unverified",
    "vendor_name": "PT Contoh",
    "due_date": None,
    "payment_method": UNLABELED,
    "field_notes": {"payment_method": "the model did not answer this field"},
}


def row(doc_id: str = "doc_001", **overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "doc_id": doc_id,
        "raw_files": f"{doc_id}.png",
        "content_group": doc_id,
        "doc_type": "faktur",
        "label_mode": "prelabel",
        "needs_redaction": "n",
    }
    values.update(overrides)
    return ManifestRow.model_validate(values)


@pytest.fixture
def paths(tmp_path: Path) -> DatasetPaths:
    return DatasetPaths(tmp_path)


def finished_label(doc_id: str = "doc_001", mode: str = "prelabel") -> GroundTruth:
    form = LabelForm.start(row(doc_id, label_mode=mode))
    for name in form.fields:
        form.mark_absent(name)
    form.mark_rows_absent()
    return form.build_label()


# ------------------------------------------------------------------ which documents


class TestOverview:
    def rows(self) -> list[ManifestRow]:
        return [
            row("doc_001"),
            row("doc_002", label_mode="blind"),
            row("doc_003", gt_from="doc_001", content_group="doc_001", label_mode=None),
            row("cons_001", capture="constructed", label_mode=None),
            row("doc_004", label_mode=None),
        ]

    def test_only_documents_a_person_labels_are_listed(self) -> None:
        assert [r.doc_id for r in labelable_rows(self.rows())] == ["doc_001", "doc_002"]

    def test_every_state(self, paths: DatasetPaths) -> None:
        rows = [row("a"), row("b"), row("c"), row("d", label_mode="blind")]
        write_label(paths.ground_truth_dir, finished_label("a"))
        form = LabelForm.start(row("b"))
        form.mark_absent("total")
        save_wip(form, paths)
        write_label(paths.drafts_dir, GroundTruth.model_validate({**DRAFT, "doc_id": "c"}))
        states = {item.doc_id: item.state for item in overview(rows, paths)}
        assert states == {"a": "labeled", "b": "in_progress", "c": "drafted", "d": "todo"}

    def test_a_label_wins_over_saved_work_and_a_draft(self, paths: DatasetPaths) -> None:
        write_label(paths.ground_truth_dir, finished_label("a"))
        write_label(paths.drafts_dir, GroundTruth.model_validate({**DRAFT, "doc_id": "a"}))
        save_wip(LabelForm.start(row("a")), paths)
        assert overview([row("a")], paths)[0].state == "labeled"

    def test_a_draft_does_not_make_a_blind_document_drafted(self, paths: DatasetPaths) -> None:
        write_label(paths.drafts_dir, GroundTruth.model_validate({**DRAFT, "doc_id": "b"}))
        assert overview([row("b", label_mode="blind")], paths)[0].state == "todo"

    def test_the_overview_carries_what_the_picker_shows(self, paths: DatasetPaths) -> None:
        (item,) = overview([row("a", split="test", doc_type="invoice")], paths)
        assert (item.label_mode, item.split, item.doc_type) == ("prelabel", "test", "invoice")


# ------------------------------------------------------------------ opening a document


class TestLoadForm:
    def test_an_empty_form_when_there_is_nothing(self, paths: DatasetPaths) -> None:
        opened = load_form(row(), paths)
        assert opened.source == "empty" and opened.warning == ""
        assert opened.form.progress() == (0, 14)

    def test_a_draft_prefills_a_prelabeled_document(self, paths: DatasetPaths) -> None:
        write_label(paths.drafts_dir, GroundTruth.model_validate(DRAFT))
        opened = load_form(row(), paths)
        assert opened.source == "draft"
        assert opened.form.fields["vendor_name"].text == "PT Contoh"
        assert (
            opened.form.pending() == opened.form.pending()[:] and len(opened.form.pending()) == 14
        )

    def test_a_blind_document_with_a_draft_is_refused(self, paths: DatasetPaths) -> None:
        write_label(paths.drafts_dir, GroundTruth.model_validate({**DRAFT, "doc_id": "doc_002"}))
        with pytest.raises(BlindDraftError, match="BLIND document but has a model draft"):
            load_form(row("doc_002", label_mode="blind"), paths)

    def test_the_draft_of_one_document_is_not_used_for_another(self, paths: DatasetPaths) -> None:
        write_label(paths.drafts_dir, GroundTruth.model_validate({**DRAFT, "doc_id": "other"}))
        assert load_form(row("doc_001"), paths).source == "empty"

    def test_a_finished_label_reopens_decided(self, paths: DatasetPaths) -> None:
        write_label(paths.ground_truth_dir, finished_label())
        opened = load_form(row(), paths)
        assert opened.source == "finished label" and opened.form.can_save

    def test_a_finished_label_wins_over_the_draft(self, paths: DatasetPaths) -> None:
        write_label(paths.ground_truth_dir, finished_label())
        write_label(paths.drafts_dir, GroundTruth.model_validate(DRAFT))
        assert load_form(row(), paths).source == "finished label"

    def test_saved_work_wins_over_everything(self, paths: DatasetPaths) -> None:
        write_label(paths.ground_truth_dir, finished_label())
        write_label(paths.drafts_dir, GroundTruth.model_validate(DRAFT))
        form = LabelForm.start(row())
        form.mark_unlabeled("tax")
        save_wip(form, paths)
        opened = load_form(row(), paths)
        assert opened.source == "saved work" and opened.form.fields["tax"].status == "unlabeled"

    def test_saved_work_that_cannot_be_read_is_ignored_with_a_warning(
        self, paths: DatasetPaths
    ) -> None:
        wip_dir(paths).mkdir(parents=True)
        (wip_dir(paths) / "doc_001.json").write_text("{nope", encoding="utf-8")
        write_label(paths.drafts_dir, GroundTruth.model_validate(DRAFT))
        opened = load_form(row(), paths)
        assert opened.source == "draft" and "saved work could not be used" in opened.warning

    def test_saved_work_for_another_mode_is_ignored(self, paths: DatasetPaths) -> None:
        save_wip(LabelForm.start(row(label_mode="blind")), paths)  # saved while it was blind
        opened = load_form(row(label_mode="prelabel"), paths)
        assert opened.source == "empty" and "different label_mode" in opened.warning

    def test_saved_work_for_another_document_is_ignored(self, paths: DatasetPaths) -> None:
        save_wip(LabelForm.start(row("doc_002")), paths)
        (wip_dir(paths) / "doc_001.json").write_text(
            (wip_dir(paths) / "doc_002.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        assert load_form(row("doc_001"), paths).source == "empty"

    def test_an_unreadable_draft_does_not_stop_labeling(self, paths: DatasetPaths) -> None:
        paths.drafts_dir.mkdir(parents=True)
        (paths.drafts_dir / "doc_001.json").write_text("{", encoding="utf-8")
        opened = load_form(row(), paths)
        assert opened.source == "empty" and "the draft could not be read" in opened.warning

    @pytest.mark.parametrize(
        "overrides", [{"gt_from": "doc_009"}, {"capture": "constructed"}, {"label_mode": None}]
    )
    def test_documents_a_person_does_not_label_are_refused(
        self, paths: DatasetPaths, overrides: dict[str, Any]
    ) -> None:
        with pytest.raises(LabelingError):
            load_form(row(**overrides), paths)


class TestSavedWork:
    def test_it_is_written_atomically_and_read_back(self, paths: DatasetPaths) -> None:
        form = LabelForm.start(row())
        form.set_text("vendor_name", "A")
        form.verify("vendor_name")
        target = save_wip(form, paths)
        assert target == wip_dir(paths) / "doc_001.json"
        assert [p.name for p in wip_dir(paths).iterdir()] == ["doc_001.json"]
        assert LabelForm.from_json(json.loads(target.read_text(encoding="utf-8"))) == form

    def test_it_lives_under_outputs_which_is_not_committed(self, paths: DatasetPaths) -> None:
        assert wip_dir(paths) == paths.root / "outputs" / "labeling"
        gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
        assert "outputs/*" in gitignore

    def test_clearing_it(self, paths: DatasetPaths) -> None:
        save_wip(LabelForm.start(row()), paths)
        clear_wip("doc_001", paths)
        assert not (wip_dir(paths) / "doc_001.json").exists()
        clear_wip("doc_001", paths)  # already gone: fine


# ------------------------------------------------------------------ pages


class TestPages:
    def ingest(
        self, paths: DatasetPaths, pages: int = 1, size: tuple[int, int] = (120, 90)
    ) -> None:
        paths.raw_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (200, 200, 200)).save(paths.raw_dir / "doc_001.png")
        ingest_document(
            row(),
            raw_dir=paths.raw_dir,
            out_dir=paths.pages_dir / "doc_001",
            settings=settings(),
        )

    def test_the_stored_pages_are_the_ingested_ones(self, paths: DatasetPaths) -> None:
        self.ingest(paths)
        assert [p.name for p in page_files("doc_001", paths)] == ["p1.png"]

    def test_a_document_that_was_not_ingested_has_no_pages(self, paths: DatasetPaths) -> None:
        with pytest.raises(PagesUnavailable, match="not ingested yet"):
            page_files("doc_001", paths)

    def test_a_page_that_changed_since_ingestion_is_not_shown(self, paths: DatasetPaths) -> None:
        self.ingest(paths)
        page = paths.pages_dir / "doc_001" / "p1.png"
        page.write_bytes(page.read_bytes() + b"x")
        with pytest.raises(PagesUnavailable, match="no longer matches"):
            page_files("doc_001", paths)

    def test_the_preview_pages_are_listed_in_page_order(self, paths: DatasetPaths) -> None:
        folder = paths.preview_dir / "doc_001"
        folder.mkdir(parents=True)
        for number in (10, 2, 1):
            Image.new("RGB", (10, 10)).save(folder / f"p{number}.png")
        (folder / "notes.txt").write_text("x", encoding="utf-8")
        assert [p.name for p in page_files("doc_001", paths, preview=True)] == [
            "p1.png",
            "p2.png",
            "p10.png",
        ]

    def test_no_preview_says_how_to_make_one(self, paths: DatasetPaths) -> None:
        with pytest.raises(PagesUnavailable, match="--preview --doc doc_001"):
            page_files("doc_001", paths, preview=True)

    def test_the_preview_is_never_the_default(self, paths: DatasetPaths) -> None:
        folder = paths.preview_dir / "doc_001"
        folder.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(folder / "p1.png")
        with pytest.raises(PagesUnavailable, match="not ingested yet"):
            page_files("doc_001", paths)  # unredacted pages must be asked for

    @pytest.mark.parametrize(("fmt", "mime"), [("PNG", "image/png"), ("JPEG", "image/jpeg")])
    def test_a_page_is_shown_exactly_as_stored(self, tmp_path: Path, fmt: str, mime: str) -> None:
        path = tmp_path / ("p1.png" if fmt == "PNG" else "p1.jpg")
        Image.new("RGB", (37, 23), (10, 20, 30)).save(path, fmt)
        url, size = page_data_url(path)
        assert size == (37, 23) and url.startswith(f"data:{mime};base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == path.read_bytes()


# ------------------------------------------------------------------ the viewer


def make_html(**overrides: Any) -> str:
    args: dict[str, Any] = {
        "doc_id": "doc_007",
        "page": 2,
        "natural_size": (1654, 2339),
        "width_px": 900,
    }
    args.update(overrides)
    return viewer_html("data:image/png;base64,AAAA", **args)


class TestViewerHtml:
    def test_it_is_self_contained_and_carries_the_page_facts(self) -> None:
        html = make_html()
        assert 'src="data:image/png;base64,AAAA"' in html
        assert "const NATURAL_W = 1654, NATURAL_H = 2339" in html
        assert 'DOC = "doc_007", PAGE = 2' in html
        assert "width: 900px" in html
        assert "__" not in html.replace("__proto__", "")  # every placeholder was filled
        assert "http://" not in html and "https://" not in html  # nothing is fetched

    def test_the_geometry_is_the_shared_block(self) -> None:
        assert GEOMETRY_JS.strip() in make_html()

    def test_the_doc_id_cannot_break_out_of_the_script(self) -> None:
        html = make_html(doc_id='x";alert(1);//')
        assert 'DOC = "x\\";alert(1);//"' in html

    def test_an_unredacted_preview_is_marked(self) -> None:
        assert "UNREDACTED preview" in make_html(unredacted=True)
        assert "UNREDACTED" not in make_html(unredacted=False)

    def test_the_height_fits_the_page_at_the_chosen_width(self) -> None:
        assert viewer_height((1000, 2000), 500) == 1000 + 90
        assert viewer_height((1654, 2339), 900) == round(900 * 2339 / 1654) + 90


# ------------------------------------------------------------------ the geometry, run under Node

NODE = shutil.which("node")


def run_js(body: str, data: Any = None) -> Any:
    """Run `body` in Node after the viewer's geometry block, with `DATA` bound to `data`. The
    script goes in through stdin (a long one does not fit on a Windows command line)."""
    script = f"{GEOMETRY_JS}\nconst DATA = {json.dumps(data)};\n{body}"
    done = subprocess.run(
        [NODE or "node", "-"],
        input=script,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def py_pixel(
    client: tuple[float, float], rect: dict[str, float], size: tuple[int, int]
) -> tuple[int, int]:
    """The Python reference of the viewer's pixel rule (what the JS must agree with)."""
    x = math.floor((client[0] - rect["left"]) * size[0] / rect["width"])
    y = math.floor((client[1] - rect["top"]) * size[1] / rect["height"])
    return min(max(x, 0), size[0] - 1), min(max(y, 0), size[1] - 1)


PIXELS_JS = """
console.log(JSON.stringify(DATA.map(c => {
  const p = pagePixel(c.client[0], c.client[1], c.rect, c.size[0], c.size[1]);
  return [p.x, p.y];
})));
"""
BOXES_JS = """
console.log(JSON.stringify(DATA.map(p => boxFromDrag({x: p[0][0], y: p[0][1]}, {x: p[1][0], y: p[1][1]}))));
"""  # noqa: E501


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
class TestGeometryInNode:
    def test_the_pixel_under_the_cursor_agrees_with_the_python_reference(self) -> None:
        rng = random.Random(5)
        cases = []
        for _ in range(300):
            size = (rng.randint(50, 3000), rng.randint(50, 3000))
            width = rng.uniform(200, 1800)
            rect = {
                "left": rng.uniform(0, 300),
                "top": rng.uniform(-500, 300),
                "width": width,
                "height": width * size[1] / size[0],
            }
            client = (rng.uniform(-50, 2200), rng.uniform(-600, 2600))  # some outside the image
            cases.append({"client": client, "rect": rect, "size": size})
        expected = [list(py_pixel(c["client"], c["rect"], c["size"])) for c in cases]
        assert run_js(PIXELS_JS, cases) == expected

    def test_the_corners_of_the_displayed_image_are_the_first_and_last_pixels(self) -> None:
        rect = {"left": 10, "top": 20, "width": 827, "height": 1169.5}
        cases = [
            {"client": [10, 20], "rect": rect, "size": [1654, 2339]},
            {"client": [10 + 827 - 0.001, 20 + 1169.5 - 0.001], "rect": rect, "size": [1654, 2339]},
            {"client": [10 + 5000, 20 + 5000], "rect": rect, "size": [1654, 2339]},  # far outside
            {"client": [-100, -100], "rect": rect, "size": [1654, 2339]},
        ]
        assert run_js(PIXELS_JS, cases) == [[0, 0], [1653, 2338], [1653, 2338], [0, 0]]

    def test_a_dragged_box_covers_both_end_pixels_in_any_direction(self) -> None:
        pairs = [
            [[10, 20], [10, 20]],
            [[10, 20], [14, 25]],
            [[14, 25], [10, 20]],
            [[14, 20], [10, 25]],
        ]
        assert run_js(BOXES_JS, pairs) == [
            [10, 20, 1, 1],
            [10, 20, 5, 6],
            [10, 20, 5, 6],
            [10, 20, 5, 6],
        ]

    def test_a_dragged_box_blacks_out_exactly_the_dragged_pixels(self) -> None:
        # the whole chain: drag in the viewer -> box -> the redaction code
        size = (60, 40)
        rect = {"left": 0, "top": 0, "width": 120, "height": 80}  # displayed at 2x
        drag = [(30.5, 20.5), (59.9, 41.2)]  # client positions
        corners = [{"client": list(c), "rect": rect, "size": list(size)} for c in drag]
        a, b = run_js(PIXELS_JS, corners)
        (box,) = run_js(BOXES_JS, [[a, b]])
        ((x, y, w, h),) = validate_boxes([box], size, where="viewer")
        painted = apply_boxes(Image.new("RGB", size, "white"), [(x, y, w, h)])
        black = {
            (px, py)
            for px in range(size[0])
            for py in range(size[1])
            if painted.getpixel((px, py)) == (0, 0, 0)
        }
        first, last = py_pixel(drag[0], rect, size), py_pixel(drag[1], rect, size)
        wanted = {
            (px, py)
            for px in range(min(first[0], last[0]), max(first[0], last[0]) + 1)
            for py in range(min(first[1], last[1]), max(first[1], last[1]) + 1)
        }
        assert black == wanted and len(black) == w * h

    def test_the_yaml_snippet_is_what_redactions_yaml_takes(self) -> None:
        snippet = run_js("console.log(JSON.stringify(yamlSnippet(DATA[0], DATA[1], DATA[2])));",
                         ["doc_007", 2, [120, 340, 400, 60]])  # fmt: skip
        assert snippet == "doc_007:\n  2:\n    - [120, 340, 400, 60]"
        parsed = parse_redactions(snippet)
        assert parsed["doc_007"][2].boxes == ((120, 340, 400, 60),)


DOM_STUB = """
const listeners = {};
const copied = [];
const RECT = DATA.rect;
function element(id) {
  return {
    id, style: {}, textContent: "", innerHTML: "",
    addEventListener(type, handler) { listeners[id + ":" + type] = handler; },
    getBoundingClientRect() { return RECT; },
  };
}
const elements = {};
for (const id of ["page", "sel", "pos", "box", "yaml", "copy"]) elements[id] = element(id);
const document = { getElementById: (id) => elements[id] };
const window = { addEventListener(type, handler) { listeners["window:" + type] = handler; } };
const navigator = {
  clipboard: { writeText(text) { copied.push(text); return Promise.resolve(); } },
};
function mouse(x, y) { return { clientX: x, clientY: y, preventDefault() {} }; }
"""


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
class TestViewerScriptInNode:
    """The viewer's own script, run for real against a minimal DOM: no browser needed to know
    that its handlers are wired and that a drag ends up as a box and a YAML snippet."""

    def run(self, actions: str, *, width: int = 120, natural: tuple[int, int] = (60, 40)) -> Any:
        html = make_html(natural_size=natural, width_px=width)
        match = re.search(r"<script>(.*)</script>", html, re.S)
        assert match is not None
        height = round(width * natural[1] / natural[0])
        rect = {"left": 0, "top": 0, "width": width, "height": height}
        body = (
            DOM_STUB
            + match.group(1)
            + actions
            + "console.log(JSON.stringify({pos: elements.pos.innerHTML,"
            " box: elements.box.textContent,"
            " yaml: elements.yaml.textContent, yamlShown: elements.yaml.style.display,"
            " sel: elements.sel.style, copied}));"
        )
        done = subprocess.run(
            [NODE or "node", "-"],
            input=f"const DATA = {json.dumps({'rect': rect})};\n{body}",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)

    def test_the_script_loads_and_registers_its_handlers(self) -> None:
        out = self.run("")
        assert out["yaml"] == "" and out["box"] == "" and out["sel"] == {}  # nothing drawn yet

    def test_hovering_reads_the_pixel_in_page_coordinates(self) -> None:
        out = self.run('listeners["page:mousemove"](mouse(60.4, 40.9));')
        assert out["pos"] == "x <b>30</b> y <b>20</b> (page pixels)"  # displayed at 2x

    def test_a_drag_draws_the_box_and_prints_the_yaml(self) -> None:
        out = self.run(
            'listeners["page:mousedown"](mouse(30.5, 20.5));'
            'listeners["page:mousemove"](mouse(59.9, 41.2));'
            'listeners["window:mouseup"]();'
        )
        assert out["box"] == "[15, 10, 15, 11]"
        assert out["yamlShown"] == "block"
        assert out["yaml"] == "doc_007:\n  2:\n    - [15, 10, 15, 11]"
        assert out["sel"] == {"display": "block", "left": "30px", "top": "20px",
                              "width": "30px", "height": "22px"}  # fmt: skip

    def test_dragging_backwards_gives_the_same_box(self) -> None:
        forwards = self.run(
            'listeners["page:mousedown"](mouse(10, 10));listeners["page:mousemove"](mouse(50, 30));'
            'listeners["window:mouseup"]();'
        )
        backwards = self.run(
            'listeners["page:mousedown"](mouse(50, 30));listeners["page:mousemove"](mouse(10, 10));'
            'listeners["window:mouseup"]();'
        )
        assert forwards["box"] == backwards["box"]

    def test_moving_without_pressing_draws_nothing(self) -> None:
        out = self.run('listeners["page:mousemove"](mouse(60, 40));')
        assert out["sel"] == {} and out["box"] == "" and out["yaml"] == ""

    def test_a_mouse_up_with_no_drag_does_nothing(self) -> None:
        out = self.run('listeners["window:mouseup"]();')
        assert out["yaml"] == "" and out["sel"] == {}

    def test_the_box_never_leaves_the_page(self) -> None:
        out = self.run(
            'listeners["page:mousedown"](mouse(100, 70));'
            'listeners["page:mousemove"](mouse(5000, 5000));'
            'listeners["window:mouseup"]();'
        )
        assert out["box"] == "[50, 35, 10, 5]"  # clamped to the 60 x 40 page

    def test_copy_sends_the_snippet_to_the_clipboard(self) -> None:
        out = self.run(
            'listeners["page:mousedown"](mouse(0, 0));listeners["page:mousemove"](mouse(11, 5));'
            'listeners["window:mouseup"]();listeners["copy:click"]();'
        )
        assert out["copied"] == ["doc_007:\n  2:\n    - [0, 0, 6, 3]"]

    def test_copy_before_any_box_does_nothing(self) -> None:
        assert self.run('listeners["copy:click"]();')["copied"] == []

    def test_the_selection_follows_the_zoom(self) -> None:
        # the same drag at another display width selects the same page pixels
        small = self.run(
            'listeners["page:mousedown"](mouse(15, 10));listeners["page:mousemove"](mouse(30, 20));'
            'listeners["window:mouseup"]();',
            width=60,
        )
        big = self.run(
            'listeners["page:mousedown"](mouse(30, 20));listeners["page:mousemove"](mouse(60, 40));'
            'listeners["window:mouseup"]();',
            width=120,
        )
        assert small["box"] == big["box"]
