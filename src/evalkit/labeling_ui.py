"""Everything the labeling app needs that is not a widget: which documents there are and where each
stands, how a form is (re)opened, where unfinished work is kept, which page images to show, and the
page viewer. Plain Python, so it is tested without a browser (tools/label_app.py is a thin layer).

Opening a document, in order of precedence:
  1. work in progress (outputs/labeling/<doc_id>.json, gitignored) - never lost on a refresh;
  2. the finished label, reopened with every field decided;
  3. the model's draft (label_mode=prelabel only), every field still pending;
  4. an empty form.
A BLIND document skips 3 and refuses outright if a draft exists for it: that is an integrity
problem to fix, not something to hide.

The viewer shows a stored page at a chosen width and reads coordinates off it in STORED-PAGE
PIXELS (the space redactions.yaml uses): hovering shows the pixel under the cursor and dragging
draws a box, shown as `[x, y, w, h]` with a ready-to-paste redactions.yaml snippet. The geometry
is a few lines of JavaScript (`GEOMETRY_JS`) that tests run with Node against a Python reference.
"""

import base64
import json
import re
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from PIL import Image

from docextract.ingest.manifest import ManifestRow
from docextract.paths import DatasetPaths
from evalkit.groundtruth import GroundTruth
from evalkit.gt_store import GroundTruthError, label_path, read_label
from evalkit.labeling import BlindDraftError, LabelForm, LabelingError, labelable_mode
from evalkit.prelabel import PrelabelRefused, stored_pages

State = Literal["labeled", "in_progress", "drafted", "todo"]
T = TypeVar("T")
_PAGE_FILE = re.compile(r"p(\d+)\.(png|jpg)")


def wip_dir(paths: DatasetPaths) -> Path:
    return paths.root / "outputs" / "labeling"


# ---------------------------------------------------------------- which documents


@dataclass(frozen=True)
class DocOverview:
    doc_id: str
    label_mode: str
    split: str | None
    doc_type: str | None
    state: State


def labelable_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    """The documents a person labels by hand (not borrowers, not constructed, with a mode)."""
    out = []
    for row in rows:
        try:
            labelable_mode(row)
        except LabelingError:
            continue
        out.append(row)
    return out


def overview(rows: list[ManifestRow], paths: DatasetPaths) -> list[DocOverview]:
    result = []
    for row in labelable_rows(rows):
        if label_path(paths.ground_truth_dir, row.doc_id).exists():
            state: State = "labeled"
        elif label_path(wip_dir(paths), row.doc_id).exists():
            state = "in_progress"
        elif row.label_mode == "prelabel" and label_path(paths.drafts_dir, row.doc_id).exists():
            state = "drafted"
        else:
            state = "todo"
        result.append(DocOverview(row.doc_id, str(row.label_mode), row.split, row.doc_type, state))
    return result


# ---------------------------------------------------------------- opening and keeping work


@dataclass(frozen=True)
class Opened:
    form: LabelForm
    source: Literal["saved work", "finished label", "draft", "empty"]
    warning: str = ""  # something the person should know (e.g. saved work that could not be read)
    draft_model_note: str = ""


def load_form(row: ManifestRow, paths: DatasetPaths) -> Opened:
    mode = labelable_mode(row)
    draft_file = label_path(paths.drafts_dir, row.doc_id)
    if mode == "blind" and draft_file.exists():
        raise BlindDraftError(
            f"{row.doc_id} is a BLIND document but has a model draft ({draft_file}): "
            "delete it before labeling this document"
        )
    warning = ""
    work = label_path(wip_dir(paths), row.doc_id)
    if work.is_file():
        try:
            form = LabelForm.from_json(json.loads(work.read_text(encoding="utf-8")))
            if form.doc_id == row.doc_id and form.label_mode == mode:
                return Opened(form, "saved work")
            warning = "saved work belongs to a different label_mode and was ignored"
        except (OSError, ValueError, LabelingError) as exc:
            warning = f"saved work could not be used and was ignored ({exc})"
    finished = label_path(paths.ground_truth_dir, row.doc_id)
    if finished.is_file():
        return Opened(LabelForm.from_label(row, read_label(finished)), "finished label", warning)
    if draft_file.is_file():  # a blind document with a draft was refused above
        try:
            draft: GroundTruth = read_label(draft_file)
        except GroundTruthError as exc:
            return Opened(LabelForm.start(row), "empty", f"the draft could not be read ({exc})")
        return Opened(LabelForm.start(row, draft), "draft", warning)
    return Opened(LabelForm.start(row), "empty", warning)


def save_wip(form: LabelForm, paths: DatasetPaths) -> Path:
    target = label_path(wip_dir(paths), form.doc_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(form.to_json(), indent=1), encoding="utf-8")
    temporary.replace(target)
    return target


def clear_wip(doc_id: str, paths: DatasetPaths) -> None:
    label_path(wip_dir(paths), doc_id).unlink(missing_ok=True)


def frozen_input(
    state: MutableMapping[str, Any], name: str, version: int, make: Callable[[], T]
) -> T:
    """What a widget is given as its input, held still until `version` changes.

    `st.data_editor` keeps the user's edits as deltas against the data it was first given. If that
    input is rebuilt from the edited result on every rerun (as the line-item grid once did), the
    deltas are applied to a different table and the first edit is lost. So the input is built once
    per version and handed back unchanged on every rerun; bumping the version (verify, paste, load)
    is the only way to show something new.
    """
    held = state.get(name)
    if held is None or held[0] != version:
        held = (version, make())
        state[name] = held
    return held[1]  # type: ignore[no-any-return]


# ---------------------------------------------------------------- pages


class PagesUnavailable(Exception):
    """The pages cannot be shown (not ingested, changed, or no preview)."""


def page_files(doc_id: str, paths: DatasetPaths, *, preview: bool = False) -> list[Path]:
    """The stored (redacted) pages, verified against ingest.json; or, for finding redaction
    coordinates, the UNREDACTED preview pages (never labeled from)."""
    if not preview:
        try:
            return stored_pages(doc_id, paths.pages_dir)
        except PrelabelRefused as exc:
            raise PagesUnavailable(str(exc)) from exc
    folder = paths.preview_dir / doc_id
    found = sorted(
        (path for path in folder.glob("p*.*") if _PAGE_FILE.fullmatch(path.name)),
        key=lambda path: int(_PAGE_FILE.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
    )
    if not found:
        raise PagesUnavailable(
            f"{doc_id}: no preview pages in {folder} "
            f"(run: python scripts/ingest.py --preview --doc {doc_id})"
        )
    return found


def page_data_url(path: Path) -> tuple[str, tuple[int, int]]:
    """A page as a data URL (exactly the stored bytes) and its pixel size."""
    data = path.read_bytes()
    with Image.open(path) as image:
        size = image.size
        mime = "image/png" if image.format == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", size


# ---------------------------------------------------------------- the viewer

# Pure geometry, kept apart so tests can run it under Node. A pixel is identified like the
# redaction code does: floor of the position scaled to the STORED page, clamped to the page; a
# box dragged between two pixels covers both of them (x, y, w, h with w, h >= 1).
GEOMETRY_JS = """
function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }
function pagePixel(clientX, clientY, rect, naturalW, naturalH) {
  const x = Math.floor((clientX - rect.left) * naturalW / rect.width);
  const y = Math.floor((clientY - rect.top) * naturalH / rect.height);
  return { x: clamp(x, 0, naturalW - 1), y: clamp(y, 0, naturalH - 1) };
}
function boxFromDrag(a, b) {
  return [Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(a.x - b.x) + 1, Math.abs(a.y - b.y) + 1];
}
function yamlSnippet(doc, page, box) {
  return doc + ":\\n  " + page + ":\\n    - [" + box.join(", ") + "]";
}
"""

_VIEWER = """<!doctype html><meta charset="utf-8">
<style>
 body { margin: 0; font: 13px system-ui, sans-serif; }
 #bar { padding: 4px 8px; background: #f3f3f3; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 2; }
 #bar b { font-variant-numeric: tabular-nums; }
 #yaml { margin: 4px 8px; padding: 4px 6px; background: #fafafa; border: 1px solid #e3e3e3; display: none; user-select: all; }
 #stage { position: relative; display: inline-block; cursor: crosshair; }
 #page { display: block; width: __WIDTH__px; user-select: none; -webkit-user-drag: none; }
 #sel { position: absolute; border: 2px solid #e00; background: rgba(255,0,0,.12); display: none; pointer-events: none; }
 .warn { color: #b00; font-weight: 600; }
</style>
<div id="bar">__BANNER__<span id="pos">move over the page</span> &nbsp;|&nbsp; box <b id="box">drag to draw one</b>
 &nbsp;<button id="copy" type="button">copy YAML</button></div>
<pre id="yaml"></pre>
<div id="stage"><img id="page" src="__SRC__" alt="page __PAGE__"><div id="sel"></div></div>
<script>
__GEOMETRY__
const NATURAL_W = __W__, NATURAL_H = __H__, DOC = __DOC__, PAGE = __PAGE__;
const img = document.getElementById("page"), sel = document.getElementById("sel");
const pos = document.getElementById("pos"), boxEl = document.getElementById("box");
const yamlEl = document.getElementById("yaml");
let anchor = null, current = null, lastBox = null;
function at(e) { return pagePixel(e.clientX, e.clientY, img.getBoundingClientRect(), NATURAL_W, NATURAL_H); }
function show(box) {
  const r = img.getBoundingClientRect(), sx = r.width / NATURAL_W, sy = r.height / NATURAL_H;
  sel.style.display = "block";
  sel.style.left = box[0] * sx + "px"; sel.style.top = box[1] * sy + "px";
  sel.style.width = box[2] * sx + "px"; sel.style.height = box[3] * sy + "px";
}
img.addEventListener("mousemove", function (e) {
  const p = at(e);
  pos.innerHTML = "x <b>" + p.x + "</b> y <b>" + p.y + "</b> (page pixels)";
  if (anchor) { current = p; lastBox = boxFromDrag(anchor, p); show(lastBox); boxEl.textContent = "[" + lastBox.join(", ") + "]"; }
});
img.addEventListener("mousedown", function (e) { e.preventDefault(); anchor = at(e); current = anchor; });
window.addEventListener("mouseup", function () {
  if (!anchor) return;
  lastBox = boxFromDrag(anchor, current); anchor = null;
  show(lastBox); boxEl.textContent = "[" + lastBox.join(", ") + "]";
  yamlEl.style.display = "block"; yamlEl.textContent = yamlSnippet(DOC, PAGE, lastBox);
});
document.getElementById("copy").addEventListener("click", function () {
  if (!lastBox) return;
  const text = yamlSnippet(DOC, PAGE, lastBox);
  if (navigator.clipboard) navigator.clipboard.writeText(text).catch(function () {});
  yamlEl.style.display = "block"; yamlEl.textContent = text;
});
</script>
"""  # noqa: E501  (an HTML/CSS/JS template: its lines are not Python)


def viewer_html(
    data_url: str,
    *,
    doc_id: str,
    page: int,
    natural_size: tuple[int, int],
    width_px: int,
    unredacted: bool = False,
) -> str:
    """The self-contained viewer for one page, to be shown with `components.html`."""
    banner = (
        '<span class="warn">UNREDACTED preview: for finding redaction boxes only. '
        "Never label from it, never share it.</span> &nbsp;|&nbsp; "
        if unredacted
        else ""
    )
    replacements = {
        "__GEOMETRY__": GEOMETRY_JS,
        "__BANNER__": banner,
        "__SRC__": data_url,
        "__WIDTH__": str(width_px),
        "__W__": str(natural_size[0]),
        "__H__": str(natural_size[1]),
        "__DOC__": json.dumps(doc_id),
        "__PAGE__": str(page),
    }
    html = _VIEWER
    for marker, value in replacements.items():
        html = html.replace(marker, value)
    return html


def viewer_height(natural_size: tuple[int, int], width_px: int) -> int:
    """The iframe height that shows the whole page plus the coordinate bar."""
    return round(width_px * natural_size[1] / natural_size[0]) + 90
