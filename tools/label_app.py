"""The labeling app.

    python -m uv run streamlit run tools/label_app.py

Left: the document's STORED (redacted) pages, with the pixel under the cursor and a draggable box
that prints `[x, y, w, h]` and a redactions.yaml snippet. Right: the form, one field per row.
Every field starts PENDING and has to be decided: `verified` (what is in the box is what the page
prints), `absent` (the page does not print it: null) or `unlabeled` (skipped in scoring). The label
can be saved only when nothing is pending. A pre-labeled document starts from the model's draft;
a BLIND document starts empty and never sees one.

Work is kept in outputs/labeling/ as you go, so a refresh loses nothing. Saving writes
ground_truth/<doc_id>.json and its own ground_truth/CHANGELOG.md line; replacing a finished label
asks for a reason.

This file is deliberately thin: the rules live in evalkit/labeling.py (tested without a browser),
the page viewer and bookkeeping in evalkit/labeling_ui.py.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from docextract.ingest.manifest import ManifestError, ManifestRow, read_manifest
from docextract.paths import DatasetPaths, find_repo_root
from evalkit.groundtruth import SCALAR_FIELD_NAMES
from evalkit.gt_store import check_labels, label_path
from evalkit.labeling import (
    ROW_CELLS,
    UNLABELED_CELL,
    BlindDraftError,
    LabelForm,
    LabelingError,
    parse_pasted_rows,
    save_label,
)
from evalkit.labeling_ui import (
    Opened,
    PagesUnavailable,
    clear_wip,
    frozen_input,
    load_form,
    overview,
    page_data_url,
    page_files,
    save_wip,
    viewer_height,
    viewer_html,
)

PATHS = DatasetPaths(Path(os.environ.get("DOCEXTRACT_ROOT") or find_repo_root()))
BADGES = {
    "pending": ":orange[pending]",
    "verified": ":green[verified]",
    "absent": ":gray[absent (null)]",
    "unlabeled": ":blue[unlabeled]",
}
STATE_MARK = {"labeled": "done", "in_progress": "in progress", "drafted": "draft", "todo": "to do"}


# ---------------------------------------------------------------- state and callbacks


def key(doc: str, *parts: str) -> str:
    return ":".join((doc, *parts))


def opened(row: ManifestRow) -> Opened:
    forms: dict[str, Opened] = st.session_state.setdefault("forms", {})
    if row.doc_id not in forms:
        forms[row.doc_id] = load_form(row, PATHS)
    return forms[row.doc_id]


def form_of(doc: str) -> LabelForm:
    return st.session_state["forms"][doc].form


def persist(doc: str) -> None:
    save_wip(form_of(doc), PATHS)


def on_text(doc: str, name: str) -> None:
    form_of(doc).set_text(name, st.session_state[key(doc, name, "text")])
    persist(doc)


def on_verify(doc: str, name: str) -> None:
    form = form_of(doc)
    form.set_text(name, st.session_state[key(doc, name, "text")])
    form.verify(name)
    st.session_state[key(doc, name, "text")] = form.fields[name].text  # what is stored
    persist(doc)


def on_absent(doc: str, name: str) -> None:
    form_of(doc).mark_absent(name)
    st.session_state[key(doc, name, "text")] = ""
    persist(doc)


def on_unlabeled(doc: str, name: str) -> None:
    form_of(doc).mark_unlabeled(name)
    st.session_state[key(doc, name, "text")] = ""
    persist(doc)


def on_note(doc: str, name: str) -> None:
    form_of(doc).set_note(name, st.session_state[key(doc, name, "note")])
    persist(doc)


def on_doc_notes(doc: str) -> None:
    form_of(doc).notes = st.session_state[key(doc, "notes")]
    persist(doc)


def on_paste(doc: str, append: bool) -> None:
    """Read the spreadsheet cells in the paste box into the rows (a callback, because a widget's
    own state can only be cleared before the script draws it again)."""
    result = parse_pasted_rows(st.session_state.get(key(doc, "rows_paste"), ""))
    problems = key(doc, "rows_paste_problems")
    if result.problems:
        st.session_state[problems] = result.problems  # the grid is left exactly as it was
        return
    st.session_state.pop(problems, None)
    form_of(doc).paste_rows(result.rows, append=append)
    version = st.session_state.get(key(doc, "rows_version"), 0)
    st.session_state[key(doc, "rows_version")] = version + 1  # the grid shows the new rows
    st.session_state[key(doc, "rows_paste")] = ""
    persist(doc)


def discard(doc: str) -> None:
    clear_wip(doc, PATHS)
    for name in [k for k in st.session_state if str(k).startswith(f"{doc}:")]:
        del st.session_state[name]
    st.session_state["forms"].pop(doc, None)


@st.cache_data(show_spinner=False)
def cached_page(path: str, modified: float) -> tuple[str, tuple[int, int]]:
    return page_data_url(Path(path))


# ---------------------------------------------------------------- widgets


def field_widget(doc: str, name: str, form: LabelForm) -> None:
    state = form.fields[name]
    text_key = key(doc, name, "text")
    st.session_state.setdefault(text_key, state.text)
    st.session_state.setdefault(key(doc, name, "note"), state.note)
    with st.container(border=True):
        label, box, verify, absent, unlabeled = st.columns([2.2, 5, 1.3, 1.3, 1.4])
        label.markdown(f"**{name}**  \n{BADGES[state.status]}")
        box.text_input(
            name,
            key=text_key,
            label_visibility="collapsed",
            on_change=on_text,
            args=(doc, name),
        )
        verify.button(
            "verified",
            key=key(doc, name, "verify"),
            on_click=on_verify,
            args=(doc, name),
            type="primary" if state.status == "verified" else "secondary",
            use_container_width=True,
        )
        absent.button(
            "absent",
            key=key(doc, name, "absent"),
            on_click=on_absent,
            args=(doc, name),
            type="primary" if state.status == "absent" else "secondary",
            use_container_width=True,
        )
        unlabeled.button(
            "unlabeled",
            key=key(doc, name, "unlabeled"),
            on_click=on_unlabeled,
            args=(doc, name),
            type="primary" if state.status == "unlabeled" else "secondary",
            use_container_width=True,
        )
        if state.proposal is not None:
            hint = f" ({state.hint})" if state.hint else ""
            st.caption(f"draft says: {state.proposal}{hint}")
        if state.error:
            st.error(state.error)
        st.text_input(
            f"note on {name}",
            key=key(doc, name, "note"),
            label_visibility="collapsed",
            placeholder="note (optional): redacted, total not printed, ...",
            on_change=on_note,
            args=(doc, name),
        )


def rows_widget(doc: str, form: LabelForm) -> None:
    version = st.session_state.setdefault(key(doc, "rows_version"), 0)
    st.session_state.setdefault(key(doc, "line_items", "note"), form.rows_note)
    with st.container(border=True):
        st.markdown(f"**line_items**  \n{BADGES[form.rows_status]}")
        if form.rows_proposal is not None:
            hint = f" ({form.rows_hint})" if form.rows_hint else ""
            st.caption(f"draft says: {form.rows_proposal}{hint}")
        st.caption(
            "One row per printed row, in order. Leave a cell empty if the row does not show it; "
            f"type {UNLABELED_CELL} to leave a cell unlabeled. Amounts as printed."
        )
        frame = frozen_input(
            st.session_state,
            key(doc, "rows_source"),
            version,
            lambda: pd.DataFrame(form.rows or [], columns=list(ROW_CELLS)).astype(str),
        )
        edited = st.data_editor(
            frame,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key=key(doc, "rows_editor", str(version)),
        )
        before = [dict(item) for item in form.rows]
        form.set_rows(edited.fillna("").to_dict("records"))
        if form.rows != before:  # only real edits write saved work, never just looking
            persist(doc)
        with st.expander("Paste rows from a spreadsheet"):
            st.caption(
                "Copy a block of cells from Excel or Sheets and paste it here. Columns: "
                f"{', '.join(ROW_CELLS)} (fewer is fine; a header line is skipped)."
            )
            st.text_area(
                "pasted cells",
                key=key(doc, "rows_paste"),
                height=140,
                label_visibility="collapsed",
            )
            replace, add = st.columns(2)
            replace.button(
                "replace the rows",
                key=key(doc, "rows", "paste_replace"),
                on_click=on_paste,
                args=(doc, False),
                use_container_width=True,
            )
            add.button(
                "add after the last row",
                key=key(doc, "rows", "paste_add"),
                on_click=on_paste,
                args=(doc, True),
                use_container_width=True,
            )
            for problem in st.session_state.get(key(doc, "rows_paste_problems"), []):
                st.error(problem)
        verify, absent, unlabeled = st.columns(3)
        if verify.button("verify rows", key=key(doc, "rows", "verify"), use_container_width=True):
            form.verify_rows()
            st.session_state[key(doc, "rows_version")] = version + 1  # show what is stored
            persist(doc)
            st.rerun()
        if absent.button(
            "no itemization (null)", key=key(doc, "rows", "absent"), use_container_width=True
        ):
            form.mark_rows_absent()
            persist(doc)
            st.rerun()
        if unlabeled.button(
            "unlabeled", key=key(doc, "rows", "unlabeled"), use_container_width=True
        ):
            form.mark_rows_unlabeled()
            persist(doc)
            st.rerun()
        for problem in form.rows_errors:
            st.error(problem)
        st.text_input(
            "note on line_items",
            key=key(doc, "line_items", "note"),
            label_visibility="collapsed",
            placeholder="note (optional)",
            on_change=on_note,
            args=(doc, "line_items"),
        )


def viewer(doc: str) -> None:
    st.subheader("Pages")
    preview = st.toggle(
        "UNREDACTED preview (only to find redaction coordinates; never label from it)",
        key=key(doc, "preview"),
    )
    try:
        files = page_files(doc, PATHS, preview=preview)
    except PagesUnavailable as exc:
        st.warning(str(exc))
        return
    page = (
        st.number_input("Page", 1, len(files), 1, key=key(doc, "page"), step=1)
        if len(files) > 1
        else 1
    )
    width = st.slider("Zoom (display width in px)", 400, 1800, 900, 50, key=key(doc, "zoom"))
    path = files[int(page) - 1]
    url, size = cached_page(str(path), path.stat().st_mtime)
    components.html(
        viewer_html(
            url, doc_id=doc, page=int(page), natural_size=size, width_px=width, unredacted=preview
        ),
        height=viewer_height(size, width),
        scrolling=True,
    )
    st.caption(f"{path.name}: {size[0]} x {size[1]} px. Coordinates are in these pixels.")


def save_section(doc: str, row: ManifestRow, form: LabelForm) -> None:
    st.divider()
    finished = label_path(PATHS.ground_truth_dir, doc).exists()
    reason = ""
    if finished:
        st.info(f"{doc} already has a finished label. Replacing it is logged in CHANGELOG.md.")
        reason = st.text_input("Why is the finished label being changed?", key=key(doc, "reason"))
    pending = form.pending()
    if pending:
        st.caption(f"{len(pending)} field(s) still pending: {', '.join(pending)}")
    if st.button("Save label", key="save", type="primary", disabled=not form.can_save):
        try:
            result = save_label(
                form.build_label(),
                row,
                ground_truth_dir=PATHS.ground_truth_dir,
                changelog=PATHS.changelog,
                drafts_dir=PATHS.drafts_dir,
                reason=reason,
            )
        except LabelingError as exc:
            st.error(str(exc))
        else:
            clear_wip(doc, PATHS)
            st.session_state["forms"][doc] = Opened(form, "finished label")
            if result.status == "unchanged":
                st.success(f"Nothing to save: {result.path.name} already says exactly this.")
            else:
                st.success(f"Saved {result.path} ({result.status}).")
                for line in result.changelog:
                    st.code(line, language=None)


def form_panel(doc: str, row: ManifestRow, current: Opened) -> None:
    form = current.form
    st.subheader(f"Label {doc}")
    if row.label_mode == "blind":
        st.warning(
            "BLIND document: labeled from the page alone. No draft exists and none is shown."
        )
    elif current.source == "draft":
        st.info(
            "Pre-label draft loaded. Check EVERY field against the page: nothing is decided "
            "until you click verified, absent or unlabeled."
        )
        if form.draft_notes:
            st.caption(f"draft: {form.draft_notes}")
    st.caption(f"opened from: {current.source}")
    if current.warning:
        st.warning(current.warning)
    done, total = form.progress()
    st.progress(done / total, text=f"{done} of {total} fields decided")
    for name in SCALAR_FIELD_NAMES:
        field_widget(doc, name, form)
        if name == "payment_method":  # the table sits between the header facts and the totals
            rows_widget(doc, form)
    st.session_state.setdefault(key(doc, "notes"), form.notes)
    st.text_area(
        "Notes on the document",
        key=key(doc, "notes"),
        on_change=on_doc_notes,
        args=(doc,),
        placeholder="anything else a reader of this label should know",
    )
    save_section(doc, row, form)


# ---------------------------------------------------------------- the page


def main() -> None:
    st.set_page_config(page_title="Label documents", layout="wide")
    st.title("Label documents")
    try:
        rows = read_manifest(PATHS.manifest)
    except (ManifestError, OSError) as exc:
        st.error(f"The manifest cannot be read: {exc}")
        st.stop()
    docs = overview(rows, PATHS)
    if not docs:
        st.info("No documents to label yet: run scripts/assign_split.py so they have a label_mode.")
        st.stop()
    by_id: dict[str, ManifestRow] = {row.doc_id: row for row in rows}

    with st.sidebar:
        st.header("Documents")
        labels = {
            d.doc_id: f"{d.doc_id}  ·  {d.doc_type or '?'}  ·  {d.split or '-'}  ·  "
            f"{d.label_mode}  ·  {STATE_MARK[d.state]}"
            for d in docs
        }
        doc = st.selectbox("Document", list(labels), format_func=labels.__getitem__, key="doc")
        counts: dict[str, int] = {}
        for item in docs:
            counts[item.state] = counts.get(item.state, 0) + 1
        st.caption(", ".join(f"{n} {STATE_MARK[s]}" for s, n in sorted(counts.items())))
        problems = check_labels(rows, PATHS.ground_truth_dir, PATHS.drafts_dir)
        if problems:
            with st.expander(f"{len(problems)} integrity problem(s)", expanded=True):
                for problem in problems:
                    st.error(problem)
        st.button("Discard my unsaved work here", key="discard", on_click=discard, args=(doc,))

    row = by_id[doc]
    try:
        current = opened(row)
    except (BlindDraftError, LabelingError) as exc:
        st.error(str(exc))
        st.stop()

    left, right = st.columns([3, 2])
    with left:
        viewer(doc)
    with right:
        form_panel(doc, row, current)


main()
