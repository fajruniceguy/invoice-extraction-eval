"""data/redactions.yaml: which parts of which stored pages are burned black at ingestion.

    doc_007:                     # a doc_id from the manifest
      1:                         # page number of the STORED document (after page_range)
        - [120, 340, 400, 60]    # x, y, w, h in stored-page pixels
      2:
        size: [1654, 2339]       # optional: the page size these coordinates were measured on
        boxes:
          - [10, 10, 50, 50]

Everything here is strict, because a mistake that is accepted quietly is a page that is stored
unredacted: a repeated key (PyYAML would silently keep only the last one), a page number that is
not a positive integer, an empty entry, a box that is not four whole numbers. The pinned `size` is
checked against the page as it is actually rendered, so coordinates measured at another DPI or
size cap are caught instead of covering the wrong pixels.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from docextract.ingest.manifest import ManifestRow
from docextract.ingest.redact import Box, RedactionError, as_pixel, parse_box

_PAGE_KEY = re.compile(r"[0-9]+")
_PAGE_KEYS = {"boxes", "size"}


@dataclass(frozen=True)
class PageRedaction:
    boxes: tuple[Box, ...]
    size: tuple[int, int] | None = None  # the stored page size the boxes were measured on


Redactions = dict[str, dict[int, PageRedaction]]  # doc_id -> 1-based page -> its boxes


class _UniqueKeyLoader(yaml.SafeLoader):
    """A SafeLoader that refuses a repeated mapping key instead of keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=True)
            try:
                repeated = key in seen
            except TypeError:  # unhashable key: the parent class reports it
                continue
            if repeated:
                raise ConstructorError(
                    None, None, f"the key {key!r} appears twice", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _yaml_problem(exc: yaml.YAMLError) -> str:
    if isinstance(exc, yaml.MarkedYAMLError):
        line = f"line {exc.problem_mark.line + 1}: " if exc.problem_mark else ""
        return f"{line}{exc.problem or exc.context or 'not valid YAML'}"
    return str(exc)


def _page_number(key: Any, where: str) -> int:
    if isinstance(key, int) and not isinstance(key, bool):
        number = key
    elif isinstance(key, str) and _PAGE_KEY.fullmatch(key):
        number = int(key)
    else:
        raise RedactionError(f"{where}: {key!r} is not a page number (pages are 1, 2, 3, ...)")
    if number < 1:
        raise RedactionError(f"{where}: page numbers start at 1, got {key!r}")
    return number


def _parse_size(raw: Any, where: str) -> tuple[int, int]:
    if isinstance(raw, str) or not isinstance(raw, Sequence) or len(raw) != 2:
        raise RedactionError(f"{where}: size must be [width, height], got {raw!r}")
    width, height = (as_pixel(v, f"{where} size") for v in raw)
    if width < 1 or height < 1:
        raise RedactionError(f"{where}: size must be positive, got [{width}, {height}]")
    return (width, height)


def _parse_page(value: Any, where: str) -> PageRedaction:
    size: tuple[int, int] | None = None
    if isinstance(value, Mapping):
        unknown = sorted(str(key) for key in value if key not in _PAGE_KEYS)
        if unknown:
            raise RedactionError(f"{where}: unknown key(s) {', '.join(unknown)} (use boxes, size)")
        if "boxes" not in value:
            raise RedactionError(f"{where}: a page given as a mapping needs a 'boxes' list")
        if value.get("size") is not None:
            size = _parse_size(value["size"], where)
        raw_boxes = value["boxes"]
    else:
        raw_boxes = value
    if isinstance(raw_boxes, str) or not isinstance(raw_boxes, Sequence) or not raw_boxes:
        raise RedactionError(
            f"{where}: expected a non-empty list of [x, y, w, h] boxes, got {raw_boxes!r}"
        )
    boxes = tuple(
        parse_box(raw, f"{where} box {index}") for index, raw in enumerate(raw_boxes, start=1)
    )
    return PageRedaction(boxes, size)


def _parse_doc(doc_id: str, pages: Any, source: str) -> dict[int, PageRedaction]:
    if not isinstance(pages, Mapping) or not pages:
        raise RedactionError(
            f"{source}: {doc_id}: expected page numbers with boxes, got {pages!r} "
            "(remove the entry if the document needs no redaction)"
        )
    found: dict[int, PageRedaction] = {}
    for key, value in pages.items():
        number = _page_number(key, f"{source}: {doc_id}")
        if number in found:
            raise RedactionError(f"{source}: {doc_id}: page {number} is listed twice")
        found[number] = _parse_page(value, f"{source}: {doc_id} page {number}")
    return found


def parse_redactions(text: str, *, source: str = "redactions.yaml") -> Redactions:
    """Parse and shape-check the file's text. Page size and manifest are checked elsewhere."""
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise RedactionError(f"{source}: {_yaml_problem(exc)}") from exc
    if data is None:  # an empty file, or only comments
        return {}
    if not isinstance(data, dict):
        raise RedactionError(f"{source}: the top level must map doc ids to pages")
    result: Redactions = {}
    for doc_id, pages in data.items():
        if not isinstance(doc_id, str) or not doc_id:
            raise RedactionError(f"{source}: {doc_id!r} is not a doc id (they are text)")
        result[doc_id] = _parse_doc(doc_id, pages, source)
    return result


def read_redactions(path: Path, *, must_exist: bool = False) -> Redactions:
    """Load `path`. A missing file means "no redactions" unless `must_exist` (an explicit path)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if must_exist:
            raise RedactionError(f"{path}: no such file") from None
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise RedactionError(f"{path}: cannot be read ({exc})") from exc
    return parse_redactions(text, source=path.name)


def check_against_manifest(redactions: Redactions, rows: Sequence[ManifestRow]) -> list[str]:
    """Problems between the file and the manifest, for the WHOLE file (not just one document)."""
    by_id = {row.doc_id: row for row in rows}
    problems: list[str] = []
    for doc_id in redactions:
        row = by_id.get(doc_id)
        if row is None:
            problems.append(
                f"{doc_id}: not in the manifest (a typo? the document it was meant for "
                "would be stored unredacted)"
            )
        elif row.is_derived:
            problems.append(
                f"{doc_id}: is a degraded variant, which copies the pages of {row.gt_from}; "
                f"put the boxes on {row.gt_from}"
            )
    return problems


def boxes_of(entry: Mapping[int, PageRedaction]) -> dict[int, list[list[int]]]:
    """The plain page -> boxes mapping `ingest_document` takes."""
    return {page: [list(box) for box in found.boxes] for page, found in entry.items()}


def pinned_sizes_of(entry: Mapping[int, PageRedaction]) -> dict[int, tuple[int, int]]:
    return {page: found.size for page, found in entry.items() if found.size is not None}
