from pathlib import Path

from evalkit.gt_store import PLACEHOLDER, append_changelog

HEADER = (
    "# Ground-truth changelog\n\nFormat: `YYYY-MM-DD | file | field | old -> new | reason`\n\n---\n"
)


def test_it_creates_the_file_and_its_directory(tmp_path: Path) -> None:
    log = tmp_path / "new" / "CHANGELOG.md"
    append_changelog(log, ["2026-10-06 | a.json | x | 1 -> 2 | why"])
    assert log.read_text(encoding="utf-8") == "\n2026-10-06 | a.json | x | 1 -> 2 | why\n"


def test_it_replaces_the_placeholder_and_keeps_the_header(tmp_path: Path) -> None:
    log = tmp_path / "CHANGELOG.md"
    log.write_text(f"{HEADER}\n{PLACEHOLDER}\n", encoding="utf-8")
    append_changelog(log, ["line one", "line two"])
    text = log.read_text(encoding="utf-8")
    assert PLACEHOLDER not in text
    assert text == f"{HEADER}\nline one\nline two\n"


def test_later_lines_follow_earlier_ones_in_order(tmp_path: Path) -> None:
    log = tmp_path / "CHANGELOG.md"
    log.write_text(f"{HEADER}\n{PLACEHOLDER}\n", encoding="utf-8")
    append_changelog(log, ["one"])
    append_changelog(log, ["two", "three"])
    lines = log.read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if line in ("one", "two", "three")] == ["one", "two", "three"]
    assert lines[-1] == "three"
    assert lines.count("") >= 1 and PLACEHOLDER not in lines


def test_trailing_blank_lines_do_not_pile_up(tmp_path: Path) -> None:
    log = tmp_path / "CHANGELOG.md"
    log.write_text("# log\n\n\n\n", encoding="utf-8")
    append_changelog(log, ["entry"])
    assert log.read_text(encoding="utf-8") == "# log\n\nentry\n"


def test_existing_entries_are_never_touched(tmp_path: Path) -> None:
    log = tmp_path / "CHANGELOG.md"
    log.write_text("# log\n\nold entry | with | pipes\n", encoding="utf-8")
    append_changelog(log, ["new entry"])
    assert log.read_text(encoding="utf-8").startswith("# log\n\nold entry | with | pipes\n")
    assert log.read_text(encoding="utf-8").endswith("new entry\n")


def test_the_log_is_written_with_unix_line_endings(tmp_path: Path) -> None:
    log = tmp_path / "CHANGELOG.md"
    append_changelog(log, ["entry"])
    assert b"\r" not in log.read_bytes()
