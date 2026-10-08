"""docs/demo/degraded_prompt.txt is a frozen artifact: the production prompt with three documented
edits, used to show the regression gate catching a worse prompt (docs/demo/regression_demo.md).
Its hash is pinned, so it cannot drift silently; a later edit of the production prompt does not
break it (it stays what it was)."""

import hashlib

from docextract.extract.prompt import PROMPT_SHA256, SYSTEM_PROMPT, prompt_sha256
from ingest_support import ROOT

PATH = ROOT / "docs" / "demo" / "degraded_prompt.txt"
PINNED = "1e50e734beaac6fd09244d9673adaa93c9b56dc02826fc0d14511863185cfd7c"


def text() -> str:
    return PATH.read_text(encoding="utf-8")  # universal newlines: a CRLF checkout hashes the same


def test_the_file_is_the_frozen_artifact() -> None:
    assert hashlib.sha256(text().encode()).hexdigest() == PINNED


def test_it_lacks_what_the_edits_remove() -> None:
    removed = [
        "subtotal: the amount before tax as printed",  # E1 (swapped definitions)
        "amount to pay, ONLY if printed as such",
        '"Rp 1.250.000,00" is 1250000.00',  # E2 (thousands of rupiah)
        "Never guess, infer or compute a value that is not printed",  # E3 (best estimate)
    ]
    for phrase in removed:
        assert phrase not in text(), phrase


def test_it_holds_what_the_edits_add() -> None:
    added = [
        "subtotal: the grand total or amount to pay.",
        "total: the amount before tax (on a faktur the DPP), ONLY if printed as such",
        "Express every money amount in thousands of rupiah",
        '"Rp 1.250.000,00" is 1250.00',
        "Never leave a field empty",
    ]
    for phrase in added:
        assert phrase in text(), phrase


def test_it_is_another_prompt_to_the_run_record() -> None:
    assert prompt_sha256(text()) != PROMPT_SHA256
    assert text() != SYSTEM_PROMPT
