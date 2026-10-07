import importlib
import subprocess
import sys

import pytest

MODULES = [
    "docextract",
    "docextract.providers",
    "docextract.ingest",
    "docextract.extract",
    "docextract.validate",
    "docextract.confidence",
    "docextract.api",
    "evalkit",
    "evalkit.comparators",
    "evalkit.matching",
    "evalkit.metrics",
    "evalkit.taxonomy",
    "evalkit.scorecard",
    "evalkit.compare",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_importing_confidence_before_validate_works() -> None:
    """docextract.confidence names docextract.validate's reason codes, so importing it first
    must not meet a half-initialized package (validate's own imports of confidence are inside
    validate_document for exactly this reason)."""
    code = (
        "import docextract.confidence.formula, docextract.confidence.abstain,"
        " docextract.validate"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
