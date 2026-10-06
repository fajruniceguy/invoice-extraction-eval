import importlib

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
