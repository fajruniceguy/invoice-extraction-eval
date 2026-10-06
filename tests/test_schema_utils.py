import json
from typing import Any

import pytest
from pydantic import BaseModel

from docextract.extract.schema import DocumentExtraction
from docextract.providers.schema_utils import simplify_schema


class Inner(BaseModel):
    name: str
    qty: int = 1


class Outer(BaseModel):
    title: str  # a property that happens to be called "title"
    inner: Inner
    inners: list[Inner]
    maybe: Inner | None = None


def walk(node: Any) -> Any:
    """Yield every dict in a nested JSON structure."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def test_refs_are_inlined_and_defs_removed() -> None:
    schema = simplify_schema(Outer.model_json_schema())
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    assert schema["properties"]["inner"]["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["inners"]["items"]["properties"]["name"] == {"type": "string"}


def test_nullable_union_is_kept_but_inlined() -> None:
    schema = simplify_schema(Outer.model_json_schema())
    options = schema["properties"]["maybe"]["anyOf"]
    assert {"type": "null"} in options
    assert any(option.get("type") == "object" for option in options)


def test_title_and_default_keywords_are_dropped() -> None:
    schema = simplify_schema(Outer.model_json_schema())
    assert "title" not in schema
    inner = schema["properties"]["inner"]
    assert "title" not in inner
    assert "default" not in inner["properties"]["qty"]
    assert "title" not in inner["properties"]["qty"]


def test_a_property_named_title_survives() -> None:
    schema = simplify_schema(Outer.model_json_schema())
    assert schema["properties"]["title"] == {"type": "string"}
    assert "title" in schema["required"]


def test_input_is_not_mutated() -> None:
    original = Outer.model_json_schema()
    snapshot = json.dumps(original, sort_keys=True)
    simplify_schema(original)
    assert json.dumps(original, sort_keys=True) == snapshot


def test_sibling_keywords_next_to_a_ref_are_kept() -> None:
    schema = {
        "$defs": {"X": {"type": "string", "title": "X"}},
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/X", "description": "an x"}},
    }
    assert simplify_schema(schema)["properties"]["a"] == {"type": "string", "description": "an x"}


def test_recursive_reference_is_rejected() -> None:
    schema = {
        "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
        "$ref": "#/$defs/Node",
    }
    with pytest.raises(ValueError, match="recursive"):
        simplify_schema(schema)


@pytest.mark.parametrize("ref", ["#/definitions/X", "http://x/y.json", "#/$defs/Missing"])
def test_bad_refs_are_rejected(ref: str) -> None:
    with pytest.raises(ValueError):
        simplify_schema({"$defs": {}, "properties": {"a": {"$ref": ref}}})


def test_real_extraction_schema_is_flat() -> None:
    schema = simplify_schema(DocumentExtraction.model_json_schema())
    text = json.dumps(schema)
    assert "$ref" not in text
    assert "$defs" not in text
    # no DocumentExtraction property is called "title"/"default", so any dict carrying one of
    # them as a key would be a leftover schema keyword
    for node in walk(schema):
        assert "title" not in node
        assert "default" not in node
    total = schema["properties"]["total"]
    assert set(total["properties"]) == {
        "value",
        "confidence",
        "status",
        "evidence",
        "page",
        "reasons",
    }
    assert schema["properties"]["line_items"]["items"]["properties"]["amount"]
