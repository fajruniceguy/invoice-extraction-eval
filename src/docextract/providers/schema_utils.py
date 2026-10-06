"""JSON-Schema clean-up for strict structured-output endpoints.

Pydantic emits `$defs`/`$ref`, `title` and `default` keywords that some OpenAI-compatible
servers (notably Gemini's compatibility layer) reject or mishandle. `simplify_schema`
inlines every `$ref` and drops those keywords. Everything else is left untouched.
"""

from typing import Any

_REF_PREFIX = "#/$defs/"
_DROPPED = {"title", "default", "$defs"}
_SCHEMA_MAPS = {"properties"}  # keyword -> {name: schema}
_SCHEMA_LISTS = {"anyOf", "oneOf", "allOf", "prefixItems"}  # keyword -> [schema, ...]
_SCHEMA_VALUES = {"items", "additionalProperties", "not", "contains"}  # keyword -> schema


def simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return _simplify(schema, schema.get("$defs", {}), ())


def _simplify(node: dict[str, Any], defs: dict[str, Any], stack: tuple[str, ...]) -> dict[str, Any]:
    if "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith(_REF_PREFIX):
            raise ValueError(f"unsupported $ref: {ref!r}")
        name = ref[len(_REF_PREFIX) :]
        if name in stack:
            raise ValueError(f"recursive schema reference is not supported: {name}")
        if name not in defs:
            raise ValueError(f"$ref to unknown definition: {name}")
        target = _simplify(defs[name], defs, (*stack, name))
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        return {**target, **_simplify(siblings, defs, stack)}

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED:
            continue
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {name: _simplify(sub, defs, stack) for name, sub in value.items()}
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            out[key] = [_simplify(sub, defs, stack) for sub in value]
        elif key in _SCHEMA_VALUES and isinstance(value, dict):
            out[key] = _simplify(value, defs, stack)
        else:
            out[key] = value
    return out
