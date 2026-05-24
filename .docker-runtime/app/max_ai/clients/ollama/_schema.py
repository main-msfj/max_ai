"""
JSON Schema cleanup for Ollama tool calling.

Pydantic generates JSON Schema with metadata that's useful for OpenAPI
docs but adds noise (or outright confuses) local LLMs running through
Ollama. This module strips that noise down to what the model actually
needs to decide how to call a tool.

What we strip:
  - ``title`` at the schema root and inside every property — pydantic
    auto-generates these from field names; the LLM already sees the
    field name as the property key.
  - ``additionalProperties: false`` — strict-mode artifact. Local
    models like qwen3 / llama3.1 are inconsistent with it and it
    doesn't help the model pick correct arguments.

What we keep (per property):
  ``type``, ``description``, ``default``, ``enum``, ``items``,
  ``properties``, ``required``, ``anyOf``, ``oneOf``, ``allOf``,
  ``$ref``, ``format``, ``minimum``, ``maximum``, ``minLength``,
  ``maxLength``, ``pattern``.

This is intentionally Ollama-specific. Anthropic and OpenAI clients
will have their own variants in their own packages — they have
different tolerance for strict-mode and different conventions.
"""

from __future__ import annotations

import typing as t


# Keys we keep on every property. Anything outside this set is dropped.
_PROPERTY_KEEP_KEYS: frozenset[str] = frozenset(
    {
        # core type info
        "type",
        "description",
        "default",
        "enum",
        # composition
        "anyOf",
        "oneOf",
        "allOf",
        "$ref",
        # nested shapes
        "items",
        "properties",
        "required",
        # format hints the model can use
        "format",
        # numeric constraints
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        # string constraints
        "minLength",
        "maxLength",
        "pattern",
        # array constraints
        "minItems",
        "maxItems",
    }
)


def clean_json_schema(schema: dict[str, t.Any]) -> dict[str, t.Any]:
    """Clean a JSON Schema dict for use as an Ollama tool ``parameters`` field.

    Returns a new dict — the input is not mutated.

    Args:
        schema: A JSON Schema dict, typically produced by
            ``FunctionAsTool.parameters`` (which comes from a pydantic
            ``TypeAdapter.json_schema()`` call).

    Returns:
        A cleaned dict with ``type``, ``properties`` and (when
        non-empty) ``required``. Nested object/array schemas are
        cleaned recursively.

    Example:
        >>> raw = {
        ...     "additionalProperties": False,
        ...     "properties": {
        ...         "query": {"title": "Query", "type": "string"},
        ...         "limit": {"default": 5, "title": "Limit", "type": "integer"},
        ...     },
        ...     "required": ["query"],
        ...     "title": "search_docs_params",
        ...     "type": "object",
        ... }
        >>> clean_json_schema(raw)
        {'type': 'object', 'properties': {'query': {'type': 'string'}, 'limit': {'type': 'integer', 'default': 5}}, 'required': ['query']}
    """
    cleaned: dict[str, t.Any] = {"type": schema.get("type", "object")}

    properties = schema.get("properties")
    if isinstance(properties, dict):
        cleaned["properties"] = {
            name: _clean_property(prop) for name, prop in properties.items()
        }
    else:
        cleaned["properties"] = {}

    required = schema.get("required")
    if isinstance(required, list) and required:
        cleaned["required"] = list(required)

    return cleaned


def _clean_property(prop: t.Any) -> dict[str, t.Any]:
    """Clean a single property dict, recursing into nested shapes."""
    if not isinstance(prop, dict):
        # Defensive — pydantic should never give us this, but if a
        # caller passes a hand-rolled schema, don't crash.
        return {"type": "string"}

    cleaned: dict[str, t.Any] = {}
    for key, value in prop.items():
        if key not in _PROPERTY_KEEP_KEYS:
            continue
        cleaned[key] = _recurse(key, value)

    # Every property needs a type for the LLM to make sense of it.
    # If the source schema only had {"$ref": ...} or {"anyOf": ...},
    # leave it; the model can still read those.
    if "type" not in cleaned and not any(
        k in cleaned for k in ("$ref", "anyOf", "oneOf", "allOf")
    ):
        cleaned["type"] = "string"

    return cleaned


def _recurse(key: str, value: t.Any) -> t.Any:
    """Recurse into nested schemas where applicable."""
    if key == "properties" and isinstance(value, dict):
        # Nested object — clean each sub-property.
        return {name: _clean_property(p) for name, p in value.items()}

    if key == "items" and isinstance(value, dict):
        # Array items schema — single property-like shape.
        return _clean_property(value)

    if key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
        return [_clean_property(v) for v in value]

    return value