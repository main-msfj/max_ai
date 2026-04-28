"""Tests for clean_json_schema."""

from __future__ import annotations

from max_ai.clients.ollama._schema import clean_json_schema


# -------- BASIC SHAPE -----------------------------------------------------------
def test_strips_root_title_and_additional_properties():
    raw = {
        "additionalProperties": False,
        "properties": {"query": {"title": "Query", "type": "string"}},
        "required": ["query"],
        "title": "search_docs_params",
        "type": "object",
    }
    result = clean_json_schema(raw)
    assert "title" not in result
    assert "additionalProperties" not in result
    assert result["type"] == "object"


def test_strips_property_titles():
    raw = {
        "type": "object",
        "properties": {
            "query": {"title": "Query", "type": "string"},
            "limit": {"title": "Limit", "type": "integer", "default": 5},
        },
        "required": ["query"],
    }
    result = clean_json_schema(raw)
    assert result["properties"]["query"] == {"type": "string"}
    assert result["properties"]["limit"] == {"type": "integer", "default": 5}


def test_keeps_required_when_non_empty():
    raw = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    assert clean_json_schema(raw)["required"] == ["q"]


def test_drops_empty_required():
    raw = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": [],
    }
    assert "required" not in clean_json_schema(raw)


def test_drops_missing_required():
    raw = {"type": "object", "properties": {"q": {"type": "string"}}}
    assert "required" not in clean_json_schema(raw)


# -------- DESCRIPTION + DEFAULT + ENUM ARE PRESERVED -----------------------------
def test_preserves_description():
    raw = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "Search query", "title": "Q"}
        },
    }
    result = clean_json_schema(raw)
    assert result["properties"]["q"] == {
        "type": "string",
        "description": "Search query",
    }


def test_preserves_default_and_enum():
    raw = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "default": "fast",
                "enum": ["fast", "deep"],
                "title": "Mode",
            },
        },
    }
    result = clean_json_schema(raw)
    assert result["properties"]["mode"] == {
        "type": "string",
        "default": "fast",
        "enum": ["fast", "deep"],
    }


# -------- NESTED SHAPES -----------------------------------------------------------
def test_recurses_into_array_items():
    raw = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string", "title": "Tag"},
            },
        },
    }
    result = clean_json_schema(raw)
    assert result["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_recurses_into_nested_object_properties():
    raw = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "title": "Filter",
                "properties": {
                    "name": {"type": "string", "title": "Name"},
                    "age": {"type": "integer", "title": "Age"},
                },
                "required": ["name"],
            },
        },
    }
    result = clean_json_schema(raw)
    nested = result["properties"]["filter"]
    assert "title" not in nested
    assert nested["type"] == "object"
    assert nested["properties"]["name"] == {"type": "string"}
    assert nested["properties"]["age"] == {"type": "integer"}


def test_recurses_into_anyof():
    raw = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string", "title": "X"},
                    {"type": "integer", "title": "Y"},
                ],
            },
        },
    }
    result = clean_json_schema(raw)
    assert result["properties"]["value"]["anyOf"] == [
        {"type": "string"},
        {"type": "integer"},
    ]


# -------- DEFENSIVE -----------------------------------------------------------
def test_empty_schema_returns_object_with_empty_properties():
    assert clean_json_schema({}) == {"type": "object", "properties": {}}


def test_does_not_mutate_input():
    raw = {
        "type": "object",
        "properties": {"q": {"type": "string", "title": "Q"}},
        "title": "Original",
    }
    snapshot = {
        "type": "object",
        "properties": {"q": {"type": "string", "title": "Q"}},
        "title": "Original",
    }
    clean_json_schema(raw)
    assert raw == snapshot