"""Rebuild a Pydantic model from a JSON Schema, so an output format can be
stored as data (a class is code; its schema is not)."""

from __future__ import annotations

import typing as t

from pydantic import BaseModel, Field, create_model

_PRIMITIVES: dict[str, t.Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "null": type(None),
}


def schema_of(model: type[BaseModel]) -> dict[str, t.Any]:
    """The storable form of an output format."""
    return model.model_json_schema()


def model_from_schema(schema: dict[str, t.Any], name: str | None = None) -> type[BaseModel]:
    """A model with the schema's shape: fields, types, required, defaults and
    descriptions. Custom validators and methods of the original class are
    code, so they are not part of the schema and are not restored."""
    defs = schema.get("$defs", {})
    built: dict[str, type[BaseModel]] = {}

    def resolve(node: dict[str, t.Any], hint: str) -> t.Any:
        if "$ref" in node:
            ref = node["$ref"].rsplit("/", 1)[-1]
            if ref not in built:
                built[ref] = build(defs[ref], ref)
            return built[ref]
        if "enum" in node:
            return t.Literal[tuple(node["enum"])]
        if "const" in node:
            return t.Literal[node["const"]]
        for key in ("anyOf", "oneOf"):
            if key in node:
                options = tuple(resolve(option, hint) for option in node[key])
                return t.Union[options] if len(options) > 1 else options[0]
        kind = node.get("type")
        if isinstance(kind, list):
            return t.Union[tuple(resolve({**node, "type": k}, hint) for k in kind)]
        if kind == "array":
            return list[resolve(node.get("items", {}), hint)]
        if kind == "object":
            if "properties" in node:
                return build(node, node.get("title", hint))
            return dict[str, resolve(node.get("additionalProperties") or {}, hint)]
        return _PRIMITIVES.get(kind, t.Any)

    def build(node: dict[str, t.Any], model_name: str) -> type[BaseModel]:
        required = set(node.get("required", ()))
        fields: dict[str, t.Any] = {}
        for field, prop in node.get("properties", {}).items():
            annotation = resolve(prop, f"{model_name}_{field}")
            default = prop.get("default", ... if field in required else None)
            fields[field] = (annotation, Field(default, description=prop.get("description")))
        return create_model(model_name, __doc__=node.get("description"), **fields)

    return build(schema, name or schema.get("title", "Output"))


__all__ = ["model_from_schema", "schema_of"]
