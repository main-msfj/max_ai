"""Dotted-path references to Pydantic model classes.

Used wherever a model *type* must survive serialization (agent
``output_format`` in component configs, ``AssistantMessage.structured_output``
in persisted run contexts). Only importable, module-level classes can be
referenced — a class defined inside a function has no importable path.
"""

from __future__ import annotations

import importlib

from pydantic import BaseModel

from ..base.component import _check_allowed


def type_ref(value: type[BaseModel] | None) -> str | None:
    """Return the dotted import path for a model class, or None."""
    if value is None:
        return None
    return f"{value.__module__}.{value.__qualname__}"


def load_type_ref(value: str | None) -> type[BaseModel] | None:
    """Resolve a dotted import path back to a model class.

    Raises:
        ValueError: malformed reference.
        TypeError: the reference resolves to something that is not a
            Pydantic model class.
        ImportError / AttributeError: the module or attribute is gone.
    """
    if value is None:
        return None
    module_name, _, attr = value.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid type reference: {value!r}")
    # Stored sessions name this path; importing runs code, so allowlist it.
    _check_allowed(value)
    loaded = getattr(importlib.import_module(module_name), attr)
    if not isinstance(loaded, type) or not issubclass(loaded, BaseModel):
        raise TypeError(f"Type reference must be a Pydantic model: {value!r}")
    return loaded
