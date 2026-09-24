"""Serializable configuration for LocalContextRegistryConfig."""

from __future__ import annotations

from pydantic import BaseModel

from ....base.context import LogBookToolMode


class LocalContextRegistryConfig(BaseModel):
    """Configuration options for ``LocalContextRegistry``."""
    user_id: str
    session_id: str
    base_path: str
    tool_mode: LogBookToolMode = LogBookToolMode.READ_ONLY
