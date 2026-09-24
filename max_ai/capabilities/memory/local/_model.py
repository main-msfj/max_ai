"""Serializable configuration for LocalMemoryRegistryConfig."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ....base.memory import MemoryToolMode


class LocalMemoryRegistryConfig(BaseModel):
    """Configuration options for ``LocalMemoryRegistry``."""
    base_path: str
    user_id: str | None = None
    session_id: str | None = None
    tool_mode: MemoryToolMode = MemoryToolMode.FULL
    context_days: int | None = 30
    embedding: dict[str, Any] | None = None  # a serialized CoreEmbedding
