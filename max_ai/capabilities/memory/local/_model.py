"""Serializable configuration for LocalMemoryRegistryConfig."""

from __future__ import annotations

from pydantic import BaseModel

from ....base.memory import MemoryToolMode


class LocalMemoryRegistryConfig(BaseModel):
    user_id: str
    session_id: str
    base_path: str
    tool_mode: MemoryToolMode = MemoryToolMode.FULL
    context_days: int | None = 30
