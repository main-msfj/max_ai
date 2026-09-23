"""Serializable configuration for SQLiteMemoryRegistryConfig."""

from __future__ import annotations

from pydantic import BaseModel

from ....base.memory import MemoryToolMode


class SQLiteMemoryRegistryConfig(BaseModel):
    user_id: str
    base_path: str
    tool_mode: MemoryToolMode = MemoryToolMode.FULL
    db_name: str = "memory.sqlite3"
    merge_similarity_threshold: float = 0.85
    context_days: int | None = 30
