"""Serializable configuration for SQLiteMemoryRegistry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ....base.memory import MemoryToolMode


class SQLiteMemoryRegistryConfig(BaseModel):
    base_path: str
    db_name: str = Field(default="memory.sqlite3", min_length=1)
    user_id: str | None = None
    session_id: str | None = None
    tool_mode: MemoryToolMode = MemoryToolMode.FULL
    context_days: int | None = Field(default=30, ge=0)
    search_limit: int = Field(default=20, ge=1, le=100)
    embedding: dict[str, Any] | None = None  # a serialized CoreEmbedding
