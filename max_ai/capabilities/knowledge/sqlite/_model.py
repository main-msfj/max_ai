"""Serializable configuration for SQLiteKnowledgeRegistry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ....base.knowledge import KnowledgeToolMode


class SQLiteKnowledgeRegistryConfig(BaseModel):
    """Configuration options for ``SQLiteKnowledgeRegistry``."""
    name: str
    description: str
    base_path: str
    db_name: str = Field(default="knowledge.sqlite3", min_length=1)
    tool_mode: KnowledgeToolMode = KnowledgeToolMode.FULL
    min_score: float = Field(default=0.2, ge=-1, le=1)
    embedding: dict[str, Any] | None = None  # a serialized CoreEmbedding
