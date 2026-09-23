"""Serializable configuration for MongoDBMemoryRegistry."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ....base.memory import MemoryToolMode


class MongoDBMemoryRegistryConfig(BaseModel):
    user_id: str
    session_id: str
    tool_mode: MemoryToolMode = MemoryToolMode.FULL
    context_days: int | None = Field(default=30, ge=0)
    search_limit: int = Field(default=20, ge=1, le=100, strict=True)

    # Connection. Stores the env var name, never the URI or credentials.
    database: str = Field(default="max_ai", min_length=1)
    collection: str = Field(default="memory", min_length=1)
    uri_env: str = Field(default="MONGODB_URI", min_length=1)
    server_selection_timeout_ms: int = Field(default=5000, gt=0, strict=True)
