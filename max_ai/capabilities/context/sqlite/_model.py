"""Serializable configuration for SQLiteContextRegistryConfig."""

from __future__ import annotations

from pydantic import BaseModel

from ....base.context import LogBookToolMode


class SQLiteContextRegistryConfig(BaseModel):
    """Configuration options for ``SQLiteContextRegistry``."""
    user_id: str
    session_id: str
    base_path: str
    tool_mode: LogBookToolMode = LogBookToolMode.READ_ONLY
    db_name: str = "context.sqlite3"
