"""Serializable configuration for MongoDBSessionStore."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MongoDBSessionStoreConfig(BaseModel):
    """Configuration options for ``MongoDBSessionStore``."""
    database: str = "max_ai"
    collection: str = "sessions"
    uri_env: str = Field(default="MONGODB_URI", description="Env var holding the URI (never the URI).")
    server_selection_timeout_ms: int = Field(default=5000, gt=0)
