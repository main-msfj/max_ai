"""Serializable configuration for MongoDBKnowledgeRegistry."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ....base.knowledge import KnowledgeToolMode


class MongoDBKnowledgeRegistryConfig(BaseModel):
    name: str
    description: str
    tool_mode: KnowledgeToolMode = KnowledgeToolMode.FULL

    # Connection. Stores the env var name, never the URI or credentials.
    database: str = Field(default="max_ai", min_length=1)
    collection: str = Field(default="knowledge", min_length=1)
    uri_env: str = Field(default="MONGODB_URI", min_length=1)
    server_selection_timeout_ms: int = Field(default=5000, gt=0, strict=True)
