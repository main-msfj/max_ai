"""Serializable configuration for LocalKnowledgeRegistryConfig."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ....base.knowledge import KnowledgeToolMode


class LocalKnowledgeRegistryConfig(BaseModel):
    """Configuration options for ``LocalKnowledgeRegistry``."""
    name: str
    description: str
    base_path: str
    tool_mode: KnowledgeToolMode = KnowledgeToolMode.FULL
    embedding: dict[str, Any] | None = None  # a serialized CoreEmbedding
