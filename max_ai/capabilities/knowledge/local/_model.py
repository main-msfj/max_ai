"""Serializable configuration for LocalKnowledgeRegistryConfig."""

from __future__ import annotations

from pydantic import BaseModel

from ....base.knowledge import KnowledgeToolMode


class LocalKnowledgeRegistryConfig(BaseModel):
    name: str
    description: str
    base_path: str
    tool_mode: KnowledgeToolMode = KnowledgeToolMode.FULL
