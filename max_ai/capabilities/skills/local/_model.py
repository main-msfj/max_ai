"""Serializable configuration for LocalSkillRegistryConfig."""

from __future__ import annotations

from pydantic import BaseModel


class LocalSkillRegistryConfig(BaseModel):
    """Configuration options for ``LocalSkillRegistry``."""
    source: str
    skills: list[str]
