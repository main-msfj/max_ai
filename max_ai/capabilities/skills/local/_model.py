"""Serializable configuration for LocalSkillRegistryConfig."""

from __future__ import annotations

from pydantic import BaseModel


class LocalSkillRegistryConfig(BaseModel):
    source: str
    skills: list[str]
