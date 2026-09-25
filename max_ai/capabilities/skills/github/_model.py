"""Serializable configuration for GithubSkillRegistry."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GithubSkillRegistryConfig(BaseModel):
    """Configuration options for ``GithubSkillRegistry``."""
    source: str
    skills: list[str]
    ref: str = "main"
    path: str = Field(default="", description="Folder inside the repo that holds the skills.")
    token_env: str | None = Field(
        default=None, description="Env var holding a GitHub token; never the token itself."
    )
