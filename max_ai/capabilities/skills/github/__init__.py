"""GitHub-backed skill registry, available through max_ai.capabilities.skills.github."""

from ._model import GithubSkillRegistryConfig
from ._registry import GithubSkillRegistry

__all__ = ["GithubSkillRegistry", "GithubSkillRegistryConfig"]
