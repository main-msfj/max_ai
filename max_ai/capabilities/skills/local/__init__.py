"""Local-filesystem skill registry, available through max_ai.capabilities.skills.local."""

from ._model import LocalSkillRegistryConfig
from ._registry import LocalSkillRegistry

__all__ = ["LocalSkillRegistry", "LocalSkillRegistryConfig"]
