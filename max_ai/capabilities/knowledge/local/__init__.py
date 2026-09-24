"""Filesystem-backed knowledge registry, available through max_ai.capabilities.knowledge.local."""

from ._model import LocalKnowledgeRegistryConfig
from ._registry import LocalKnowledgeRegistry

__all__ = ["LocalKnowledgeRegistry", "LocalKnowledgeRegistryConfig"]
