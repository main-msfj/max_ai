"""Filesystem-backed memory registry, available through max_ai.capabilities.memory.to_delete.local."""

from ._model import LocalMemoryRegistryConfig
from ._registry import LocalMemoryRegistry

__all__ = ["LocalMemoryRegistry", "LocalMemoryRegistryConfig"]
