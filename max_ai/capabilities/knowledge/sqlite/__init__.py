"""Knowledge in a local SQLite file, searched by meaning (stored vectors)."""

from ._model import SQLiteKnowledgeRegistryConfig
from ._registry import SQLiteKnowledgeRegistry

__all__ = ["SQLiteKnowledgeRegistry", "SQLiteKnowledgeRegistryConfig"]
