"""MongoDB knowledge sources with indexed text search."""

from ._model import MongoDBKnowledgeRegistryConfig
from ._registry import MongoDBKnowledgeRegistry

__all__ = ["MongoDBKnowledgeRegistry", "MongoDBKnowledgeRegistryConfig"]
