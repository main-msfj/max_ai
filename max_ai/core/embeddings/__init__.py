"""Lightweight local embeddings shared by the capabilities."""

from .lightweight import (
    DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL,
    get_lightweight_embedding,
    get_lightweight_embeddings,
)

__all__ = [
    "DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL",
    "get_lightweight_embedding",
    "get_lightweight_embeddings",
]
