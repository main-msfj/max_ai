"""Embeddings: the contract lives in ``max_ai.base.embedding.CoreEmbedding``.

``FastEmbedEmbedding`` (local, the default) and ``OpenAIEmbedding`` (API,
for serverless). ``rank`` orders items by cosine similarity.
"""

from ...base.embedding import CoreEmbedding
from .fastembed import FastEmbedEmbedding, FastEmbedEmbeddingConfig
from .lightweight import (
    DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL,
    get_lightweight_embedding,
    get_lightweight_embeddings,
)
from .openai import OpenAIEmbedding, OpenAIEmbeddingConfig
from .similarity import cosine_similarity, rank
from .vectors import pack_vector, unpack_vector

__all__ = [
    "CoreEmbedding",
    "DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL",
    "FastEmbedEmbedding",
    "FastEmbedEmbeddingConfig",
    "OpenAIEmbedding",
    "OpenAIEmbeddingConfig",
    "cosine_similarity",
    "get_lightweight_embedding",
    "get_lightweight_embeddings",
    "pack_vector",
    "rank",
    "unpack_vector",
]
