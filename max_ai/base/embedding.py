"""Contract for turning text into vectors (semantic search in memory,
knowledge and past conversations).

``embed`` validates, batches and caches; implementations only write
``_embed`` for one batch. A text is embedded once per model: later
searches reuse its vector from the in-memory cache.
"""

from __future__ import annotations

import hashlib
import typing as t
from abc import ABC, abstractmethod
from collections import OrderedDict

from pydantic import BaseModel

from ..errors.embeddings import EmbeddingError
from .component import ComponentBase


class _DefaultEmbedding:
    """Sentinel: caller didn't pass ``embedding``, so use the framework's
    default (``FastEmbedEmbedding``). Distinct from ``None``, which means
    the caller explicitly wants no semantic search."""

    def __repr__(self) -> str:
        """
        Return a readable representation of this embedding provider.

        Returns
        -------
        str
            The resulting text value.
        """
        return "DEFAULT_EMBEDDING"


DEFAULT_EMBEDDING = _DefaultEmbedding()


class CoreEmbedding(ComponentBase[BaseModel], ABC):
    """
    Define the interface for converting text into vectors.
    """
    component_type = "embedding"

    batch_size: t.ClassVar[int] = 64
    cache_size: t.ClassVar[int] = 10_000

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Names the model; vectors of different models are never mixed."""

    async def embed(self, texts: t.Sequence[str]) -> list[list[float]]:
        """One vector per text, in order."""
        if isinstance(texts, str) or any(not isinstance(text, str) for text in texts):
            raise EmbeddingError.invalid_texts()
        cache = self._vectors()
        keys = [self._key(text) for text in texts]
        missing = list(dict.fromkeys(text for text, key in zip(texts, keys) if key not in cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            for text, vector in zip(batch, await self._embed(batch), strict=True):
                cache[self._key(text)] = [float(value) for value in vector]
        vectors = [cache[key] for key in keys]
        for key in keys:
            cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        """
        Convert one text string into an embedding vector.

        Parameters
        ----------
        text : str
            Text to validate or embed.

        Returns
        -------
        list[float]
            The resulting list.
        """
        if not isinstance(text, str):
            raise EmbeddingError.invalid_text()
        return (await self.embed([text]))[0]

    @abstractmethod
    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Vectors for one batch of texts (never empty, never cached)."""

    # -------- CACHE -----------------------------------------------------------
    def _vectors(self) -> OrderedDict[str, list[float]]:
        """
        Return the process-local cache of computed embedding vectors.

        Returns
        -------
        OrderedDict[str, list[float]]
            The ordered cache of text embeddings.
        """
        if not hasattr(self, "_cache"):
            self._cache: OrderedDict[str, list[float]] = OrderedDict()
        return self._cache

    def _key(self, text: str) -> str:
        """
        Build the cache key for a text string.

        Parameters
        ----------
        text : str
            Text to validate or embed.

        Returns
        -------
        str
            The resulting text value.
        """
        return hashlib.sha256(f"{self.model_id}\0{text}".encode()).hexdigest()


__all__ = ["CoreEmbedding"]
