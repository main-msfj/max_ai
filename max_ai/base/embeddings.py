"""Lightweight embedding helpers shared by local capabilities."""

from __future__ import annotations

import functools
import typing as t

from ..errors.embeddings import EmbeddingError


DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@functools.lru_cache(maxsize=4)
def _get_text_embedding_model(model_name: str) -> t.Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise EmbeddingError.dependency_missing() from exc

    return TextEmbedding(model_name=model_name)


def get_lightweight_embedding(
    text: str,
    *,
    model_name: str = DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL,
) -> list[float]:
    """Return a small local embedding vector for a single text string."""
    if not isinstance(text, str):  # type: ignore[unreachable]
        raise EmbeddingError.invalid_text()

    model = _get_text_embedding_model(model_name)
    vectors = model.embed([text])
    try:
        vector = next(iter(vectors))
    except StopIteration:
        return []

    if hasattr(vector, "tolist"):
        return list(vector.tolist())
    return [float(value) for value in vector]


def get_lightweight_embeddings(
    texts: t.Sequence[str],
    *,
    model_name: str = DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL,
) -> list[list[float]]:
    """Return local embedding vectors for multiple texts."""
    for text in texts:
        if not isinstance(text, str):  # type: ignore[unreachable]
            raise EmbeddingError.invalid_texts()

    model = _get_text_embedding_model(model_name)
    vectors = model.embed(list(texts))
    results: list[list[float]] = []
    for vector in vectors:
        if hasattr(vector, "tolist"):
            results.append(list(vector.tolist()))
        else:
            results.append([float(value) for value in vector])
    return results
