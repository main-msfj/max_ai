from __future__ import annotations

import pytest

from max_ai.base import embeddings
from max_ai.errors.embeddings import EmbeddingError


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeEmbeddingModel:
    def embed(self, texts):
        for index, text in enumerate(texts):
            yield FakeVector([float(index), float(len(text))])


def test_get_lightweight_embedding_returns_single_vector(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "_get_text_embedding_model",
        lambda model_name: FakeEmbeddingModel(),
    )

    result = embeddings.get_lightweight_embedding("hello")

    assert result == [0.0, 5.0]


def test_get_lightweight_embeddings_returns_batch(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "_get_text_embedding_model",
        lambda model_name: FakeEmbeddingModel(),
    )

    result = embeddings.get_lightweight_embeddings(["hi", "there"])

    assert result == [[0.0, 2.0], [1.0, 5.0]]


def test_get_lightweight_embedding_rejects_non_string():
    with pytest.raises(EmbeddingError, match="Embedding text must be a string") as exc_info:
        embeddings.get_lightweight_embedding(123)  # type: ignore[arg-type]
    assert exc_info.value.kind == "invalid_text"


def test_get_lightweight_embeddings_rejects_non_string_items():
    with pytest.raises(EmbeddingError, match="Embedding texts must contain only strings") as exc_info:
        embeddings.get_lightweight_embeddings(["ok", 123])  # type: ignore[list-item]
    assert exc_info.value.kind == "invalid_texts"
