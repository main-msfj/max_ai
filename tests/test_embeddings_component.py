"""CoreEmbedding: batching, caching, and semantic search in knowledge and memory."""

from __future__ import annotations

import importlib.util
import json

import pytest
from pydantic import BaseModel

from max_ai.base.embedding import CoreEmbedding
from max_ai.capabilities.knowledge.local import LocalKnowledgeRegistry
from max_ai.capabilities.memory.local import LocalMemoryRegistry
from max_ai.core.embeddings import FastEmbedEmbedding, OpenAIEmbedding, cosine_similarity, rank
from max_ai.errors.embeddings import EmbeddingError

VOCABULARY = ["cat", "dog", "food", "vegetarian", "meat", "short", "answers"]


class NoConfig(BaseModel):
    pass


class KeywordEmbedding(CoreEmbedding):
    """One dimension per vocabulary word; counts what it really embeds."""

    component_schema = NoConfig
    batch_size = 2

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "keywords"

    def _to_config(self) -> NoConfig:
        return NoConfig()

    @classmethod
    def _from_config(cls, config: NoConfig) -> "KeywordEmbedding":
        return cls()

    async def _embed(self, texts):
        self.batches.append(list(texts))
        return [[float(word in text.lower()) for word in VOCABULARY] for text in texts]


# -------- CONTRACT -----------------------------------------------------------
async def test_texts_are_embedded_once_in_batches_and_in_order():
    embedding = KeywordEmbedding()
    first = await embedding.embed(["cat", "dog", "cat", "food"])
    assert first[0] == first[2] and first[0] != first[1]
    assert embedding.batches == [["cat", "dog"], ["food"]]  # batch_size 2, duplicates once

    await embedding.embed(["dog", "meat"])
    assert embedding.batches[-1] == ["meat"]  # "dog" came from the cache
    with pytest.raises(EmbeddingError):
        await embedding.embed("not a list")
    with pytest.raises(EmbeddingError):
        await embedding.embed(["ok", 3])


def test_rank_orders_by_similarity_and_drops_weak_matches():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0 and cosine_similarity([], [1]) == 0.0
    ranked = rank([1, 0], ["a", "b", "c"], [[1, 0], [0, 1], [1, 1]], limit=5, min_score=0.1)
    assert [item for _, item in ranked] == ["a", "c"]


def test_embeddings_serialize_without_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    stored = OpenAIEmbedding(dimensions=256).serialize().model_dump_json()
    assert "sk-secret" not in stored and '"api_key_env":"OPENAI_API_KEY"' in stored
    assert CoreEmbedding.deserialize(stored).config.dimensions == 256
    assert CoreEmbedding.deserialize(FastEmbedEmbedding().serialize()).model_name.endswith("multilingual-MiniLM-L12-v2")


# -------- KNOWLEDGE -----------------------------------------------------------
async def test_knowledge_search_ranks_by_meaning_with_one_batch_per_search(tmp_path):
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "pets.json").write_text(json.dumps([
        {"content": "Cats sleep a lot"}, {"content": "Dogs need walks"}, {"content": "Pizza recipe"},
    ]))
    embedding = KeywordEmbedding()
    registry = LocalKnowledgeRegistry(name="pets", description="d", base_path=tmp_path, embedding=embedding)
    async with registry:
        found = await registry.search("tell me about my cat", limit=2)
        assert [b.content for b in found] == ["Cats sleep a lot"]
        await registry.search("and the dog?")
    assert embedding.batches[-1] == ["and the dog?"]  # the blocks came from the cache

    stored = registry.serialize()
    assert stored.config["embedding"]["provider"].endswith("KeywordEmbedding")


# -------- MEMORY -----------------------------------------------------------
async def remember(memory: LocalMemoryRegistry, session: str, category: str, text: str) -> None:
    await memory.bind("ana", session).create_or_update(category, text)


async def test_memory_search_by_meaning_across_the_users_sessions(tmp_path):
    memory = LocalMemoryRegistry(base_path=tmp_path, embedding=KeywordEmbedding())
    await remember(memory, "s1", "diet", "She is vegetarian, no meat")
    await remember(memory, "s2", "style", "Prefers short answers")
    await remember(memory, "s3", "pets", "Has a dog")

    found = await memory.bind("ana", "s3").search_memory("does she eat meat?")
    assert [(r.session_id, r.category) for r in found] == [("s1", "diet")]  # never its own session

    words_only = LocalMemoryRegistry(base_path=tmp_path)  # no embedding: plain words
    assert await words_only.bind("ana", "s3").search_memory("eat meat") == []
    assert [r.category for r in await words_only.bind("ana", "s3").search_memory("vegetarian")] == ["diet"]


@pytest.mark.skipif(importlib.util.find_spec("fastembed") is None, reason="needs maxai[embeddings]")
async def test_the_default_model_understands_spanish_and_english(tmp_path):
    memory = LocalMemoryRegistry(base_path=tmp_path, embedding=FastEmbedEmbedding())
    await remember(memory, "s1", "dieta", "El usuario es vegetariano")
    await remember(memory, "s1", "estilo", "Prefiere respuestas cortas")
    found = await memory.bind("ana", "s2").search_memory("does the user eat meat?")
    assert found and found[0].category == "dieta"
