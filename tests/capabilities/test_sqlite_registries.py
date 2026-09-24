"""SQLite memory and knowledge: stored vectors, scoping and search."""

from __future__ import annotations

import asyncio
import importlib.util

import pytest
from pydantic import BaseModel

from max_ai.base.embedding import CoreEmbedding
from max_ai.base.memory import MemoryToolMode
from max_ai.capabilities.knowledge.sqlite import SQLiteKnowledgeRegistry
from max_ai.capabilities.memory.sqlite import SQLiteMemoryRegistry
from max_ai.core import KnowledgeBlock
from max_ai.core.embeddings import FastEmbedEmbedding

VOCABULARY = ["cat", "dog", "food", "vegetarian", "meat", "short", "answers", "refund", "shipping"]


class NoConfig(BaseModel):
    pass


class KeywordEmbedding(CoreEmbedding):
    """One dimension per vocabulary word; records every text it embeds."""

    component_schema = NoConfig

    def __init__(self, name: str = "keywords") -> None:
        self.name = name
        self.embedded: list[str] = []

    @property
    def model_id(self) -> str:
        return self.name

    def _to_config(self) -> NoConfig:
        return NoConfig()

    @classmethod
    def _from_config(cls, config: NoConfig) -> "KeywordEmbedding":
        return cls()

    async def _embed(self, texts):
        self.embedded.extend(texts)
        return [[float(word in text.lower()) for word in VOCABULARY] for text in texts]


# -------- MEMORY -----------------------------------------------------------
def memory(tmp_path, **options) -> SQLiteMemoryRegistry:
    return SQLiteMemoryRegistry(base_path=tmp_path, **options)


async def test_memory_categories_per_user_and_session(tmp_path):
    registry = memory(tmp_path)
    ana, beto = registry.bind("ana", "s1"), registry.bind("beto", "s1")
    assert "created" in await ana.create_or_update("diet", "vegetarian")
    assert "updated" in await ana.create_or_update("diet", "vegan")
    await beto.create_or_update("diet", "eats everything")

    assert [(r.category, r.memory) for r in await ana.get_context()] == [("diet", "vegan")]
    assert [r.memory for r in await beto.get_context()] == ["eats everything"]
    assert await ana.delete_memory("diet") == "Memory deleted: diet"
    assert await ana.get_context() == [] and await beto.list_category() == ["diet"]
    await registry.disconnect()


async def test_memory_search_by_words_without_an_embedding(tmp_path):
    registry = memory(tmp_path)
    await registry.bind("ana", "s1").create_or_update("diet", "She is vegetarian")
    await registry.bind("ana", "s2").create_or_update("pets", "Has a vegetarian dog")
    found = await registry.bind("ana", "s2").search_memory("vegetarian")
    assert [(r.session_id, r.category) for r in found] == [("s1", "diet")]  # never its own session
    await registry.disconnect()


async def test_memory_vectors_are_stored_and_reused_after_a_restart(tmp_path):
    first = memory(tmp_path, embedding=KeywordEmbedding())
    await first.bind("ana", "s1").create_or_update("diet", "She is vegetarian, no meat")
    await first.bind("ana", "s1").create_or_update("style", "Prefers short answers")
    await first.disconnect()

    embedding = KeywordEmbedding()  # a new process: empty cache
    second = memory(tmp_path, embedding=embedding)
    found = await second.bind("ana", "s2").search_memory("does she eat meat?")
    assert [r.category for r in found] == ["diet"]
    assert embedding.embedded == ["does she eat meat?"]  # only the query
    await second.disconnect()


async def test_memory_rows_without_vectors_or_from_another_model_are_embedded_once(tmp_path):
    plain = memory(tmp_path)
    await plain.bind("ana", "s1").create_or_update("diet", "She is vegetarian")
    await plain.disconnect()

    embedding = KeywordEmbedding("model-b")
    registry = memory(tmp_path, embedding=embedding)
    searcher = registry.bind("ana", "s2")
    assert [r.category for r in await searcher.search_memory("vegetarian food")] == ["diet"]
    assert embedding.embedded == ["diet: She is vegetarian", "vegetarian food"]
    await searcher.search_memory("meat")
    assert embedding.embedded[-1] == "meat"  # the stored vector was saved and reused
    await registry.disconnect()


async def test_concurrent_writes_from_many_runs(tmp_path):
    registry = memory(tmp_path)
    await asyncio.gather(*(
        registry.bind(f"user{i % 5}", f"s{i}").create_or_update("note", f"note {i}") for i in range(50)
    ))
    counts = [len(await registry.bind(f"user{u}", f"s{u}").get_context()) for u in range(5)]
    assert counts == [1, 1, 1, 1, 1]
    await registry.disconnect()


def test_memory_serializes_without_its_scope(tmp_path):
    registry = memory(tmp_path, tool_mode=MemoryToolMode.READ_ONLY, embedding=FastEmbedEmbedding())
    back = SQLiteMemoryRegistry.deserialize(registry.serialize().model_dump_json())
    assert back.path == tmp_path.resolve() / "memory.sqlite3" and back.user_id is None
    assert isinstance(back.embedding, FastEmbedEmbedding) and back.tool_mode == MemoryToolMode.READ_ONLY


# -------- KNOWLEDGE -----------------------------------------------------------
def knowledge(tmp_path, name="support", embedding=None) -> SQLiteKnowledgeRegistry:
    return SQLiteKnowledgeRegistry(name=name, description="d", base_path=tmp_path,
                                   embedding=embedding or KeywordEmbedding())


async def test_knowledge_blocks_are_embedded_once_when_written(tmp_path):
    writer = knowledge(tmp_path)
    assert await writer.upsert_block("refunds", KnowledgeBlock(content="Refund policy: 30 days",
                                                               metadata={"source": "faq.md"}))
    assert not await writer.upsert_block("refunds", KnowledgeBlock(content="Refund policy: 60 days"))
    await writer.upsert_block("shipping", KnowledgeBlock(content="Shipping takes 3 days"))
    await writer.disconnect()

    embedding = KeywordEmbedding()
    reader = knowledge(tmp_path, embedding=embedding)
    [block] = await reader.search("how do I get a refund?", limit=3)
    assert (block.content, block.metadata) == ("Refund policy: 60 days", {})
    assert embedding.embedded == ["how do I get a refund?"]  # blocks came from the database

    assert await reader.delete_block("refunds") and not await reader.delete_block("refunds")
    assert await reader.search("refund") == []
    await reader.disconnect()


async def test_knowledge_sources_share_a_file_but_not_their_blocks(tmp_path):
    support, pets = knowledge(tmp_path, "support"), knowledge(tmp_path, "pets")
    await support.upsert_block("a", KnowledgeBlock(content="Refunds in 30 days"))
    await pets.upsert_block("b", KnowledgeBlock(content="Cats and dogs"))
    assert [b.content for b in await pets.search("refund cat")] == ["Cats and dogs"]
    assert await support.search("") == []
    with pytest.raises(ValueError, match="limit"):
        await support.search("x", limit=0)
    await support.disconnect()
    await pets.disconnect()


async def test_changing_the_embedding_model_re_embeds_the_blocks(tmp_path):
    first = knowledge(tmp_path, embedding=KeywordEmbedding("model-a"))
    await first.upsert_block("a", KnowledgeBlock(content="Shipping takes 3 days"))
    await first.disconnect()
    embedding = KeywordEmbedding("model-b")
    second = knowledge(tmp_path, embedding=embedding)
    assert await second.search("shipping")
    assert embedding.embedded == ["Shipping takes 3 days", "shipping"]
    await second.disconnect()


@pytest.mark.skipif(importlib.util.find_spec("fastembed") is None, reason="needs maxai[embeddings]")
async def test_default_model_finds_spanish_documents_from_english_questions(tmp_path):
    docs = SQLiteKnowledgeRegistry(name="soporte", description="d", base_path=tmp_path)
    await docs.upsert_block("devoluciones", KnowledgeBlock(content="Puedes devolver un producto dentro de 30 días."))
    await docs.upsert_block("envios", KnowledgeBlock(content="Los envíos tardan de 3 a 5 días hábiles."))
    [best, *_] = await docs.search("how long does delivery take?")
    assert "envíos" in best.content
    await docs.disconnect()
