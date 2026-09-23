"""MongoDB adapter contracts without requiring an external database."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from max_ai.base.memory import MemoryRecord
from max_ai.capabilities.knowledge.mongodb import MongoDBKnowledgeRegistry
from max_ai.capabilities.memory.mongodb import MongoDBMemoryRegistry
from max_ai.core import KnowledgeBlock


def connected(registry, documents=()):
    collection = MagicMock()
    cursor = MagicMock()
    cursor.__aiter__.return_value = list(documents)
    cursor.sort.return_value = cursor
    cursor.limit.return_value = cursor
    collection.find.return_value = cursor
    registry._collection = collection
    registry._connected = True
    return collection, cursor


async def test_memory_queries_always_scope_user_and_session():
    memory = MongoDBMemoryRegistry("alice", "current", search_limit=3)
    doc = dict(category="project", memory="python", updated=datetime.now(UTC), session_id="old")
    collection, cursor = connected(memory, [doc])
    assert (await memory.get_context())[0].memory == "python"
    assert collection.find.call_args.args[0] == {"user_id": "alice", "session_id": "current"}
    result = await memory.search_memory("python")
    assert result[0].session_id == "old"
    assert collection.find.call_args.args[0] == {
        "user_id": "alice", "session_id": {"$ne": "current"}, "$text": {"$search": "python"},
    }
    cursor.limit.assert_called_with(3)


async def test_memory_upsert_and_delete_keep_exact_category_identity():
    memory = MongoDBMemoryRegistry("alice", "current")
    collection, _ = connected(memory)
    collection.update_one = AsyncMock(return_value=SimpleNamespace(upserted_id="new"))
    assert await memory.create_or_update(" Project ", "first") == "New memory category created: Project"
    args = collection.update_one.call_args
    assert args.args[0] == {"user_id": "alice", "session_id": "current", "category": "Project"}
    assert args.args[1]["$set"]["updated"].tzinfo is not None
    assert args.kwargs["collation"] == {"locale": "simple"}
    collection.update_one.return_value = SimpleNamespace(upserted_id=None)
    assert await memory.create_or_update("Project", "replacement") == "Memory updated: Project"
    assert collection.update_one.call_args.args[1]["$set"]["memory"] == "replacement"
    collection.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=1))
    assert await memory.delete_memory("Project") == "Memory deleted: Project"
    assert collection.delete_one.call_args.args[0] == args.args[0]


async def test_concurrent_insert_retries_only_the_same_identity():
    from pymongo.errors import DuplicateKeyError

    memory = MongoDBMemoryRegistry("alice", "current")
    collection, _ = connected(memory)
    collection.update_one = AsyncMock(side_effect=[
        DuplicateKeyError("race"), SimpleNamespace(upserted_id=None, matched_count=1),
    ])
    assert not await memory._write_memory(MemoryRecord(category="project", memory="new"))
    first, retry = collection.update_one.call_args_list
    assert first.args == retry.args
    assert first.kwargs["upsert"] is True
    assert "upsert" not in retry.kwargs


async def test_knowledge_search_and_ingestion_are_source_scoped():
    knowledge = MongoDBKnowledgeRegistry("manual", "Search the manual")
    collection, cursor = connected(knowledge, [dict(content="Python guide", tokens=2, metadata={"page": 1})])
    result = await knowledge.search(" Python ", limit=2)
    assert result == [KnowledgeBlock(content="Python guide", tokens=2, metadata={"page": 1})]
    assert collection.find.call_args.args[0] == {"source": "manual", "$text": {"$search": "Python"}}
    cursor.limit.assert_called_with(2)
    collection.update_one = AsyncMock(return_value=SimpleNamespace(upserted_id="new"))
    assert await knowledge.upsert_block(" intro ", result[0])
    assert collection.update_one.call_args.args[0] == {"source": "manual", "block_id": "intro"}
    collection.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=0))
    assert not await knowledge.delete_block("intro")
    assert collection.delete_one.call_args.args[0] == {"source": "manual", "block_id": "intro"}
    assert [tool.name for tool in knowledge.tools] == ["search_manual"]


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5])
async def test_knowledge_rejects_unbounded_or_invalid_limits(limit):
    knowledge = MongoDBKnowledgeRegistry("manual", "Search")
    with pytest.raises(ValueError, match="limit"):
        await knowledge.search("Python", limit=limit)
    assert not knowledge._connected


@pytest.mark.parametrize("registry", [
    MongoDBMemoryRegistry("alice", "session", database="custom", collection="mem", uri_env="TEST_MONGO_URI"),
    MongoDBKnowledgeRegistry("manual", "Search", database="custom", collection="docs", uri_env="TEST_MONGO_URI"),
])
def test_component_roundtrip_does_not_serialize_credentials(registry, monkeypatch):
    monkeypatch.setenv("TEST_MONGO_URI", "mongodb://secret:password@host/")
    config = registry.serialize()
    assert "password" not in config.model_dump_json()
    restored = type(registry).deserialize(config)
    assert restored._to_config() == registry._to_config()
    assert not restored._connected


async def test_connection_failure_closes_client_and_allows_retry(monkeypatch):
    client = MagicMock()
    client.admin.command = AsyncMock()
    client.close = AsyncMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("pymongo.AsyncMongoClient", factory)
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    memory = MongoDBMemoryRegistry("alice", "session")
    memory._create_indexes = AsyncMock(side_effect=RuntimeError("index failed"))
    with pytest.raises(RuntimeError, match="index failed"):
        await memory.list_category()
    client.close.assert_awaited_once()
    assert not memory._connected
    assert memory._mongo_client is None
    memory._create_indexes.side_effect = None
    await memory._ensure_connected()
    assert memory._connected
    await memory.disconnect()
    assert not memory._connected
    await memory._ensure_connected()
    assert factory.call_count == 3
    await memory.disconnect()


async def test_missing_uri_is_reported_before_connecting(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with pytest.raises(ValueError, match="MONGODB_URI"):
        await MongoDBMemoryRegistry("alice", "session").connect()


def test_memory_tool_describes_text_retrieval():
    tool = next(t for t in MongoDBMemoryRegistry("alice", "s").tools if t.name == "search_memory")
    assert "words or phrases" in tool.description


async def test_concurrent_connections_own_one_client(monkeypatch):
    client = MagicMock()
    client.admin.command = AsyncMock()
    client.close = AsyncMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("pymongo.AsyncMongoClient", factory)
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    memory = MongoDBMemoryRegistry("alice", "session")
    memory._create_indexes = AsyncMock()
    await asyncio.gather(*(memory.connect() for _ in range(5)))
    factory.assert_called_once()
    await memory.disconnect()
    client.close.assert_awaited_once()
