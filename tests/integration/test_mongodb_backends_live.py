"""Live backend checks with MAXAI_TEST_MONGODB_URI set explicitly."""

import asyncio
import os
from uuid import uuid4

import pytest

from max_ai.capabilities.knowledge.mongodb import MongoDBKnowledgeRegistry
from max_ai.capabilities.memory.mongodb import MongoDBMemoryRegistry
from max_ai.core import KnowledgeBlock

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not os.getenv("MAXAI_TEST_MONGODB_URI"), reason="Set MAXAI_TEST_MONGODB_URI to test MongoDB",
)]


async def test_real_mongodb_isolation_replacement_search_and_reconnect():
    from pymongo import AsyncMongoClient

    database = "maxai_test_" + uuid4().hex
    options = {"database": database, "uri_env": "MAXAI_TEST_MONGODB_URI"}
    current = MongoDBMemoryRegistry("alice", "current", **options)
    past = MongoDBMemoryRegistry("alice", "past", **options)
    other = MongoDBMemoryRegistry("bob", "past", **options)
    docs = MongoDBKnowledgeRegistry("manual", "Search", **options)
    other_docs = MongoDBKnowledgeRegistry("private", "Search", **options)
    registries = [current, past, other, docs, other_docs]
    cleanup = AsyncMongoClient(os.environ["MAXAI_TEST_MONGODB_URI"])
    try:
        await current.create_or_update("project", "python current")
        await past.create_or_update("project", "python historical")
        await other.create_or_update("project", "python private")
        results = await current.search_memory("python")
        assert [(r.session_id, r.memory) for r in results] == [("past", "python historical")]
        assert results[0].updated.tzinfo is not None
        await asyncio.gather(*(current.create_or_update("race", str(i)) for i in range(10)))
        assert (await current.list_category()).count("race") == 1
        await past.create_or_update("project", "rust historical")
        assert await current.search_memory("python") == []
        assert "deleted" in await past.delete_memory("project")
        assert len(await current.get_context()) == 2
        await current.create_or_update("Project", "case-sensitive")
        assert "Project" in await current.list_category()
        await current.disconnect()
        assert len(await current.get_context()) == 3
        await docs.upsert_block("intro", KnowledgeBlock(content="Python tutorial", metadata={"page": 1}))
        await other_docs.upsert_block("intro", KnowledgeBlock(content="Python private"))
        assert [b.content for b in await docs.search("python")] == ["Python tutorial"]
        await docs.upsert_block("intro", KnowledgeBlock(content="Rust tutorial"))
        assert await docs.search("python") == []
        assert (await docs.search("rust"))[0].metadata == {}
        assert await docs.delete_block("intro")
        assert await docs.search("rust") == []
    finally:
        for registry in registries:
            await registry.disconnect()
        await cleanup.drop_database(database)
        await cleanup.close()
