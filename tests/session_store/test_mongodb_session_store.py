"""MongoDBSessionStore: the same contract as LocalSessionStore, in MongoDB.

Most tests use an in-memory stand-in for the collection; the last one runs
against a real MongoDB when MONGODB_URI is set.
"""

from __future__ import annotations

import copy
import os
import uuid
from types import SimpleNamespace

import pytest

from max_ai.capabilities.session_store import MongoDBSessionStore
from max_ai.core.messages import AssistantMessage, ToolCall

from .test_local_session_store import conversation


class FakeCollection:
    """Enough of PyMongo's async collection for the store."""

    def __init__(self):
        self.docs: list[dict] = []

    def _match(self, query):
        return [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]

    @staticmethod
    def _project(doc, projection):
        return {k: copy.deepcopy(v) for k, v in doc.items() if projection.get(k)}

    async def find_one(self, query, projection):
        found = self._match(query)
        return self._project(found[0], projection) if found else None

    async def replace_one(self, query, doc, upsert):
        for old in self._match(query):
            self.docs.remove(old)
        self.docs.append(copy.deepcopy(doc))

    def find(self, query, projection):
        async def cursor():
            for doc in self._match(query):
                yield self._project(doc, projection)
        return cursor()

    async def delete_one(self, query):
        found = self._match(query)
        if found:
            self.docs.remove(found[0])
        return SimpleNamespace(deleted_count=len(found[:1]))


def connected_store() -> tuple[MongoDBSessionStore, FakeCollection]:
    store, collection = MongoDBSessionStore(), FakeCollection()
    store._collection, store._connected = collection, True
    return store, collection


async def test_round_trip_keeps_the_whole_conversation():
    store, collection = connected_store()
    ctx = conversation()
    info = await store.save(ctx)
    assert (info.title, info.message_count, info.compactions) == ("arma un scraper", 3, 2)
    loaded = await store.load("alice", "s1")
    assert loaded.model_dump() == ctx.model_dump()
    assert isinstance(collection.docs[0]["context"], str)  # "$ref"-style keys stay safe


async def test_saving_again_replaces_and_users_are_isolated():
    store, collection = connected_store()
    await store.save(conversation("alice", "s1"))
    await store.save(conversation("alice", "s1", first="versión nueva"))
    await store.save(conversation("bob", "s1", first="otra cosa"))
    assert len(collection.docs) == 2
    assert [s.title for s in await store.list_sessions("alice")] == ["versión nueva"]
    assert (await store.load("bob", "s1")).messages[1].text() == "otra cosa"
    assert await store.load("alice", "nope") is None


async def test_a_document_under_the_wrong_key_is_refused():
    store, collection = connected_store()
    await store.save(conversation("alice", "s1"))
    collection.docs[0]["user_id"] = "bob"  # tampered row
    with pytest.raises(ValueError, match="does not belong"):
        await store.load("bob", "s1")


async def test_list_is_newest_first_and_delete():
    store, _ = connected_store()
    for n in range(3):
        await store.save(conversation(session=f"s{n}", first=f"tarea {n}"))
    assert [s.session_id for s in await store.list_sessions("alice")] == ["s2", "s1", "s0"]
    assert len(await store.list_sessions("alice", limit=2)) == 2
    assert await store.delete("alice", "s1") is True
    assert await store.delete("alice", "s1") is False
    assert [s.session_id for s in await store.list_sessions("alice")] == ["s2", "s0"]


async def test_unsafe_ids_never_reach_the_database():
    store, collection = connected_store()
    with pytest.raises(ValueError, match="Invalid"):
        await store.load("alice", {"$ne": None})  # query injection attempt
    assert collection.docs == []


def test_config_stores_the_env_var_name_never_the_uri(monkeypatch):
    monkeypatch.setenv("MY_MONGO", "mongodb://user:secret@host/")
    store = MongoDBSessionStore(collection="chats", uri_env="MY_MONGO")
    stored = store.serialize().model_dump_json()
    assert "secret" not in stored
    back = MongoDBSessionStore.deserialize(stored)
    assert (back.config.collection, back.config.uri_env) == ("chats", "MY_MONGO")


async def test_missing_uri_is_reported(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with pytest.raises(ValueError, match="MONGODB_URI"):
        await MongoDBSessionStore().load("alice", "s1")


@pytest.mark.skipif(not os.getenv("MONGODB_URI"), reason="needs MONGODB_URI")
async def test_against_a_real_mongodb():
    collection = f"sessions_test_{uuid.uuid4().hex[:8]}"
    ctx = conversation()
    ctx.messages.append(AssistantMessage(source="llm", content="", tool_calls=[
        ToolCall(id="t1", tool_name="schema", parameters={"$ref": "#/defs/x", "a.b": 1})]))
    first, second = MongoDBSessionStore(collection=collection), MongoDBSessionStore(collection=collection)
    async with first, second:  # two processes sharing one database
        try:
            await first.save(ctx)
            loaded = await second.load("alice", "s1")
            assert loaded.model_dump() == ctx.model_dump()
            assert [s.session_id for s in await second.list_sessions("alice")] == ["s1"]
            assert await second.delete("alice", "s1") is True
        finally:
            await first._collection.drop()
