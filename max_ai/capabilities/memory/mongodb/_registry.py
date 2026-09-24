"""Memory scoped to (user, session, category), stored in MongoDB."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from ....base.embedding import CoreEmbedding
from ....base.memory import (
    CoreMemoryRegistry,
    MemoryRecord,
    MemorySearchResult,
    MemoryToolMode,
)
from ....core.embeddings import rank
from ...mongodb_vector import MongoVectorIndex
from ._model import MongoDBMemoryRegistryConfig


class MongoDBMemoryRegistry(CoreMemoryRegistry):
    """Owns one async client; PyMongo is an optional dependency."""

    component_schema = MongoDBMemoryRegistryConfig
    component_type = "memory"
    component_provider_override = "max_ai.capabilities.memory.mongodb.MongoDBMemoryRegistry"

    def __init__(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        tool_mode: MemoryToolMode = MemoryToolMode.FULL,
        *,
        database: str = "max_ai",
        collection: str = "memory",
        uri_env: str = "MONGODB_URI",
        server_selection_timeout_ms: int = 5000,
        context_days: int | None = 30,
        search_limit: int = 20,
        embedding: CoreEmbedding | None = None,
    ) -> None:
        super().__init__(user_id, session_id, tool_mode, context_days=context_days,
                         embedding=embedding)
        self._vector_index = MongoVectorIndex(f"{collection}_vector", ["user_id", "session_id", "model_id"])
        self._mongo_config = MongoDBMemoryRegistryConfig(
            user_id=self.user_id, session_id=self.session_id, tool_mode=tool_mode,
            context_days=context_days, search_limit=search_limit,
            database=database, collection=collection, uri_env=uri_env,
            server_selection_timeout_ms=server_selection_timeout_ms,
        )
        self.search_limit = search_limit
        self._mongo_lock = asyncio.Lock()
        self._mongo_client: Any = None
        self._collection: Any = None

    def _to_config(self) -> MongoDBMemoryRegistryConfig:
        # Scope comes from the instance: a bound copy shares _mongo_config.
        embedding = self.embedding.serialize().model_dump(exclude_none=True) if self.embedding else None
        return self._mongo_config.model_copy(
            update={"user_id": self.user_id, "session_id": self.session_id, "embedding": embedding},
            deep=True,
        )

    @classmethod
    def _from_config(cls, config: MongoDBMemoryRegistryConfig) -> "MongoDBMemoryRegistry":
        embedding = CoreEmbedding.deserialize(config.embedding) if config.embedding else None
        return cls(**config.model_dump(exclude={"embedding"}), embedding=embedding)

    # -------- CONNECTION -----------------------------------------------------------
    async def connect(self) -> None:
        async with self._mongo_lock:
            if self._mongo_client is not None:
                return
            try:
                from pymongo import AsyncMongoClient
            except ImportError as error:
                raise ImportError(
                    "Install MongoDB support with: pip install 'maxai[mongodb]'"
                ) from error

            config = self._mongo_config
            uri = os.environ.get(config.uri_env)
            if not uri or not uri.strip():
                raise ValueError(f"Set {config.uri_env} to your MongoDB connection URI")
            client = AsyncMongoClient(
                uri, tz_aware=True,
                serverSelectionTimeoutMS=config.server_selection_timeout_ms,
            )
            try:
                await client.admin.command("ping")
                collection = client[config.database][config.collection]
                await self._create_indexes(collection)
            except BaseException:
                await client.close()
                raise
            self._mongo_client = client
            self._collection = collection
            self._connected = True

    async def disconnect(self) -> None:
        async with self._mongo_lock:
            client = self._mongo_client
            self._mongo_client = None
            self._collection = None
            self._connected = False
            if client is not None:
                await client.close()

    async def _create_indexes(self, collection: Any) -> None:
        await collection.create_index(
            [("user_id", 1), ("session_id", 1), ("category", 1)],
            unique=True, name="memory_identity", collation={"locale": "simple"},
        )
        await collection.create_index(
            [("user_id", 1), ("category", "text"), ("memory", "text")],
            name="memory_text", default_language="none", collation={"locale": "simple"},
        )

    # -------- STORAGE -----------------------------------------------------------
    def _scope(self) -> dict:
        return {"user_id": self.user_id, "session_id": self.session_id}

    async def _read_session(self) -> list[MemoryRecord]:
        await self._ensure_connected()
        cursor = self._collection.find(
            self._scope(), {"_id": 0, "vector": 0, "model_id": 0}, collation={"locale": "simple"},
        )
        return [MemoryRecord.model_validate(doc) async for doc in cursor]

    async def _write_memory(self, record: MemoryRecord) -> bool:
        """Replace the category atomically, including a concurrent first-insert race."""
        from pymongo.errors import DuplicateKeyError

        await self._ensure_connected()
        identity = {**self._scope(), "category": record.category}
        document = record.model_dump()
        if self.embedding is not None:  # stored once, reused by every search
            document["vector"] = await self.embedding.embed_one(_text(record.category, record.memory))
            document["model_id"] = self.embedding.model_id
        fields = {"$set": document}
        try:
            result = await self._collection.update_one(
                identity, fields, upsert=True, collation={"locale": "simple"},
            )
        except DuplicateKeyError:
            result = await self._collection.update_one(
                identity, fields, collation={"locale": "simple"},
            )
            if not result.matched_count:
                raise
        return result.upserted_id is not None

    async def _delete_memory(self, category: str) -> bool:
        await self._ensure_connected()
        result = await self._collection.delete_one(
            {**self._scope(), "category": category}, collation={"locale": "simple"},
        )
        return result.deleted_count > 0

    # Without a vector index, the user's most recent memories are ranked in Python.
    semantic_candidates: int = 500

    async def _search_memory(self, text: str) -> list[MemorySearchResult]:
        await self._ensure_connected()
        if self.embedding is not None:
            return await self._search_by_meaning(text)
        cursor = self._collection.find(
            {"user_id": self.user_id, "session_id": {"$ne": self.session_id},
             "$text": {"$search": text}},
            {"_id": 0, "score": {"$meta": "textScore"}},
            collation={"locale": "simple"},
        ).sort([("score", {"$meta": "textScore"}), ("updated", -1)]).limit(self.search_limit)
        return [MemorySearchResult.model_validate(doc) async for doc in cursor]

    async def _search_by_meaning(self, text: str) -> list[MemorySearchResult]:
        """Other sessions of this user by meaning: $vectorSearch when the
        server has it, else the most recent ones ranked in Python."""
        assert self.embedding is not None
        model_id = self.embedding.model_id
        await self._embed_missing()
        query = await self.embedding.embed_one(text)
        scope = {"user_id": self.user_id, "session_id": {"$ne": self.session_id}, "model_id": model_id}
        fields = {"category": 1, "memory": 1, "updated": 1, "session_id": 1}
        found = await self._vector_index.search(
            self._collection, query, filter=scope, limit=self.search_limit,
            min_score=self.semantic_min_score, projection=fields,
        )
        if found is None:
            cursor = self._collection.find(scope, {**fields, "vector": 1, "_id": 0}).sort(
                "updated", -1).limit(self.semantic_candidates)
            docs = [doc async for doc in cursor]
            ranked = rank(query, docs, [d["vector"] for d in docs],
                          limit=self.search_limit, min_score=self.semantic_min_score)
            found = [doc for _, doc in ranked]
        return [MemorySearchResult.model_validate(doc) for doc in found]

    async def _embed_missing(self) -> None:
        """This user's memories written before the embedding (or with another
        model) get their vector once, and keep it."""
        assert self.embedding is not None
        model_id = self.embedding.model_id
        cursor = self._collection.find(
            {"user_id": self.user_id,
             "$or": [{"model_id": {"$ne": model_id}}, {"vector": {"$exists": False}}]},
            {"_id": 0, "session_id": 1, "category": 1, "memory": 1},
        )
        stale = [doc async for doc in cursor]
        texts = [_text(d["category"], d["memory"]) for d in stale]
        for doc, vector in zip(stale, await self.embedding.embed(texts)):
            await self._collection.update_one(
                {"user_id": self.user_id, "session_id": doc["session_id"], "category": doc["category"]},
                {"$set": {"vector": vector, "model_id": model_id}},
            )

    def as_tools(self):
        tools = super().as_tools()
        for tool in tools:
            if tool.name == "search_memory":
                tool.description = (
                    "Search words or phrases in memories from other sessions of the same user. "
                    "Results identify their source session; treat them as historical context."
                )
        return tools


def _text(category: str, memory: str) -> str:
    return f"{category}: {memory}"
