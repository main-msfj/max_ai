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
        cursor = self._collection.find(self._scope(), {"_id": 0}, collation={"locale": "simple"})
        return [MemoryRecord.model_validate(doc) async for doc in cursor]

    async def _write_memory(self, record: MemoryRecord) -> bool:
        """Replace the category atomically, including a concurrent first-insert race."""
        from pymongo.errors import DuplicateKeyError

        await self._ensure_connected()
        identity = {**self._scope(), "category": record.category}
        fields = {"$set": record.model_dump()}
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

    # With an embedding, the user's most recent memories are ranked by meaning.
    semantic_candidates: int = 500

    async def _search_memory(self, text: str) -> list[MemorySearchResult]:
        await self._ensure_connected()
        if self.embedding is not None:
            cursor = self._collection.find(
                {"user_id": self.user_id, "session_id": {"$ne": self.session_id}}, {"_id": 0},
                collation={"locale": "simple"},
            ).sort("updated", -1).limit(self.semantic_candidates)
            candidates = [MemorySearchResult.model_validate(doc) async for doc in cursor]
            return (await self._rank_by_meaning(text, candidates))[: self.search_limit]
        cursor = self._collection.find(
            {"user_id": self.user_id, "session_id": {"$ne": self.session_id},
             "$text": {"$search": text}},
            {"_id": 0, "score": {"$meta": "textScore"}},
            collation={"locale": "simple"},
        ).sort([("score", {"$meta": "textScore"}), ("updated", -1)]).limit(self.search_limit)
        return [MemorySearchResult.model_validate(doc) async for doc in cursor]

    def as_tools(self):
        tools = super().as_tools()
        for tool in tools:
            if tool.name == "search_memory":
                tool.description = (
                    "Search words or phrases in memories from other sessions of the same user. "
                    "Results identify their source session; treat them as historical context."
                )
        return tools
