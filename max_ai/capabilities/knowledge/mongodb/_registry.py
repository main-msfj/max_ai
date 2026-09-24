"""Knowledge blocks scoped to a named source; ingestion is not an agent tool."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from ....base.embedding import DEFAULT_EMBEDDING, CoreEmbedding
from ....base.knowledge import CoreKnowledgeRegistry, KnowledgeToolMode
from ....core import KnowledgeBlock
from ....core.embeddings import FastEmbedEmbedding, rank
from ...mongodb_vector import MongoVectorIndex
from ._model import MongoDBKnowledgeRegistryConfig


class MongoDBKnowledgeRegistry(CoreKnowledgeRegistry):
    """Owns one async client; PyMongo is an optional dependency."""

    component_schema = MongoDBKnowledgeRegistryConfig
    component_type = "knowledge"
    component_provider_override = "max_ai.capabilities.knowledge.mongodb.MongoDBKnowledgeRegistry"

    def __init__(
        self,
        name: str,
        description: str,
        tool_mode: KnowledgeToolMode = KnowledgeToolMode.FULL,
        *,
        database: str = "max_ai",
        collection: str = "knowledge",
        uri_env: str = "MONGODB_URI",
        server_selection_timeout_ms: int = 5000,
        embedding: CoreEmbedding | None = DEFAULT_EMBEDDING,
        min_score: float = 0.2,
    ) -> None:
        """Initialize ``MongoDBKnowledgeRegistry``.

Parameters
----------
name : str
    Value supplied for ``name``.
description : str
    Value supplied for ``description``.
tool_mode : KnowledgeToolMode
    Value supplied for ``tool_mode``.
database : str
    Value supplied for ``database``.
collection : str
    Value supplied for ``collection``.
uri_env : str
    Value supplied for ``uri_env``.
server_selection_timeout_ms : int
    Value supplied for ``server_selection_timeout_ms``.
embedding : CoreEmbedding | None
    Value supplied for ``embedding``.
min_score : float
    Value supplied for ``min_score``."""
        super().__init__(name, description, tool_mode)
        self._mongo_config = MongoDBKnowledgeRegistryConfig(
            name=self.name, description=self.description, tool_mode=tool_mode,
            database=database, collection=collection, uri_env=uri_env,
            server_selection_timeout_ms=server_selection_timeout_ms, min_score=min_score,
        )
        # With an embedding: vectors stored with each block, search by meaning
        # ($vectorSearch when the server has it, else ranked in Python).
        # DEFAULT_EMBEDDING (nothing passed): FastEmbedEmbedding. Explicit
        # None: plain MongoDB $text search instead.
        self.embedding = FastEmbedEmbedding() if embedding is DEFAULT_EMBEDDING else embedding
        self._vector_index = MongoVectorIndex(f"{collection}_vector", ["source", "model_id"])
        self._mongo_lock = asyncio.Lock()
        self._mongo_client: Any = None
        self._collection: Any = None

    def _to_config(self) -> MongoDBKnowledgeRegistryConfig:
        """Build the serializable configuration for ``MongoDBKnowledgeRegistry``."""
        embedding = self.embedding.serialize().model_dump(exclude_none=True) if self.embedding else None
        return self._mongo_config.model_copy(update={"embedding": embedding}, deep=True)

    @classmethod
    def _from_config(cls, config: MongoDBKnowledgeRegistryConfig) -> "MongoDBKnowledgeRegistry":
        """Create an instance from its configuration for ``MongoDBKnowledgeRegistry``.

Parameters
----------
config : MongoDBKnowledgeRegistryConfig
    Value supplied for ``config``."""
        embedding = CoreEmbedding.deserialize(config.embedding) if config.embedding else None
        return cls(**config.model_dump(exclude={"embedding"}), embedding=embedding)

    # -------- CONNECTION -----------------------------------------------------------
    async def connect(self) -> None:
        """Open required resources for ``MongoDBKnowledgeRegistry``."""
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
        """Release resources held for ``MongoDBKnowledgeRegistry``."""
        async with self._mongo_lock:
            client = self._mongo_client
            self._mongo_client = None
            self._collection = None
            self._connected = False
            if client is not None:
                await client.close()

    async def _create_indexes(self, collection: Any) -> None:
        """Perform the internal ``create indexes`` operation for ``MongoDBKnowledgeRegistry``.

Parameters
----------
collection : Any
    Value supplied for ``collection``."""
        await collection.create_index(
            [("source", 1), ("block_id", 1)], unique=True,
            name="knowledge_identity", collation={"locale": "simple"},
        )
        await collection.create_index(
            [("source", 1), ("content", "text")], name="knowledge_text",
            default_language="none", collation={"locale": "simple"},
        )

    # -------- SEARCH -----------------------------------------------------------
    async def search(self, query: str, limit: int = 5) -> list[KnowledgeBlock]:
        """Search MongoDBKnowledgeRegistry for matching records.

Parameters
----------
query : str
    Value supplied for ``query``.
limit : int
    Value supplied for ``limit``."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if not isinstance(query, str) or not query.strip():
            return []
        await self._ensure_connected()
        if self.embedding is not None:
            return await self._search_by_meaning(query, limit)
        cursor = self._collection.find(
            {"source": self.name, "$text": {"$search": query.strip()}},
            {"_id": 0, "content": 1, "tokens": 1, "metadata": 1,
             "score": {"$meta": "textScore"}},
            collation={"locale": "simple"},
        ).sort([("score", {"$meta": "textScore"}), ("block_id", 1)]).limit(limit)
        return [KnowledgeBlock.model_validate(doc) async for doc in cursor]

    # Without a vector index, at most this many blocks are ranked in Python.
    max_candidates: int = 5000

    async def _search_by_meaning(self, query: str, limit: int) -> list[KnowledgeBlock]:
        """Perform the internal ``search by meaning`` operation for ``MongoDBKnowledgeRegistry``.

Parameters
----------
query : str
    Value supplied for ``query``.
limit : int
    Value supplied for ``limit``."""
        assert self.embedding is not None
        await self._embed_missing()
        model_id = self.embedding.model_id
        query_vector = await self.embedding.embed_one(query)
        fields = {"content": 1, "tokens": 1, "metadata": 1}
        min_score = self._mongo_config.min_score
        found = await self._vector_index.search(
            self._collection, query_vector, filter={"source": self.name, "model_id": model_id},
            limit=limit, min_score=min_score, projection=fields,
        )
        if found is None:  # no vector index here: rank in Python
            cursor = self._collection.find(
                {"source": self.name, "model_id": model_id}, {**fields, "vector": 1, "_id": 0},
            ).limit(self.max_candidates)
            docs = [doc async for doc in cursor]
            ranked = rank(query_vector, docs, [d["vector"] for d in docs], limit=limit, min_score=min_score)
            found = [doc for _, doc in ranked]
        return [KnowledgeBlock(content=d["content"], tokens=d.get("tokens", 0),
                               metadata=d.get("metadata", {})) for d in found]

    async def _embed_missing(self) -> None:
        """Blocks written before the embedding (or with another model) get
        their vector once, and keep it."""
        assert self.embedding is not None
        model_id = self.embedding.model_id
        cursor = self._collection.find(
            {"source": self.name, "$or": [{"model_id": {"$ne": model_id}}, {"vector": {"$exists": False}}]},
            {"_id": 0, "block_id": 1, "content": 1},
        )
        stale = [doc async for doc in cursor]
        for doc, vector in zip(stale, await self.embedding.embed([d["content"] for d in stale])):
            await self._collection.update_one(
                {"source": self.name, "block_id": doc["block_id"]},
                {"$set": {"vector": vector, "model_id": model_id}},
            )

    # -------- INGESTION (application API, not an agent tool) -----------------------------
    @staticmethod
    def _block_id(block_id: str) -> str:
        """Perform the internal ``block id`` operation for ``MongoDBKnowledgeRegistry``.

Parameters
----------
block_id : str
    Value supplied for ``block_id``."""
        if not isinstance(block_id, str) or not block_id.strip():
            raise ValueError("block_id must be a non-empty string")
        return block_id.strip()

    async def upsert_block(self, block_id: str, block: KnowledgeBlock) -> bool:
        """Create/replace a whole block. Return True when newly inserted."""
        from pymongo.errors import DuplicateKeyError

        identity = {"source": self.name, "block_id": self._block_id(block_id)}
        if not isinstance(block, KnowledgeBlock):
            raise TypeError("block must be a KnowledgeBlock")
        await self._ensure_connected()
        document = block.model_dump()
        if self.embedding is not None:
            document["vector"] = await self.embedding.embed_one(block.content)
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

    async def delete_block(self, block_id: str) -> bool:
        """Delete block for ``MongoDBKnowledgeRegistry``.

Parameters
----------
block_id : str
    Value supplied for ``block_id``."""
        identity = {"source": self.name, "block_id": self._block_id(block_id)}
        await self._ensure_connected()
        result = await self._collection.delete_one(identity, collation={"locale": "simple"})
        return result.deleted_count > 0
