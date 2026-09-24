"""Sessions in MongoDB: one document per ``(user_id, session_id)``."""

from __future__ import annotations

import asyncio
import os
import typing as t

from ....base.session_store import CoreSessionStore
from ....core.model.session import SessionInfo
from ....types.run_context import RunContext
from ._model import MongoDBSessionStoreConfig

# Listing reads only these fields, never the whole conversation.
_INFO_FIELDS = {"_id": 0, "user_id": 1, "session_id": 1, "title": 1,
                "updated_at": 1, "message_count": 1, "compactions": 1}


class MongoDBSessionStore(CoreSessionStore):
    """Any number of processes or serverless invocations can save and resume
    the same user's sessions. Each document holds the listing fields plus the
    whole ``RunContext`` as a JSON string (tool arguments and schemas may use
    keys such as ``$ref`` that MongoDB won't store as field names). A
    document is capped at 16MB by MongoDB; compaction keeps contexts far
    below that. PyMongo is an optional dependency.
    """

    component_schema = MongoDBSessionStoreConfig
    component_provider_override = "max_ai.capabilities.session_store.mongodb.MongoDBSessionStore"

    def __init__(
        self,
        database: str = "max_ai",
        collection: str = "sessions",
        uri_env: str = "MONGODB_URI",
        server_selection_timeout_ms: int = 5000,
    ) -> None:
        """Initialize ``MongoDBSessionStore``.

Parameters
----------
database : str
    Value supplied for ``database``.
collection : str
    Value supplied for ``collection``.
uri_env : str
    Value supplied for ``uri_env``.
server_selection_timeout_ms : int
    Value supplied for ``server_selection_timeout_ms``."""
        super().__init__()
        self.config = MongoDBSessionStoreConfig(
            database=database, collection=collection, uri_env=uri_env,
            server_selection_timeout_ms=server_selection_timeout_ms,
        )
        self._mongo_lock = asyncio.Lock()
        self._mongo_client: t.Any = None
        self._collection: t.Any = None

    def _to_config(self) -> MongoDBSessionStoreConfig:
        """Build the serializable configuration for ``MongoDBSessionStore``."""
        return self.config.model_copy()

    @classmethod
    def _from_config(cls, config: MongoDBSessionStoreConfig) -> MongoDBSessionStore:
        """Create an instance from its configuration for ``MongoDBSessionStore``.

Parameters
----------
config : MongoDBSessionStoreConfig
    Value supplied for ``config``."""
        return cls(**config.model_dump())

    # -------- CONNECTION -----------------------------------------------------------
    async def connect(self) -> None:
        """Open required resources for ``MongoDBSessionStore``."""
        async with self._mongo_lock:
            if self._mongo_client is not None:
                return
            try:
                from pymongo import AsyncMongoClient
            except ImportError as error:
                raise ImportError("Install MongoDB support with: pip install 'maxai[mongodb]'") from error
            uri = os.environ.get(self.config.uri_env)
            if not uri or not uri.strip():
                raise ValueError(f"Set {self.config.uri_env} to your MongoDB connection URI")
            client = AsyncMongoClient(
                uri, tz_aware=True, serverSelectionTimeoutMS=self.config.server_selection_timeout_ms,
            )
            try:
                await client.admin.command("ping")
                collection = client[self.config.database][self.config.collection]
                await collection.create_index(
                    [("user_id", 1), ("session_id", 1)], unique=True, name="session_identity",
                )
                await collection.create_index(
                    [("user_id", 1), ("updated_at", -1)], name="session_recent",
                )
            except BaseException:
                await client.close()
                raise
            self._mongo_client, self._collection = client, collection

    async def disconnect(self) -> None:
        """Release resources held for ``MongoDBSessionStore``."""
        async with self._mongo_lock:
            client, self._mongo_client, self._collection = self._mongo_client, None, None
            if client is not None:
                await client.close()

    # -------- BACKEND HOOKS -----------------------------------------------------------
    async def _load(self, user_id: str, session_id: str) -> RunContext | None:
        """Perform the internal ``load`` operation for ``MongoDBSessionStore``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
session_id : str
    Value supplied for ``session_id``."""
        doc = await self._collection.find_one(
            {"user_id": user_id, "session_id": session_id}, {"_id": 0, "context": 1},
        )
        return None if doc is None else RunContext.model_validate_json(doc["context"])

    async def _save(self, ctx: RunContext, info: SessionInfo) -> None:
        """Perform the internal ``save`` operation for ``MongoDBSessionStore``.

Parameters
----------
ctx : RunContext
    Value supplied for ``ctx``.
info : SessionInfo
    Value supplied for ``info``."""
        await self._collection.replace_one(
            {"user_id": info.user_id, "session_id": info.session_id},
            {**info.model_dump(), "context": ctx.model_dump_json()},
            upsert=True,
        )

    async def _list(self, user_id: str) -> t.Iterable[SessionInfo]:
        """Perform the internal ``list`` operation for ``MongoDBSessionStore``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``."""
        cursor = self._collection.find({"user_id": user_id}, _INFO_FIELDS)
        return [SessionInfo.model_validate(doc) async for doc in cursor]

    async def _delete(self, user_id: str, session_id: str) -> bool:
        """Perform the internal ``delete`` operation for ``MongoDBSessionStore``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
session_id : str
    Value supplied for ``session_id``."""
        result = await self._collection.delete_one({"user_id": user_id, "session_id": session_id})
        return result.deleted_count > 0
