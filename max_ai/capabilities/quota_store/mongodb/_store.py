"""Usage in MongoDB: one document per ``(user_id, period)``."""

from __future__ import annotations

import asyncio
import os
import typing as t

from ....base.quota_store import CoreQuotaStore
from ....core.model.quota import QuotaUsage
from ._model import MongoDBQuotaStoreConfig


class MongoDBQuotaStore(CoreQuotaStore):
    """``add`` is one atomic ``$inc`` upsert, so any number of processes or
    serverless invocations can charge the same user at once. PyMongo is an
    optional dependency."""

    component_schema = MongoDBQuotaStoreConfig
    component_provider_override = "max_ai.capabilities.quota_store.mongodb.MongoDBQuotaStore"

    def __init__(
        self,
        database: str = "max_ai",
        collection: str = "quota",
        uri_env: str = "MONGODB_URI",
        server_selection_timeout_ms: int = 5000,
    ) -> None:
        """Initialize ``MongoDBQuotaStore``.

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
        self.config = MongoDBQuotaStoreConfig(
            database=database, collection=collection, uri_env=uri_env,
            server_selection_timeout_ms=server_selection_timeout_ms,
        )
        self._mongo_lock = asyncio.Lock()
        self._mongo_client: t.Any = None
        self._collection: t.Any = None

    def _to_config(self) -> MongoDBQuotaStoreConfig:
        """Build the serializable configuration for ``MongoDBQuotaStore``."""
        return self.config.model_copy()

    @classmethod
    def _from_config(cls, config: MongoDBQuotaStoreConfig) -> MongoDBQuotaStore:
        """Create an instance from its configuration for ``MongoDBQuotaStore``.

Parameters
----------
config : MongoDBQuotaStoreConfig
    Value supplied for ``config``."""
        return cls(**config.model_dump())

    # -------- CONNECTION -----------------------------------------------------------
    async def connect(self) -> None:
        """Open required resources for ``MongoDBQuotaStore``."""
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
                    [("user_id", 1), ("period", 1)], unique=True, name="quota_identity",
                )
            except BaseException:
                await client.close()
                raise
            self._mongo_client, self._collection = client, collection

    async def disconnect(self) -> None:
        """Release resources held for ``MongoDBQuotaStore``."""
        async with self._mongo_lock:
            client, self._mongo_client, self._collection = self._mongo_client, None, None
            if client is not None:
                await client.close()

    # -------- BACKEND HOOKS -----------------------------------------------------------
    async def _usage(self, user_id: str, period_key: str) -> QuotaUsage:
        """Perform the internal ``usage`` operation for ``MongoDBQuotaStore``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
period_key : str
    Value supplied for ``period_key``."""
        doc = await self._collection.find_one({"user_id": user_id, "period": period_key})
        return _usage(doc)

    async def _add(self, user_id: str, period_key: str, delta: QuotaUsage) -> QuotaUsage:
        """Perform the internal ``add`` operation for ``MongoDBQuotaStore``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
period_key : str
    Value supplied for ``period_key``.
delta : QuotaUsage
    Value supplied for ``delta``."""
        from pymongo import ReturnDocument

        doc = await self._collection.find_one_and_update(
            {"user_id": user_id, "period": period_key},
            {"$inc": delta.model_dump()},
            upsert=True, return_document=ReturnDocument.AFTER,
        )
        return _usage(doc)


def _usage(doc: dict[str, t.Any] | None) -> QuotaUsage:
    """Perform the internal ``usage`` operation.

Parameters
----------
doc : dict[str, t.Any] | None
    Value supplied for ``doc``."""
    if not doc:
        return QuotaUsage()
    return QuotaUsage(tokens=doc.get("tokens", 0), cost_usd=doc.get("cost_usd", 0.0), tasks=doc.get("tasks", 0))
