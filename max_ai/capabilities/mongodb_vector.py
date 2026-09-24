"""Native MongoDB vector search ($vectorSearch) for knowledge and memory.

The index is created on first use with the embedding's dimensions (and
recreated if a model with other dimensions replaces it). A MongoDB without
Search (plain Community ``mongo`` image) answers ``None`` and the registry
ranks in Python instead, so nothing breaks. Needs MongoDB Atlas, the
``mongodb/mongodb-atlas-local`` image, or Community/Enterprise with Search.
"""

from __future__ import annotations

import logging
import typing as t

logger = logging.getLogger(__name__)


class MongoVectorIndex:
    """One ``vectorSearch`` index on the ``vector`` field of a collection."""

    def __init__(self, name: str, filters: t.Sequence[str]) -> None:
        """Initialize ``MongoVectorIndex``.

Parameters
----------
name : str
    Value supplied for ``name``.
filters : t.Sequence[str]
    Value supplied for ``filters``."""
        self.name = name
        self.filters = list(filters)
        self._supported: bool | None = None  # None: not checked yet
        self._dimensions: int | None = None

    def _definition(self, dimensions: int) -> dict[str, t.Any]:
        """Perform the internal ``definition`` operation for ``MongoVectorIndex``.

Parameters
----------
dimensions : int
    Value supplied for ``dimensions``."""
        return {"fields": [
            {"type": "vector", "path": "vector", "numDimensions": dimensions, "similarity": "cosine"},
            *({"type": "filter", "path": path} for path in self.filters),
        ]}

    async def ready(self, collection: t.Any, dimensions: int) -> bool:
        """Create (or fix) the index if needed; True once it can be queried."""
        if self._supported is False:
            return False
        from pymongo.errors import OperationFailure
        from pymongo.operations import SearchIndexModel

        try:
            found = [index async for index in await collection.list_search_indexes(self.name)]
            if found and self._index_dimensions(found[0]) not in (None, dimensions):
                await collection.drop_search_index(self.name)  # another embedding model
                found = []
            if not found:
                await collection.create_search_index(SearchIndexModel(
                    definition=self._definition(dimensions), name=self.name, type="vectorSearch",
                ))
                logger.info("Created MongoDB vector index %s (%d dims)", self.name, dimensions)
                return False  # builds in the background; rank in Python meanwhile
        except OperationFailure as error:
            self._supported = False
            logger.info("MongoDB without vector search (%s): ranking in Python", error.code)
            return False
        self._supported = True
        return bool(found[0].get("queryable"))

    @staticmethod
    def _index_dimensions(index: dict[str, t.Any]) -> int | None:
        """Perform the internal ``index dimensions`` operation for ``MongoVectorIndex``.

Parameters
----------
index : dict[str, t.Any]
    Value supplied for ``index``."""
        definition = index.get("latestDefinition") or index.get("definition") or {}
        for field in definition.get("fields", []):
            if field.get("type") == "vector":
                return field.get("numDimensions")
        return None

    async def search(
        self,
        collection: t.Any,
        query_vector: list[float],
        *,
        filter: dict[str, t.Any],
        limit: int,
        min_score: float,
        projection: dict[str, t.Any],
    ) -> list[dict[str, t.Any]] | None:
        """Documents closest to ``query_vector`` (cosine above ``min_score``),
        or ``None`` when the index isn't available and the caller should rank."""
        if not await self.ready(collection, len(query_vector)):
            return None
        cursor = await collection.aggregate([
            {"$vectorSearch": {
                "index": self.name, "path": "vector", "queryVector": query_vector,
                "numCandidates": max(100, limit * 20), "limit": limit, "filter": filter,
            }},
            {"$project": {**projection, "_id": 0, "score": {"$meta": "vectorSearchScore"}}},
        ])
        # vectorSearchScore is (1 + cosine) / 2 for cosine similarity.
        return [doc async for doc in cursor if 2 * doc.pop("score") - 1 > min_score]


__all__ = ["MongoVectorIndex"]
