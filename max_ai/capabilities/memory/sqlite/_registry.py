"""Memory in one SQLite file: a row per (user, session, category).

With an ``embedding``, each memory's vector is stored next to it when it's
written, so ``search_memory`` ranks by meaning without re-embedding stored
memories (after a restart too). Rows from before the embedding was set, or
from another model, get their vector on the next search.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import typing as t
from pathlib import Path

from ....base.embedding import DEFAULT_EMBEDDING, CoreEmbedding
from ....base.memory import (
    CoreMemoryRegistry,
    MemoryRecord,
    MemorySearchResult,
    MemoryToolMode,
)
from ....core.embeddings import pack_vector, rank, unpack_vector
from ._model import SQLiteMemoryRegistryConfig

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    category   TEXT NOT NULL,
    memory     TEXT NOT NULL,
    updated    TEXT NOT NULL,
    vector     BLOB,
    model_id   TEXT,
    PRIMARY KEY (user_id, session_id, category)
);
CREATE INDEX IF NOT EXISTS memory_by_user ON memory (user_id, updated DESC);
"""


class SQLiteMemoryRegistry(CoreMemoryRegistry):
    """One connection shared by every run's bound copy; SQLite calls run in
    a thread so they never block other runs. Fits one process (or a few
    on one machine); for many servers use MongoDB."""

    component_schema = SQLiteMemoryRegistryConfig
    component_type = "memory"
    component_provider_override = "max_ai.capabilities.memory.sqlite.SQLiteMemoryRegistry"

    # Search looks at the user's most recent memories from other sessions.
    search_candidates: t.ClassVar[int] = 2000

    def __init__(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        base_path: str | Path | None = None,
        tool_mode: MemoryToolMode = MemoryToolMode.FULL,
        *,
        db_name: str = "memory.sqlite3",
        context_days: int | None = 30,
        search_limit: int = 20,
        embedding: CoreEmbedding | None = DEFAULT_EMBEDDING,
    ) -> None:
        """Initialize ``SQLiteMemoryRegistry``.

Parameters
----------
user_id : str | None
    Value supplied for ``user_id``.
session_id : str | None
    Value supplied for ``session_id``.
base_path : str | Path | None
    Value supplied for ``base_path``.
tool_mode : MemoryToolMode
    Value supplied for ``tool_mode``.
db_name : str
    Value supplied for ``db_name``.
context_days : int | None
    Value supplied for ``context_days``.
search_limit : int
    Value supplied for ``search_limit``.
embedding : CoreEmbedding | None
    Value supplied for ``embedding``."""
        if base_path is None:
            raise ValueError("SQLiteMemoryRegistry needs a base_path")
        self.base_path = Path(base_path).expanduser().resolve()
        self.db_name = db_name
        self.search_limit = search_limit
        super().__init__(user_id, session_id, tool_mode,
                         context_days=context_days, embedding=embedding)
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Perform the ``path`` operation for ``SQLiteMemoryRegistry``."""
        return self.base_path / self.db_name

    def _to_config(self) -> SQLiteMemoryRegistryConfig:
        """Build the serializable configuration for ``SQLiteMemoryRegistry``."""
        return SQLiteMemoryRegistryConfig(
            base_path=str(self.base_path), db_name=self.db_name,
            user_id=self.user_id, session_id=self.session_id, tool_mode=self.tool_mode,
            context_days=self.context_days, search_limit=self.search_limit,
            embedding=self.embedding.serialize().model_dump(exclude_none=True) if self.embedding else None,
        )

    @classmethod
    def _from_config(cls, config: SQLiteMemoryRegistryConfig) -> SQLiteMemoryRegistry:
        """Create an instance from its configuration for ``SQLiteMemoryRegistry``.

Parameters
----------
config : SQLiteMemoryRegistryConfig
    Value supplied for ``config``."""
        embedding = CoreEmbedding.deserialize(config.embedding) if config.embedding else None
        return cls(**config.model_dump(exclude={"embedding"}), embedding=embedding)

    # -------- CONNECTION -----------------------------------------------------------
    async def connect(self) -> None:
        """Open required resources for ``SQLiteMemoryRegistry``."""
        def open_db() -> sqlite3.Connection:
            """Open db for ``SQLiteMemoryRegistry``."""
            self.base_path.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, check_same_thread=False)
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)
            return db

        if self._db is None:
            self._db = await asyncio.to_thread(open_db)

    async def disconnect(self) -> None:
        """Release resources held for ``SQLiteMemoryRegistry``."""
        db, self._db = self._db, None
        if db is not None:
            await asyncio.to_thread(db.close)

    async def _run(self, work: t.Callable[[sqlite3.Connection], t.Any]) -> t.Any:
        """``work`` in a thread, holding the lock, inside one transaction."""
        await self._ensure_connected()
        db = self._db
        assert db is not None

        def locked() -> t.Any:
            """Perform the ``locked`` operation for ``SQLiteMemoryRegistry``."""
            with self._lock, db:
                return work(db)

        return await asyncio.to_thread(locked)

    # -------- STORAGE -----------------------------------------------------------
    async def _read_session(self) -> list[MemoryRecord]:
        """Perform the internal ``read session`` operation for ``SQLiteMemoryRegistry``."""
        rows = await self._run(lambda db: db.execute(
            "SELECT category, memory, updated FROM memory WHERE user_id=? AND session_id=?",
            (self.user_id, self.session_id),
        ).fetchall())
        return [MemoryRecord(category=c, memory=m, updated=u) for c, m, u in rows]

    async def _write_memory(self, record: MemoryRecord) -> bool:
        """Perform the internal ``write memory`` operation for ``SQLiteMemoryRegistry``.

Parameters
----------
record : MemoryRecord
    Value supplied for ``record``."""
        vector = model_id = None
        if self.embedding is not None:
            vector = pack_vector(await self.embedding.embed_one(_text(record.category, record.memory)))
            model_id = self.embedding.model_id

        def write(db: sqlite3.Connection) -> bool:
            """Perform the ``write`` operation for ``SQLiteMemoryRegistry``.

Parameters
----------
db : sqlite3.Connection
    Value supplied for ``db``."""
            key = (self.user_id, self.session_id, record.category)
            existed = db.execute(
                "SELECT 1 FROM memory WHERE user_id=? AND session_id=? AND category=?", key,
            ).fetchone()
            db.execute(
                "INSERT INTO memory VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (user_id, session_id, category) DO UPDATE SET "
                "memory=excluded.memory, updated=excluded.updated, "
                "vector=excluded.vector, model_id=excluded.model_id",
                (*key, record.memory, record.updated.isoformat(), vector, model_id),
            )
            return existed is None

        return await self._run(write)

    async def _delete_memory(self, category: str) -> bool:
        """Perform the internal ``delete memory`` operation for ``SQLiteMemoryRegistry``.

Parameters
----------
category : str
    Value supplied for ``category``."""
        deleted = await self._run(lambda db: db.execute(
            "DELETE FROM memory WHERE user_id=? AND session_id=? AND category=?",
            (self.user_id, self.session_id, category),
        ).rowcount)
        return deleted > 0

    async def _search_memory(self, text: str) -> list[MemorySearchResult]:
        """Other sessions of this user: by meaning with an embedding, else by words."""
        rows = await self._run(lambda db: db.execute(
            "SELECT session_id, category, memory, updated, vector, model_id FROM memory "
            "WHERE user_id=? AND session_id<>? ORDER BY updated DESC LIMIT ?",
            (self.user_id, self.session_id, self.search_candidates),
        ).fetchall())
        results = [MemorySearchResult(session_id=s, category=c, memory=m, updated=u)
                   for s, c, m, u, _, _ in rows]
        if self.embedding is None:
            needle = text.strip().lower()
            matches = [r for r in results if needle in r.category.lower() or needle in r.memory.lower()]
            return matches[: self.search_limit]

        vectors = await self._vectors(results, [(v, model) for *_, v, model in rows])
        query = await self.embedding.embed_one(text)
        ranked = rank(query, results, vectors, limit=self.search_limit,
                      min_score=self.semantic_min_score)
        return [result for _, result in ranked]

    async def _vectors(
        self, results: list[MemorySearchResult], stored: list[tuple[bytes | None, str | None]],
    ) -> list[list[float]]:
        """Stored vectors; missing or other-model ones are embedded and saved."""
        assert self.embedding is not None
        model_id = self.embedding.model_id
        stale = [i for i, (vector, model) in enumerate(stored) if vector is None or model != model_id]
        fresh = await self.embedding.embed([_text(results[i].category, results[i].memory) for i in stale])
        if stale:
            await self._run(lambda db: db.executemany(
                "UPDATE memory SET vector=?, model_id=? WHERE user_id=? AND session_id=? AND category=?",
                [(pack_vector(vector), model_id, self.user_id, results[i].session_id, results[i].category)
                 for i, vector in zip(stale, fresh)],
            ))
        vectors = [unpack_vector(v) if v is not None and m == model_id else [] for v, m in stored]
        for i, vector in zip(stale, fresh):
            vectors[i] = vector
        return vectors


def _text(category: str, memory: str) -> str:
    """Perform the internal ``text`` operation.

Parameters
----------
category : str
    Value supplied for ``category``.
memory : str
    Value supplied for ``memory``."""
    return f"{category}: {memory}"


__all__ = ["SQLiteMemoryRegistry"]
