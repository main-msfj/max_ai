"""Knowledge in one SQLite file: a row per (source, block_id) with its vector.

Blocks are embedded once, when they're written (``upsert_block``), and the
vector is stored with them; a search only embeds the query. Blocks written
with another embedding model are re-embedded on the next search.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import typing as t
from pathlib import Path

from ....base.embedding import CoreEmbedding
from ....base.knowledge import CoreKnowledgeRegistry, KnowledgeToolMode
from ....core import KnowledgeBlock
from ....core.embeddings import FastEmbedEmbedding, pack_vector, rank, unpack_vector
from ._model import SQLiteKnowledgeRegistryConfig

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    source   TEXT NOT NULL,
    block_id TEXT NOT NULL,
    content  TEXT NOT NULL,
    tokens   INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    vector   BLOB,
    model_id TEXT,
    PRIMARY KEY (source, block_id)
);
"""


class SQLiteKnowledgeRegistry(CoreKnowledgeRegistry):
    """A named knowledge source searched by meaning. The agent only searches;
    loading documents is an application API (``upsert_block``/``delete_block``).
    Sources are shared by everyone configured with the same ``name``."""

    component_schema = SQLiteKnowledgeRegistryConfig
    component_type = "knowledge"
    component_provider_override = "max_ai.capabilities.knowledge.sqlite.SQLiteKnowledgeRegistry"

    def __init__(
        self,
        name: str,
        description: str,
        base_path: str | Path,
        tool_mode: KnowledgeToolMode = KnowledgeToolMode.FULL,
        *,
        db_name: str = "knowledge.sqlite3",
        min_score: float = 0.2,
        embedding: CoreEmbedding | None = None,
    ) -> None:
        super().__init__(name, description, tool_mode)
        self.base_path = Path(base_path).expanduser().resolve()
        self.db_name = db_name
        self.min_score = min_score
        # Default: the local multilingual model (``maxai[embeddings]``).
        self.embedding = embedding if embedding is not None else FastEmbedEmbedding()
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self.base_path / self.db_name

    def _to_config(self) -> SQLiteKnowledgeRegistryConfig:
        return SQLiteKnowledgeRegistryConfig(
            name=self.name, description=self.description, base_path=str(self.base_path),
            db_name=self.db_name, tool_mode=self.tool_mode, min_score=self.min_score,
            embedding=self.embedding.serialize().model_dump(exclude_none=True),
        )

    @classmethod
    def _from_config(cls, config: SQLiteKnowledgeRegistryConfig) -> SQLiteKnowledgeRegistry:
        embedding = CoreEmbedding.deserialize(config.embedding) if config.embedding else None
        return cls(**config.model_dump(exclude={"embedding"}), embedding=embedding)

    # -------- CONNECTION -----------------------------------------------------------
    async def connect(self) -> None:
        def open_db() -> sqlite3.Connection:
            self.base_path.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, check_same_thread=False)
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)
            return db

        if self._db is None:
            self._db = await asyncio.to_thread(open_db)

    async def disconnect(self) -> None:
        db, self._db = self._db, None
        if db is not None:
            await asyncio.to_thread(db.close)

    async def _run(self, work: t.Callable[[sqlite3.Connection], t.Any]) -> t.Any:
        """``work`` in a thread, holding the lock, inside one transaction."""
        await self._ensure_connected()
        db = self._db
        assert db is not None

        def locked() -> t.Any:
            with self._lock, db:
                return work(db)

        return await asyncio.to_thread(locked)

    # -------- SEARCH -----------------------------------------------------------
    async def search(self, query: str, limit: int = 5) -> list[KnowledgeBlock]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if not isinstance(query, str) or not query.strip():
            return []
        rows = await self._run(lambda db: db.execute(
            "SELECT block_id, content, tokens, metadata, vector, model_id "
            "FROM knowledge WHERE source=?", (self.name,),
        ).fetchall())
        if not rows:
            return []
        blocks = [KnowledgeBlock(content=c, tokens=tk, metadata=json.loads(m)) for _, c, tk, m, _, _ in rows]
        vectors = await self._vectors(rows)
        ranked = rank(await self.embedding.embed_one(query), blocks, vectors,
                      limit=limit, min_score=self.min_score)
        return [block for _, block in ranked]

    async def _vectors(self, rows: list[tuple]) -> list[list[float]]:
        """Stored vectors; missing or other-model ones are embedded and saved."""
        model_id = self.embedding.model_id
        stale = [i for i, row in enumerate(rows) if row[4] is None or row[5] != model_id]
        fresh = await self.embedding.embed([rows[i][1] for i in stale])
        if stale:
            await self._run(lambda db: db.executemany(
                "UPDATE knowledge SET vector=?, model_id=? WHERE source=? AND block_id=?",
                [(pack_vector(v), model_id, self.name, rows[i][0]) for i, v in zip(stale, fresh)],
            ))
        vectors = [unpack_vector(row[4]) if row[4] is not None and row[5] == model_id else []
                   for row in rows]
        for i, vector in zip(stale, fresh):
            vectors[i] = vector
        return vectors

    # -------- INGESTION (application API, not an agent tool) -------------------------
    @staticmethod
    def _block_id(block_id: str) -> str:
        if not isinstance(block_id, str) or not block_id.strip():
            raise ValueError("block_id must be a non-empty string")
        return block_id.strip()

    async def upsert_block(self, block_id: str, block: KnowledgeBlock) -> bool:
        """Create or replace a block (embedded now). True when newly inserted."""
        block_id = self._block_id(block_id)
        if not isinstance(block, KnowledgeBlock):
            raise TypeError("block must be a KnowledgeBlock")
        vector = pack_vector(await self.embedding.embed_one(block.content))

        def write(db: sqlite3.Connection) -> bool:
            existed = db.execute(
                "SELECT 1 FROM knowledge WHERE source=? AND block_id=?", (self.name, block_id),
            ).fetchone()
            db.execute(
                "INSERT INTO knowledge VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (source, block_id) DO UPDATE SET content=excluded.content, "
                "tokens=excluded.tokens, metadata=excluded.metadata, "
                "vector=excluded.vector, model_id=excluded.model_id",
                (self.name, block_id, block.content, block.tokens,
                 json.dumps(block.metadata, ensure_ascii=False), vector, self.embedding.model_id),
            )
            return existed is None

        return await self._run(write)

    async def delete_block(self, block_id: str) -> bool:
        block_id = self._block_id(block_id)
        deleted = await self._run(lambda db: db.execute(
            "DELETE FROM knowledge WHERE source=? AND block_id=?", (self.name, block_id),
        ).rowcount)
        return deleted > 0


__all__ = ["SQLiteKnowledgeRegistry"]
