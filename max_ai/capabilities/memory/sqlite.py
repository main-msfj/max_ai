"""SQLite-backed memory registry, aligned to the corrected base contract.

It implements ONLY the storage surface plus a recall() override that uses
the persisted vectors (the base recall would re-embed every row). Tools,
merge/dedup, batch convenience, update_fact, validation and cosine all come
from the base now — they were deleted here to avoid drift.

Notable: connect() migrates older DBs (pre confidence/source/expires_at)
via ALTER TABLE, so existing memory files keep working.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import typing as t
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from ...base.embeddings import (
    get_lightweight_embedding,
    get_lightweight_embeddings,
)
from ...base.memory import (
    CoreMemoryRegistry,
    MemoryToolMode,
    MemoryRecord,
    RecallQuery,
    MergePolicy,
    EmbedOne,
    EmbedMany,
)


class SQLiteMemoryRegistryConfig(BaseModel):
    user_id: str
    base_path: str
    tool_mode: MemoryToolMode = MemoryToolMode.FULL
    db_name: str = "memory.sqlite3"
    merge_similarity_threshold: float = 0.85
    context_days: int | None = 30


class SQLiteMemoryRegistry(CoreMemoryRegistry):
    component_schema = SQLiteMemoryRegistryConfig
    component_type = "memory"

    """SQLite durable memory for a single user. Ready to be swapped for
    sqlite-vec later without touching the contract."""

    def __init__(
        self,
        user_id: str,
        base_path: str | Path,
        tool_mode: MemoryToolMode = MemoryToolMode.FULL,
        *,
        db_name: str = "memory.sqlite3",
        merge_similarity_threshold: float = 0.85,
        context_days: int | None = 30,
        embed_one: EmbedOne | None = None,
        embed_many: EmbedMany | None = None,
    ) -> None:
        super().__init__(
            user_id=user_id,
            tool_mode=tool_mode,
            merge_policy=MergePolicy(similarity_threshold=merge_similarity_threshold),
            embed_one=embed_one or get_lightweight_embedding,
            embed_many=embed_many or get_lightweight_embeddings,
            context_days=context_days,
        )
        self.base_path = Path(base_path).expanduser().resolve()
        self.db_name = self.require_type(db_name, str, "db_name")
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()  # serializes the shared connection

    # -------- COMPONENT SERIALIZATION -----------------------------------------------------------
    def _to_config(self) -> SQLiteMemoryRegistryConfig:
        return SQLiteMemoryRegistryConfig(
            user_id=self.user_id,
            base_path=str(self.base_path),
            tool_mode=self.tool_mode,
            db_name=self.db_name,
            merge_similarity_threshold=self.merge_policy.similarity_threshold,
            context_days=self.context_days,
        )

    @classmethod
    def _from_config(cls, config: SQLiteMemoryRegistryConfig) -> "SQLiteMemoryRegistry":
        return cls(
            user_id=config.user_id,
            base_path=config.base_path,
            tool_mode=config.tool_mode,
            db_name=config.db_name,
            merge_similarity_threshold=config.merge_similarity_threshold,
            context_days=config.context_days,
        )

    @property
    def db_path(self) -> Path:
        return self.base_path / "backend-local" / self.db_name

    # -------- CONNECTION LIFECYCLE -----------------------------------------------------------
    async def connect(self) -> None:
        def _open() -> sqlite3.Connection:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    user_id      TEXT NOT NULL,
                    key          TEXT NOT NULL,
                    category     TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    confidence   REAL NOT NULL DEFAULT 1.0,
                    source       TEXT,
                    last_updated TEXT NOT NULL,
                    expires_at   TEXT,
                    vector       TEXT NOT NULL,
                    PRIMARY KEY (user_id, key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_user_category "
                "ON memory (user_id, category)"
            )
            conn.commit()
            return conn

        self._conn = await asyncio.to_thread(_open)

    async def disconnect(self) -> None:
        if self._conn is not None:
            conn, self._conn = self._conn, None
            await asyncio.to_thread(conn.close)

    async def _db(self) -> sqlite3.Connection:
        await self._ensure_connected()
        if self._conn is None:
            from ...errors.memory import MemoryError
            raise MemoryError("SQLite memory is not connected.")
        return self._conn

    # -------- STORAGE SURFACE (contract) -----------------------------------------------------------
    async def _read_all(self) -> list[MemoryRecord]:
        conn = await self._db()

        def _run() -> list[MemoryRecord]:
            rows = conn.execute(
                """
                SELECT key, category, content, confidence, source,
                       last_updated, expires_at
                FROM memory WHERE user_id = ?
                ORDER BY last_updated DESC
                """,
                (self.user_id,),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

        async with self._lock:
            records = await asyncio.to_thread(_run)
        return [r for r in records if not r.is_expired()]

    async def _write_many(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        conn = await self._db()
        # One batch embedding call for all records, not one per record.
        vectors = self._embed_many([f"{r.category}\n{r.content}" for r in records])
        payload = [
            (
                self.user_id,
                r.key,
                r.category,
                r.content,
                r.confidence,
                r.source,
                r.last_updated.isoformat(),
                r.expires_at.isoformat() if r.expires_at else None,
                json.dumps(vec),
            )
            for r, vec in zip(records, vectors)
        ]

        def _run() -> None:
            conn.executemany(
                """
                INSERT INTO memory
                    (user_id, key, category, content, confidence, source,
                     last_updated, expires_at, vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    category     = excluded.category,
                    content      = excluded.content,
                    confidence   = excluded.confidence,
                    source       = excluded.source,
                    last_updated = excluded.last_updated,
                    expires_at   = excluded.expires_at,
                    vector       = excluded.vector
                """,
                payload,
            )
            conn.commit()

        async with self._lock:
            await asyncio.to_thread(_run)

    async def _delete_many(self, keys: list[str]) -> None:
        if not keys:
            return
        conn = await self._db()

        def _run() -> None:
            conn.executemany(
                "DELETE FROM memory WHERE user_id = ? AND key = ?",
                [(self.user_id, k) for k in keys],
            )
            conn.commit()

        async with self._lock:
            await asyncio.to_thread(_run)

    # -------- RETRIEVAL (override: rank against persisted vectors) -----------------------------------------------------------
    async def recall(self, query: RecallQuery) -> list[MemoryRecord]:
        conn = await self._db()
        clean_category = query.category.strip() if query.category else None

        def _run() -> list[tuple[MemoryRecord, str]]:
            sql = (
                "SELECT key, category, content, confidence, source, "
                "last_updated, expires_at, vector FROM memory WHERE user_id = ?"
            )
            params: list[t.Any] = [self.user_id]
            if clean_category:
                sql += " AND category = ?"
                params.append(clean_category)
            rows = conn.execute(sql, params).fetchall()
            return [(self._row_to_record(r), r["vector"]) for r in rows]

        async with self._lock:
            staged = await asyncio.to_thread(_run)

        candidates = [
            (rec, vec) for rec, vec in staged
            if not rec.is_expired() and rec.confidence >= query.min_confidence
        ]

        if not query.text:
            candidates.sort(key=lambda c: c[0].last_updated, reverse=True)
            return [rec for rec, _ in candidates[: query.limit]]

        qv = self._embed_one(query.text)
        scored = [(self._cosine(qv, json.loads(vec)), rec) for rec, vec in candidates]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: s[0], reverse=True)
        return [rec for _, rec in scored[: query.limit]]

    # -------- HELPERS -----------------------------------------------------------
    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            key=row["key"],
            category=row["category"],
            content=row["content"],
            confidence=row["confidence"],
            source=row["source"],
            last_updated=datetime.fromisoformat(row["last_updated"]),
            expires_at=(
                datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
            ),
        )