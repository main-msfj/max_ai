"""SQLite-backed conversation-context registry with local embeddings."""

from __future__ import annotations

import json
import math
import sqlite3
import typing as t
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel

from ...base.context import ContextBlock, CoreLogBookRegistry, LogBookToolMode
from ...base.observation import ObservationRecord
from ...base.embeddings import get_lightweight_embedding


class SQLiteContextRegistryConfig(BaseModel):
    user_id: str
    session_id: str
    base_path: str
    tool_mode: LogBookToolMode = LogBookToolMode.READ_ONLY
    db_name: str = "context.sqlite3"

class SQLiteContextRegistry(CoreLogBookRegistry):
    component_schema = SQLiteContextRegistryConfig
    component_type = "context"

    """SQLite-backed session summaries for a single user.

    The current session summary is fetched by the exact
    ``(user_id, session_id)`` pair for prompt injection. Semantic search
    scans summaries for this ``user_id`` and ranks them by vector
    similarity, which is the path used when the agent needs context from
    another session.
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        base_path: str | Path,
        tool_mode: LogBookToolMode = LogBookToolMode.READ_ONLY,
        *,
        db_name: str = "context.sqlite3",
    ) -> None:
        super().__init__(
            user_id=user_id,
            session_id=session_id,
            tool_mode=tool_mode,
        )
        self.base_path = Path(base_path).expanduser().resolve()
        self.db_name = self.require_type(db_name, str, "db_name")
        self._conn: sqlite3.Connection | None = None

    def _to_config(self) -> SQLiteContextRegistryConfig:
        return SQLiteContextRegistryConfig(
            user_id=self.user_id,
            session_id=self.session_id,
            base_path=str(self.base_path),
            tool_mode=self.tool_mode,
            db_name=self.db_name,
        )

    @classmethod
    def _from_config(cls, config: SQLiteContextRegistryConfig) -> "SQLiteContextRegistry":
        return cls(
            user_id=config.user_id,
            session_id=config.session_id,
            base_path=config.base_path,
            tool_mode=config.tool_mode,
            db_name=config.db_name,
        )

    @property
    def db_path(self) -> Path:
        return self.base_path / "backend-local" / self.db_name

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context (
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT NOT NULL,
                vector TEXT NOT NULL,
                PRIMARY KEY (user_id, session_id)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_user_timestamp "
            "ON context (user_id, timestamp)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                observation_type TEXT NOT NULL DEFAULT 'finding',
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_user_created "
            "ON observations (user_id, created_at DESC)"
        )
        self._conn.commit()

    async def disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def get_current_session_summary(self) -> str | None:
        conn = await self._ensure_db()
        row = conn.execute(
            """
            SELECT summary
            FROM context
            WHERE user_id = ? AND session_id = ?
            LIMIT 1
            """,
            (self.user_id, self.session_id),
        ).fetchone()
        if row is None:
            return None
        summary = row["summary"]
        if not isinstance(summary, str) or not summary.strip():
            return None
        return summary

    async def search(self, query: str, limit: int = 5) -> list[ContextBlock]:
        conn = await self._ensure_db()
        clean_query = self._validate_non_empty("query", query)
        query_vector = get_lightweight_embedding(clean_query)

        rows = conn.execute(
            """
            SELECT session_id, summary, timestamp, metadata, vector
            FROM context
            WHERE user_id = ?
            """,
            (self.user_id,),
        ).fetchall()

        ranked: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            score = self._cosine_similarity(query_vector, json.loads(row["vector"]))
            if score > 0:
                ranked.append((score, row))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [self._row_to_block(row, score) for score, row in ranked[:limit]]

    async def upsert_summary(
        self,
        *,
        session_id: str,
        summary: str,
        timestamp: datetime | None = None,
        metadata: dict[str, t.Any] | None = None,
    ) -> None:
        """Create or replace a stored session summary.

        This is intended for external summarization jobs, not for the
        LLM-facing tool surface.
        """
        conn = await self._ensure_db()
        clean_session_id = self._validate_non_empty("session_id", session_id)
        clean_summary = self._validate_non_empty("summary", summary)
        ts = timestamp or datetime.now(timezone.utc)
        vector = get_lightweight_embedding(clean_summary)

        conn.execute(
            """
            INSERT INTO context (
                user_id, session_id, summary, timestamp, metadata, vector
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, session_id) DO UPDATE SET
                summary = excluded.summary,
                timestamp = excluded.timestamp,
                metadata = excluded.metadata,
                vector = excluded.vector
            """,
            (
                self.user_id,
                clean_session_id,
                clean_summary,
                ts.isoformat(),
                json.dumps(metadata or {}),
                json.dumps(vector),
            ),
        )
        conn.commit()

    async def write_observation(
        self,
        content: str,
        observation_type: str = "finding",
        tags: list[str] | None = None,
    ) -> str:
        conn = await self._ensure_db()
        record = ObservationRecord(
            session_id=self.session_id,
            content=self._validate_non_empty("content", content),
            observation_type=observation_type,  # type: ignore[arg-type]
            tags=tags or [],
        )
        conn.execute(
            """
            INSERT INTO observations
                (id, user_id, session_id, content, observation_type, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                self.user_id,
                record.session_id,
                record.content,
                record.observation_type,
                json.dumps(record.tags),
                record.created_at.isoformat(),
            ),
        )
        conn.commit()
        return record.id

    async def get_observations(
        self,
        limit: int = 20,
        tags: list[str] | None = None,
    ) -> list[ObservationRecord]:
        conn = await self._ensure_db()
        rows = conn.execute(
            """
            SELECT id, session_id, content, observation_type, tags, created_at
            FROM observations
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (self.user_id, limit),
        ).fetchall()

        records = [
            ObservationRecord(
                id=row["id"],
                session_id=row["session_id"],
                content=row["content"],
                observation_type=row["observation_type"],
                tags=json.loads(row["tags"] or "[]"),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

        if tags:
            tag_set = set(tags)
            records = [r for r in records if tag_set.intersection(r.tags)]

        return records

    async def _ensure_db(self) -> sqlite3.Connection:
        await self._ensure_connected()
        if self._conn is None:
            raise RuntimeError("SQLite context is not connected.")
        return self._conn

    @staticmethod
    def _validate_non_empty(field: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be str, got {type(value).__name__}")
        clean = value.strip()
        if not clean:
            raise ValueError(f"Missing required context field: {field}")
        return clean

    @staticmethod
    def _row_to_block(row: sqlite3.Row, score: float) -> ContextBlock:
        return ContextBlock(
            session_id=row["session_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            content=row["summary"],
            score=score,
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
