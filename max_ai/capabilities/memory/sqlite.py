"""SQLite-backed memory registry with lightweight local embeddings."""

from __future__ import annotations

import json
import math
import sqlite3
import typing as t
from datetime import datetime, timezone
from pathlib import Path

from ...base.embeddings import get_lightweight_embedding
from ...base.memory import CoreMemoryRegistry, MemoryToolMode
from ...base.tools import CoreTool
from ...core import MemoryBlock
from ...errors.memory import MemoryError
from ...tools import FunctionAsTool
from ...types.tools import ToolApprovalMode


class SQLiteMemoryRegistry(CoreMemoryRegistry):
    """SQLite-backed durable memory for a single user.

    The table is intentionally simple so it can later be paired with
    ``sqlite-vec`` without changing the public capability contract.
    """

    def __init__(
        self,
        user_id: str,
        base_path: str | Path,
        tool_mode: MemoryToolMode = MemoryToolMode.FULL,
        *,
        db_name: str = "memory.sqlite3",
    ) -> None:
        super().__init__(user_id=user_id, tool_mode=tool_mode)
        self.base_path = Path(base_path).expanduser().resolve()
        self.db_name = self.require_type(db_name, str, "db_name")
        self._conn: sqlite3.Connection | None = None

    @property
    def db_path(self) -> Path:
        return self.base_path / "backend-local" / self.db_name

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                vector TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_user_category "
            "ON memory (user_id, category)"
        )
        self._conn.commit()

    async def disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def get_context(self) -> list[MemoryBlock]:
        return await self.list_facts()

    async def list_facts(self) -> list[MemoryBlock]:
        conn = await self._ensure_db()
        rows = conn.execute(
            """
            SELECT category, content, last_updated
            FROM memory
            WHERE user_id = ?
            ORDER BY last_updated DESC
            """,
            (self.user_id,),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    async def update_fact(self, key: str, value: str) -> None:
        clean_key = self._validate_non_empty("key", key)
        clean_value = self._validate_non_empty("value", value)
        await self.upsert_memory(
            key=clean_key,
            category=clean_key,
            content=clean_value,
        )

    async def upsert_memory(self, *, key: str, category: str, content: str) -> None:
        conn = await self._ensure_db()
        clean_key = self._validate_non_empty("key", key)
        clean_category = self._validate_non_empty("category", category)
        clean_content = self._validate_non_empty("content", content)
        now = datetime.now(timezone.utc).isoformat()
        vector = get_lightweight_embedding(f"{clean_category}\n{clean_content}")
        conn.execute(
            """
            INSERT INTO memory (user_id, key, category, content, last_updated, vector)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                category = excluded.category,
                content = excluded.content,
                last_updated = excluded.last_updated,
                vector = excluded.vector
            """,
            (
                self.user_id,
                clean_key,
                clean_category,
                clean_content,
                now,
                json.dumps(vector),
            ),
        )
        conn.commit()

    async def delete_fact(self, key: str) -> None:
        conn = await self._ensure_db()
        clean_key = self._validate_non_empty("key", key)
        conn.execute(
            "DELETE FROM memory WHERE user_id = ? AND key = ?",
            (self.user_id, clean_key),
        )
        conn.commit()

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        category: str | None = None,
    ) -> list[MemoryBlock]:
        conn = await self._ensure_db()
        clean_query = self._validate_non_empty("query", query)
        clean_category = (
            category.strip() if isinstance(category, str) and category.strip() else None
        )
        query_vector = get_lightweight_embedding(clean_query)

        sql = """
            SELECT category, content, last_updated, vector
            FROM memory
            WHERE user_id = ?
        """
        params: list[t.Any] = [self.user_id]
        if clean_category is not None:
            sql += " AND category = ?"
            params.append(clean_category)

        rows = conn.execute(sql, params).fetchall()
        ranked: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            score = self._cosine_similarity(query_vector, json.loads(row["vector"]))
            if score > 0:
                ranked.append((score, row))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [self._row_to_memory(row) for _, row in ranked[:limit]]

    def as_tools(self) -> list[CoreTool]:
        if self.tool_mode == MemoryToolMode.NONE:
            return []
        if self.tool_mode == MemoryToolMode.READ_ONLY:
            return [self._build_list_tool(), self._build_search_tool()]
        return [
            self._build_list_tool(),
            self._build_search_tool(),
            self._build_update_tool(),
            self._build_delete_tool(),
        ]

    def _build_search_tool(self) -> CoreTool:
        async def search_memories(
            query: str,
            limit: int = 5,
            category: str | None = None,
        ) -> list[dict[str, t.Any]]:
            """Search durable user memories by semantic meaning."""
            await self._ensure_connected()
            memories = await self.search(query, limit=limit, category=category)
            return [memory.model_dump() for memory in memories]

        return FunctionAsTool(
            search_memories,
            name="search_memories",
            description=(
                "Search durable memories about this user by semantic meaning. "
                "Use this before creating a memory if you need to avoid "
                "duplicating or contradicting existing user facts."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    def _build_update_tool(self) -> CoreTool:
        async def update_memory(key: str, category: str, content: str) -> str:
            """Create or update a durable memory about the user.

            Args:
                key: Stable short identifier for this fact.
                category: Type of memory, such as preference, profile, project,
                    goal, constraint, or relationship.
                content: The durable user fact to remember.
            """
            await self._ensure_connected()
            await self.upsert_memory(key=key, category=category, content=content)
            return f"Memory saved: {key}"

        return FunctionAsTool(
            update_memory,
            name="update_memory",
            description=(
                "Store or overwrite a durable fact about the user. Provide a "
                "stable key, a category, and concise content. Use this for "
                "long-lived preferences, user profile details, recurring goals, "
                "projects, constraints, or relationships. Do not store transient "
                "conversation details."
            ),
            approval_mode=ToolApprovalMode.ASK_APPROVED,
        )

    async def _ensure_db(self) -> sqlite3.Connection:
        await self._ensure_connected()
        if self._conn is None:
            raise MemoryError("SQLite memory is not connected.")
        return self._conn

    @staticmethod
    def _validate_non_empty(field: str, value: str) -> str:
        if not isinstance(value, str):
            raise MemoryError.invalid_type(field, "str", type(value).__name__)
        clean = value.strip()
        if not clean:
            raise MemoryError.missing(field)
        return clean

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryBlock:
        return MemoryBlock(
            category=row["category"],
            content=row["content"],
            last_updated=datetime.fromisoformat(row["last_updated"]),
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
