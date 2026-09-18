"""SQLite-backed routine registry with lightweight local embeddings."""

from __future__ import annotations

import json
import math
import sqlite3
import typing as t
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from ...base.embeddings import get_lightweight_embedding
from ...base.routines import CoreRoutineRegistry, RoutineToolMode
from ...core import RoutineBlocks
from ...types.routines import RoutineSummary


class SQLiteRoutineRegistryConfig(BaseModel):
    base_path: str
    routines: list[str] | None = None
    tool_mode: RoutineToolMode = RoutineToolMode.FULL
    db_name: str = "routines.sqlite3"


class SQLiteRoutineRegistry(CoreRoutineRegistry):
    """SQLite-backed catalog of authorized routines.

    Routines are stored in ``<base_path>/backend-local/<db_name>`` and
    searched using the same lightweight embedding helper used by the
    SQLite memory/context backends. When ``routines`` is provided, it
    acts as an authorization whitelist. When omitted, all routines in
    the database are visible.
    """

    component_schema = SQLiteRoutineRegistryConfig
    component_type = "routines"

    def __init__(
        self,
        base_path: str | Path,
        routines: list[str] | None = None,
        tool_mode: RoutineToolMode = RoutineToolMode.FULL,
        *,
        db_name: str = "routines.sqlite3",
    ) -> None:
        super().__init__(tool_mode=tool_mode)
        self.base_path = Path(base_path).expanduser().resolve()
        self.routines = (
            self._validate_routine_names(routines)
            if routines is not None
            else None
        )
        self.db_name = self.require_type(db_name, str, "db_name")
        self._conn: sqlite3.Connection | None = None

    def _to_config(self) -> SQLiteRoutineRegistryConfig:
        return SQLiteRoutineRegistryConfig(
            base_path=str(self.base_path),
            routines=list(self.routines) if self.routines is not None else None,
            tool_mode=self.tool_mode,
            db_name=self.db_name,
        )

    @classmethod
    def _from_config(cls, config: SQLiteRoutineRegistryConfig) -> "SQLiteRoutineRegistry":
        return cls(
            base_path=config.base_path,
            routines=config.routines,
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
            CREATE TABLE IF NOT EXISTS routines (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                instructions TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                vector TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    async def disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def upsert_routine(
        self,
        *,
        name: str,
        description: str,
        instructions: str,
    ) -> None:
        """Create or replace one routine in the SQLite catalog."""
        conn = await self._ensure_db()
        clean_name = self._validate_name(name)
        clean_description = self._validate_non_empty("description", description)
        clean_instructions = self._validate_non_empty("instructions", instructions)
        vector = get_lightweight_embedding(
            self._embedding_text(clean_name, clean_description, clean_instructions)
        )
        conn.execute(
            """
            INSERT INTO routines (name, description, instructions, updated_at, vector)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                description = excluded.description,
                instructions = excluded.instructions,
                updated_at = excluded.updated_at,
                vector = excluded.vector
            """,
            (
                clean_name,
                clean_description,
                clean_instructions,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(vector),
            ),
        )
        conn.commit()

    async def delete_routine(self, name: str) -> None:
        """Delete a routine from storage. Silent no-op when absent."""
        conn = await self._ensure_db()
        clean_name = self._validate_name(name)
        conn.execute("DELETE FROM routines WHERE name = ?", (clean_name,))
        conn.commit()

    async def get_catalog(self) -> list[RoutineSummary]:
        conn = await self._ensure_db()
        rows = conn.execute(
            f"""
            SELECT name, description
            FROM routines
            {self._authorization_where_clause()}
            ORDER BY name ASC
            """,
            self._authorization_params(),
        ).fetchall()
        return [
            RoutineSummary(name=row["name"], description=row["description"])
            for row in rows
        ]

    async def search(self, query: str, limit: int = 5) -> list[RoutineSummary]:
        conn = await self._ensure_db()
        clean_query = self._validate_non_empty("query", query)
        query_vector = get_lightweight_embedding(clean_query)

        rows = conn.execute(
            f"""
            SELECT name, description, vector
            FROM routines
            {self._authorization_where_clause()}
            """,
            self._authorization_params(),
        ).fetchall()

        ranked: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            score = self._cosine_similarity(query_vector, json.loads(row["vector"]))
            if score > 0:
                ranked.append((score, row))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RoutineSummary(name=row["name"], description=row["description"])
            for _, row in ranked[:limit]
        ]

    async def fetch(self, name: str) -> RoutineBlocks:
        conn = await self._ensure_db()
        clean_name = self._validate_name(name)
        if self.routines is not None and clean_name not in self.routines:
            raise ValueError(
                f"Unknown routine {clean_name!r}. "
                f"Available routines: {sorted(self.routines)}"
            )

        row = conn.execute(
            """
            SELECT name, description, instructions
            FROM routines
            WHERE name = ?
            LIMIT 1
            """,
            (clean_name,),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(
                f"Authorized routine {clean_name!r} is missing from SQLite storage."
            )
        return self._row_to_block(row)

    async def _ensure_db(self) -> sqlite3.Connection:
        await self._ensure_connected()
        if self._conn is None:
            raise RuntimeError("SQLite routine registry is not connected.")
        return self._conn


    def _authorization_where_clause(self) -> str:
        if self.routines is None:
            return ""
        return f"WHERE name IN ({self._placeholders(self.routines)})"

    def _authorization_params(self) -> tuple[str, ...]:
        return tuple(self.routines or ())

    @staticmethod
    def _validate_routine_names(routines: list[str]) -> list[str]:
        if not isinstance(routines, list):
            raise TypeError(f"routines must be a list, got {type(routines).__name__}")
        if not routines:
            raise ValueError("routines list cannot be empty")
        cleaned = [SQLiteRoutineRegistry._validate_name(name) for name in routines]
        duplicates = {name for name in cleaned if cleaned.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate routine names: {sorted(duplicates)}")
        return cleaned

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name).__name__}")
        clean = name.strip()
        if not clean:
            raise ValueError("name must be a non-empty, non-whitespace string")
        return clean

    @staticmethod
    def _validate_non_empty(field: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be str, got {type(value).__name__}")
        clean = value.strip()
        if not clean:
            raise ValueError(f"Missing required routine field: {field}")
        return clean

    @staticmethod
    def _embedding_text(name: str, description: str, instructions: str) -> str:
        return f"{name}\n{description}\n{instructions}"

    @staticmethod
    def _placeholders(values: t.Sequence[t.Any]) -> str:
        if not values:
            raise ValueError("Cannot build SQL IN clause for empty values.")
        return ", ".join("?" for _ in values)

    @staticmethod
    def _row_to_block(row: sqlite3.Row) -> RoutineBlocks:
        return RoutineBlocks(
            name=row["name"],
            description=row["description"],
            instructions=row["instructions"],
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
