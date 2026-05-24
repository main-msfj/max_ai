from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from max_ai.base.memory import MemoryToolMode
from max_ai.capabilities.memory import SQLiteMemoryRegistry
from max_ai.errors.memory import MemoryError


def fake_embedding(text: str) -> list[float]:
    lowered = text.lower()
    return [
        1.0 if "python" in lowered else 0.0,
        1.0 if "spanish" in lowered else 0.0,
        1.0 if "project" in lowered else 0.0,
    ]


@pytest.mark.asyncio
async def test_sqlite_memory_registry_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "max_ai.capabilities.memory.sqlite.get_lightweight_embedding",
        fake_embedding,
    )
    mem = SQLiteMemoryRegistry(user_id="u1", base_path=tmp_path)

    async with mem:
        assert await mem.get_context() == []

        await mem.upsert_memory(
            key="language",
            category="preference",
            content="Prefers Spanish for casual conversation.",
        )
        await mem.update_fact("project", "Working on a Python agent framework.")

        facts = await mem.list_facts()
        by_content = {fact.content: fact for fact in facts}
        assert by_content["Prefers Spanish for casual conversation."].category == "preference"
        assert by_content["Working on a Python agent framework."].category == "project"

        results = await mem.search("python framework", limit=1)
        assert len(results) == 1
        assert results[0].category == "project"

        await mem.delete_fact("language")
        after_delete = await mem.list_facts()
        assert len(after_delete) == 1
        assert after_delete[0].category == "project"

    db_path = tmp_path / "backend-local" / "memory.sqlite3"
    assert db_path.is_file()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT user_id, key, category, content, vector FROM memory").fetchall()
    assert rows == [
        (
            "u1",
            "project",
            "project",
            "Working on a Python agent framework.",
            "[1.0, 0.0, 1.0]",
        )
    ]


@pytest.mark.asyncio
async def test_sqlite_memory_search_can_filter_by_category(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "max_ai.capabilities.memory.sqlite.get_lightweight_embedding",
        fake_embedding,
    )
    mem = SQLiteMemoryRegistry(user_id="u1", base_path=tmp_path)

    async with mem:
        await mem.upsert_memory(
            key="language",
            category="preference",
            content="Prefers Spanish.",
        )
        await mem.upsert_memory(
            key="project",
            category="work",
            content="Python project.",
        )

        results = await mem.search("python spanish project", category="preference")

    assert len(results) == 1
    assert results[0].content == "Prefers Spanish."


@pytest.mark.asyncio
async def test_sqlite_memory_exposes_category_update_tool(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "max_ai.capabilities.memory.sqlite.get_lightweight_embedding",
        fake_embedding,
    )
    mem = SQLiteMemoryRegistry(
        user_id="u1",
        base_path=tmp_path,
        tool_mode=MemoryToolMode.FULL,
    )

    tool_names = {tool.name for tool in mem.tools}

    assert tool_names == {
        "list_memories",
        "search_memories",
        "update_memory",
        "delete_memory",
    }
    update_tool = next(tool for tool in mem.tools if tool.name == "update_memory")
    assert set(update_tool.parameters["properties"]) == {"key", "category", "content"}


@pytest.mark.asyncio
async def test_sqlite_memory_rejects_empty_required_fields(tmp_path: Path):
    mem = SQLiteMemoryRegistry(user_id="u1", base_path=tmp_path)

    async with mem:
        with pytest.raises(MemoryError, match="Missing required memory field: category"):
            await mem.upsert_memory(key="language", category=" ", content="Spanish")


@pytest.mark.asyncio
async def test_sqlite_memory_context_only_includes_recent_memories(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "max_ai.capabilities.memory.sqlite.get_lightweight_embedding",
        fake_embedding,
    )
    mem = SQLiteMemoryRegistry(user_id="u1", base_path=tmp_path, context_days=30)

    async with mem:
        await mem.upsert_memory(
            key="recent",
            category="preference",
            content="Prefers Spanish.",
        )
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        conn = await mem._ensure_db()
        conn.execute(
            """
            INSERT INTO memory (user_id, key, category, content, last_updated, vector)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "u1",
                "old",
                "project",
                "Old Python project.",
                old_timestamp,
                json.dumps(fake_embedding("Old Python project.")),
            ),
        )
        conn.commit()

        context = await mem.get_context()
        search_results = await mem.search("old python project")

    assert [memory.content for memory in context] == ["Prefers Spanish."]
    assert any(memory.content == "Old Python project." for memory in search_results)
