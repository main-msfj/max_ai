"""Behavioral checks for the session-memory contract, independent of backends."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from max_ai.base.memory import (
    CoreMemoryRegistry,
    MemoryRecord,
    MemorySearchResult,
    MemoryToolMode,
)
from max_ai.errors.memory import MemoryError
from max_ai.types.tool_call import ToolCallRecord


class SessionMemory(CoreMemoryRegistry):
    def __init__(self, store, user_id="user", session_id="current", **kwargs):
        super().__init__(user_id, session_id, **kwargs)
        self.store = store
        self.connect_count = 0

    async def connect(self):
        self.connect_count += 1

    async def disconnect(self):
        pass

    async def _read_session(self):
        return list(self.store.get((self.user_id, self.session_id), {}).values())

    async def _read_user(self):
        return [
            MemorySearchResult(**record.model_dump(), session_id=session)
            for (user, session), entries in self.store.items() if user == self.user_id
            for record in entries.values()
        ]

    async def _write_memory(self, record):
        entries = self.store.setdefault((self.user_id, self.session_id), {})
        created = record.category not in entries
        entries[record.category] = record
        return created

    async def _delete_memory(self, category):
        entries = self.store.get((self.user_id, self.session_id), {})
        return entries.pop(category, None) is not None

    async def _search_memory(self, text):
        # Deterministic stand-in: production backends own semantic retrieval.
        return [
            MemorySearchResult(**record.model_dump(), session_id=session)
            for (user, session), entries in self.store.items()
            if user == self.user_id and session != self.session_id
            for record in entries.values()
            if text in record.memory
        ]


async def test_replacement_and_deletion_are_scoped_to_session():
    store = {}
    current = SessionMemory(store)
    other = SessionMemory(store, session_id="other")
    assert await current.create_or_update(" project ", "first") == "New memory category created: project"
    await other.create_or_update("project", "other session")
    assert await current.create_or_update("project", "replacement") == "Memory updated: project"
    assert [(r.category, r.memory) for r in await current.get_context()] == [("project", "replacement")]
    assert await current.delete_memory("project") == "Memory deleted: project"
    assert await current.delete_memory("project") == "Memory category not found: project"
    assert [r.memory for r in await current.get_context()] == ["other session"]  # the user's other session
    assert (await other.get_context())[0].memory == "other session"
    assert current.connect_count == 1


async def test_context_window_does_not_hide_categories_or_delete_storage():
    old = MemoryRecord(category="old", memory="decision", updated=datetime.now(timezone.utc) - timedelta(days=45))
    store = {("user", "current"): {"old": old}}
    mem = SessionMemory(store)
    await mem.create_or_update("recent", "current decision")
    assert [r.category for r in await mem.get_context()] == ["recent"]
    assert await mem.list_category() == ["old", "recent"]
    all_dates = SessionMemory(store, context_days=None)
    assert [r.category for r in await all_dates.get_context()] == ["recent", "old"]


async def test_search_delegates_and_preserves_provenance_without_copying():
    store = {}
    mem = SessionMemory(store)
    await mem.create_or_update("project", "python current")
    await SessionMemory(store, session_id="past").create_or_update("project", "python historical")
    await SessionMemory(store, user_id="someone_else", session_id="past").create_or_update("project", "python private")
    results = await mem.search_memory(" python ")
    assert [(r.session_id, r.memory) for r in results] == [("past", "python historical")]
    assert [r.memory for r in await mem.get_context()] == ["python current"]
    assert await mem.search_memory("missing") == []


@pytest.mark.parametrize("mode,names", [
    (MemoryToolMode.NONE, set()),
    (MemoryToolMode.READ_ONLY, {"get_context", "list_category", "search_memory"}),
    (MemoryToolMode.FULL, {"get_context", "list_category", "search_memory", "create_or_update", "delete_memory"}),
])
async def test_tool_modes(mode, names):
    assert {tool.name for tool in SessionMemory({}, tool_mode=mode).tools} == names


async def test_tools_execute_and_return_serializable_session_context():
    mem = SessionMemory({})
    tools = {tool.name: tool for tool in mem.tools}
    assert set(tools["create_or_update"].parameters["properties"]) == {"category", "memory"}
    saved = await tools["create_or_update"].execute(ToolCallRecord(
        id="save", tool_name="create_or_update", parameters={"category": "language", "memory": "Spanish"},
    ))
    assert saved.success
    context = await tools["get_context"].execute(ToolCallRecord(
        id="context", tool_name="get_context", parameters={},
    ))
    assert context.success
    payload = json.loads(json.dumps(context.result))
    assert payload["session_id"] == "current"
    assert payload["memories"][0]["memory"] == "Spanish"
    assert datetime.fromisoformat(payload["memories"][0]["updated"]).tzinfo is not None


async def test_blank_inputs_do_not_write_or_connect():
    mem = SessionMemory({})
    with pytest.raises(MemoryError):
        await mem.create_or_update(" ", "text")
    with pytest.raises(MemoryError):
        await mem.create_or_update("category", " ")
    with pytest.raises(MemoryError):
        await mem.search_memory(" ")
    with pytest.raises(MemoryError):
        await mem.delete_memory(" ")
    assert mem.store == {}
    assert mem.connect_count == 0


async def test_context_spans_the_users_sessions_and_the_current_one_wins():
    now = datetime.now(timezone.utc)
    store = {
        ("user", "old"): {"name": MemoryRecord(category="name", memory="Marvin", updated=now - timedelta(days=3)),
                          "team": MemoryRecord(category="team", memory="Agents", updated=now - timedelta(days=3))},
        ("user", "newer"): {"team": MemoryRecord(category="team", memory="Platform", updated=now - timedelta(days=1))},
        ("user", "current"): {"name": MemoryRecord(category="name", memory="Marv", updated=now - timedelta(days=5))},
        ("someone_else", "x"): {"secret": MemoryRecord(category="secret", memory="private", updated=now)},
    }
    context = {r.category: r.memory for r in await SessionMemory(store).get_context()}
    assert context == {"name": "Marv", "team": "Platform"}  # current session first, else newest
