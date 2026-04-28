"""Tests for FileSystemRunContextStore."""

from __future__ import annotations

import pytest

from max_ai.persistence.filesystem import FileSystemRunContextStore
from max_ai.persistence.core import validate_run_id
from max_ai.types.run_context import RunContext
from max_ai.core.messages import UserMessage
from max_ai.core.tool_state import ToolCallRecord


# -------- VALIDATION -----------------------------------------------------------
@pytest.mark.parametrize("good", ["abc123", "run-id_42", "DEADBEEF", "a"])
def test_validate_run_id_accepts_safe_strings(good):
    validate_run_id(good)  # no raise


@pytest.mark.parametrize(
    "bad", ["", "../etc/passwd", "with spaces", "with/slash", "dot.dot", "tab\t"]
)
def test_validate_run_id_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        validate_run_id(bad)


def test_validate_run_id_rejects_non_string():
    with pytest.raises(ValueError):
        validate_run_id(123)  # type: ignore[arg-type]


# -------- FIXTURES -----------------------------------------------------------
@pytest.fixture
def store(tmp_path):
    return FileSystemRunContextStore(base_path=tmp_path)


@pytest.fixture
def ctx_with_message():
    ctx = RunContext()
    ctx.messages.append(UserMessage(source="user", content="hello world"))
    return ctx


# -------- BASIC CRUD -----------------------------------------------------------
@pytest.mark.asyncio
async def test_save_creates_file(store, ctx_with_message, tmp_path):
    await store.save(ctx_with_message.run_id, ctx_with_message)
    expected = tmp_path / f"{ctx_with_message.run_id}.json"
    assert expected.exists()
    assert expected.read_text(encoding="utf-8")  # non-empty


@pytest.mark.asyncio
async def test_load_returns_none_when_missing(store):
    assert await store.load("does_not_exist") is None


@pytest.mark.asyncio
async def test_save_load_round_trip(store, ctx_with_message):
    await store.save(ctx_with_message.run_id, ctx_with_message)
    loaded = await store.load(ctx_with_message.run_id)

    assert loaded is not None
    assert loaded.run_id == ctx_with_message.run_id
    assert len(loaded.messages) == 1
    assert isinstance(loaded.messages[0], UserMessage)
    assert loaded.messages[0].text() == "hello world"


@pytest.mark.asyncio
async def test_save_overwrites(store, ctx_with_message):
    await store.save(ctx_with_message.run_id, ctx_with_message)

    ctx_with_message.messages.append(UserMessage(source="user", content="second"))
    await store.save(ctx_with_message.run_id, ctx_with_message)

    loaded = await store.load(ctx_with_message.run_id)
    assert len(loaded.messages) == 2


@pytest.mark.asyncio
async def test_delete_removes_file(store, ctx_with_message):
    await store.save(ctx_with_message.run_id, ctx_with_message)
    await store.delete(ctx_with_message.run_id)
    assert await store.load(ctx_with_message.run_id) is None


@pytest.mark.asyncio
async def test_delete_missing_is_noop(store):
    # Must not raise.
    await store.delete("never_existed")


# -------- LIST -----------------------------------------------------------
@pytest.mark.asyncio
async def test_list_empty(store):
    assert await store.list() == []


@pytest.mark.asyncio
async def test_list_returns_saved_ids(store):
    ctx_a = RunContext()
    ctx_b = RunContext()
    await store.save(ctx_a.run_id, ctx_a)
    await store.save(ctx_b.run_id, ctx_b)

    ids = await store.list()
    assert set(ids) == {ctx_a.run_id, ctx_b.run_id}


@pytest.mark.asyncio
async def test_list_ignores_non_json_files(store, tmp_path):
    ctx = RunContext()
    await store.save(ctx.run_id, ctx)
    (tmp_path / "noise.txt").write_text("ignore me")
    (tmp_path / "subdir").mkdir()

    ids = await store.list()
    assert ids == [ctx.run_id]


# -------- VALIDATION AT THE BOUNDARY -----------------------------------------------------------
@pytest.mark.asyncio
async def test_save_rejects_unsafe_run_id(store, ctx_with_message):
    with pytest.raises(ValueError):
        await store.save("../escape", ctx_with_message)


@pytest.mark.asyncio
async def test_load_rejects_unsafe_run_id(store):
    with pytest.raises(ValueError):
        await store.load("../escape")


@pytest.mark.asyncio
async def test_delete_rejects_unsafe_run_id(store):
    with pytest.raises(ValueError):
        await store.delete("../escape")


# -------- DIRECTORY SETUP -----------------------------------------------------------
def test_init_creates_base_path(tmp_path):
    target = tmp_path / "nested" / "runs"
    assert not target.exists()
    FileSystemRunContextStore(base_path=target)
    assert target.exists() and target.is_dir()


# -------- MALFORMED PERSISTENCE -----------------------------------------------------------
@pytest.mark.asyncio
async def test_load_raises_on_corrupt_file(store, tmp_path):
    bad_path = tmp_path / "corrupt.json"
    bad_path.write_text("{not valid json")
    with pytest.raises(Exception):
        await store.load("corrupt")

async def test_round_trip_preserves_tool_state(store):
    ctx = RunContext()
    ctx.messages.append(UserMessage(source="user", content="do X"))
    record = ToolCallRecord(tool_name="get_weather", parameters={"city": "Tokyo"})
    ctx.tool_state.add(record)
    
    await store.save(ctx.run_id, ctx)
    loaded = await store.load(ctx.run_id)
    
    assert loaded.tool_state.has(record.id)
    assert loaded.tool_state.get(record.id).tool_name == "get_weather"
    assert loaded.tool_state.get(record.id).is_pending_approval