"""write_todos keeps the agent's list of things to do in the run context."""

import pytest

from max_ai.capabilities.tools.todos import WriteTodosTool
from max_ai.core.event_type import TodosEvent
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord


@pytest.fixture
def ctx():
    return RunContext(user_id="user-1", session_id="s1")


async def write(ctx, todos, seen=None):
    record = ToolCallRecord(tool_name="write_todos", parameters={"todos": todos})
    emit = seen.append if seen is not None else None
    return await ToolDispatcher(ToolRegistry([WriteTodosTool()])).dispatch(record, ctx, emit)


def todo(content, status="pending"):
    return {"content": content, "status": status}


async def test_the_list_is_kept_in_the_context_and_shown_outside(ctx):
    seen = []
    result = await write(ctx, [todo("Read the schema", "done"), todo("Write it", "in_progress"), todo("Test")], seen)
    assert result.result == "List updated: 1 pending, 1 in progress, 1 done."
    assert [(t.content, t.status) for t in ctx.todos] == [
        ("Read the schema", "done"), ("Write it", "in_progress"), ("Test", "pending")]
    shown = [event for event in seen if isinstance(event, TodosEvent)]
    assert len(shown) == 1 and shown[0].todos == ctx.todos


async def test_a_new_list_replaces_the_one_before(ctx):
    await write(ctx, [todo("a"), todo("b")])
    await write(ctx, [todo("c", "done")])
    assert [t.content for t in ctx.todos] == ["c"]


async def test_the_same_list_again_shows_nothing(ctx):
    seen = []
    await write(ctx, [todo("a")])
    result = await write(ctx, [todo("a")], seen)
    assert result.result == "The list is the same as before."
    assert not [event for event in seen if isinstance(event, TodosEvent)]


@pytest.mark.parametrize(
    "todos, reason",
    [
        ([todo("a", "in_progress"), todo("b", "in_progress")], "Only one thing to do can be in_progress"),
        ([todo("a"), todo(" A ")], "Two things to do are the same"),
        ([todo("  ")], "has no text"),
        ([todo("a", "active")], "is not one of"),
        ([{"content": "a"}], "'status' is a required property"),
    ],
)
async def test_a_wrong_list_says_why_and_changes_nothing(ctx, todos, reason):
    await write(ctx, [todo("kept")])
    result = await write(ctx, todos)
    assert not result.success and reason in result.error
    assert [t.content for t in ctx.todos] == ["kept"]
