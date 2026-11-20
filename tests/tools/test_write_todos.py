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


# -------- THE USER ACCEPTS THE PLAN ---------------------------------------------------
async def propose(ctx, todos):
    """A call to a tool that asks for approval: the record, as the dispatcher left it."""
    record = ToolCallRecord(tool_name="write_todos", parameters={"todos": todos})
    ctx.tool_state.records[record.id] = record
    await ToolDispatcher(ToolRegistry([WriteTodosTool(approval=True)])).dispatch(record, ctx, None)
    return record


async def test_a_new_plan_waits_for_the_user_and_is_not_the_list_yet(ctx):
    record = await propose(ctx, [todo("a"), todo("b")])
    assert record.is_awaiting_input and ctx.todos == []

    record.store_result(WriteTodosTool.answer(record, ctx, accepted=True))
    assert record.result.result == WriteTodosTool.ACCEPTED
    assert [t.content for t in ctx.todos] == ["a", "b"]


async def test_a_rejected_plan_never_becomes_the_list_and_says_what_the_user_said(ctx):
    record = await propose(ctx, [todo("a")])
    result = WriteTodosTool.answer(record, ctx, accepted=False, comment=" start with the tests ")
    assert result.result == "The user did not accept the plan. They said: start with the tests"
    assert WriteTodosTool.answer(record, ctx, accepted=False).result == WriteTodosTool.REJECTED
    assert ctx.todos == []


async def test_marking_progress_or_emptying_the_list_never_asks(ctx):
    record = await propose(ctx, [todo("a"), todo("b")])
    record.store_result(WriteTodosTool.answer(record, ctx, accepted=True))

    assert not (await propose(ctx, [todo("a", "done"), todo("b", "in_progress")])).is_awaiting_input
    assert [t.status for t in ctx.todos] == ["done", "in_progress"]
    assert not (await propose(ctx, [])).is_awaiting_input
    # Other steps are a new plan.
    assert (await propose(ctx, [todo("a", "done"), todo("c")])).is_awaiting_input


def test_the_model_is_told_about_the_approval_only_when_it_is_on():
    assert WriteTodosTool.APPROVAL in WriteTodosTool(approval=True).description
    assert WriteTodosTool.APPROVAL not in WriteTodosTool().description
