"""
Integration tests for ToolExecutor with real FunctionAsTool.

End-to-end flow: Python function → wrapped as FunctionAsTool →
executed by ToolExecutor → record consumed → message + events.

These tests don't mock — if they pass, the executor + tool integration
is solid.
"""

from __future__ import annotations

import asyncio

import pytest

import typing as t
from max_ai.base.tools import ToolContext
from max_ai.base.tool_executor import ToolExecutor
from max_ai.tools.function_as_tool import FunctionAsTool
from max_ai.core.event_type import (
    ToolCallEvent,
    ToolApprovalEvent,
    ToolCallResponseEvent,
)
from max_ai.core.messages import ToolMessage
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode
from max_ai.types.run_context import RunContext


# -------- HELPERS -----------------------------------------------------------
def make_record(tool_name: str, **overrides: t.Any) -> ToolCallRecord:
    base: dict[str, t.Any] = {"tool_name": tool_name, "parameters": {}}
    base.update(overrides)
    return ToolCallRecord(**base)


async def collect(gen) -> list:
    return [item async for item in gen]


# -------- BASIC: SYNC FUNCTION -----------------------------------------------------------
@pytest.mark.asyncio
async def test_sync_function_executes_end_to_end():
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tool = FunctionAsTool(add, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    executor = ToolExecutor(tools=[tool], agent_name="test")
    record = make_record("add", parameters={"a": 3, "b": 4})

    ctx = RunContext()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # Event + Message + Response
    assert len(items) == 3
    msg = items[1]
    assert isinstance(msg, ToolMessage)
    assert msg.success is True
    assert "7" in msg.content

    assert record.is_consumed
    assert record.result is not None
    assert record.result.result == 7


# -------- ASYNC FUNCTION -----------------------------------------------------------
@pytest.mark.asyncio
async def test_async_function_executes_end_to_end():
    async def fetch(url: str) -> str:
        """Pretend to fetch a URL."""
        await asyncio.sleep(0.01)
        return f"<content of {url}>"

    tool = FunctionAsTool(fetch, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    executor = ToolExecutor(tools=[tool], agent_name="test")
    record = make_record("fetch", parameters={"url": "https://example.com"})

    ctx = RunContext()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    msg = items[1]
    assert isinstance(msg, ToolMessage)
    assert msg.success is True
    assert "example.com" in msg.content

    assert record.is_consumed


# -------- VALIDATION FAILURE FROM REAL SCHEMA -----------------------------------------------------------
@pytest.mark.asyncio
async def test_validation_fails_when_required_param_missing():
    def echo(text: str) -> str:
        """Echo text."""
        return text

    tool = FunctionAsTool(echo, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    executor = ToolExecutor(tools=[tool], agent_name="test")
    record = make_record("echo", parameters={})  # missing 'text'

    ctx = RunContext()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # ToolCallEvent + ToolMessage(error) + ToolCallResponseEvent(failure)
    assert len(items) == 3
    msg = items[1]
    assert isinstance(msg, ToolMessage)
    assert msg.success is False

    # Tool wasn't called → record never consumed.
    assert not record.is_consumed


@pytest.mark.asyncio
async def test_validation_fails_when_param_wrong_type():
    def square(n: int) -> int:
        """Square an integer."""
        return n * n

    tool = FunctionAsTool(square, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    executor = ToolExecutor(tools=[tool], agent_name="test")
    record = make_record("square", parameters={"n": "not an int"})

    ctx = RunContext()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    response = [i for i in items if isinstance(i, ToolCallResponseEvent)][0]
    assert response.tool_result is not None
    assert response.tool_result.success is False


# -------- FUNCTION RAISES -----------------------------------------------------------
@pytest.mark.asyncio
async def test_function_raising_exception_becomes_failure():
    def divide(a: int, b: int) -> float:
        """Divide a by b."""
        return a / b

    tool = FunctionAsTool(divide, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    executor = ToolExecutor(tools=[tool], agent_name="test")
    record = make_record("divide", parameters={"a": 10, "b": 0})

    ctx = RunContext()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    msg = items[1]
    assert isinstance(msg, ToolMessage)
    assert msg.success is False
    assert "ZeroDivisionError" in (msg.error or "") or "division" in (msg.error or "").lower()

    assert record.is_consumed
    assert record.result is not None
    assert record.result.success is False


# -------- ASK_APPROVED FLOW -----------------------------------------------------------
@pytest.mark.asyncio
async def test_ask_approved_pauses_for_user_input():
    def delete_user(user_id: str) -> str:
        """Delete a user."""
        return f"deleted {user_id}"

    tool = FunctionAsTool(delete_user, approval_mode=ToolApprovalMode.ASK_APPROVED)
    executor = ToolExecutor(tools=[tool], agent_name="test")
    record = make_record("delete_user", parameters={"user_id": "u_42"})

    ctx = RunContext()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # ToolCallEvent + ToolApprovalEvent (pause).
    assert len(items) == 2
    assert isinstance(items[0], ToolCallEvent)
    assert isinstance(items[1], ToolApprovalEvent)

    assert record.is_pending_approval


@pytest.mark.asyncio
async def test_ask_approved_after_user_approves_runs():
    def delete_user(user_id: str) -> str:
        """Delete a user."""
        return f"deleted {user_id}"

    tool = FunctionAsTool(delete_user, approval_mode=ToolApprovalMode.ASK_APPROVED)
    executor = ToolExecutor(tools=[tool], agent_name="test")
    record = make_record("delete_user", parameters={"user_id": "u_42"})
    record.approve(reason="confirmed by user")

    ctx = RunContext()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    assert len(items) == 3
    assert isinstance(items[2], ToolCallResponseEvent)
    assert items[2].tool_result is not None
    assert items[2].tool_result.success is True

    assert record.is_consumed
    assert record.was_auto_approved is False  # was manually approved


# -------- PARALLEL REAL FUNCTIONS -----------------------------------------------------------
@pytest.mark.asyncio
async def test_parallel_two_real_async_functions():
    async def slow_a() -> str:
        await asyncio.sleep(0.05)
        return "A done"

    async def slow_b() -> str:
        await asyncio.sleep(0.05)
        return "B done"

    tool_a = FunctionAsTool(slow_a, name="a", approval_mode=ToolApprovalMode.AUTO_APPROVED)
    tool_b = FunctionAsTool(slow_b, name="b", approval_mode=ToolApprovalMode.AUTO_APPROVED)
    executor = ToolExecutor(tools=[tool_a, tool_b], agent_name="test")

    rec_a = make_record("a")
    rec_b = make_record("b")
    ctx = RunContext()
    ctx.tool_state.add(rec_a)
    ctx.tool_state.add(rec_b)

    items = await collect(executor.execute_tool_call(ctx, [rec_a, rec_b]))

    # Each: Event + Message + Response → 6 total.
    assert len(items) == 6
    assert rec_a.is_consumed
    assert rec_b.is_consumed
    assert rec_a.result is not None and rec_a.result.success is True
    assert rec_b.result is not None and rec_b.result.success is True


# -------- TOOLCONTEXT INJECTION -----------------------------------------------------------
@pytest.mark.asyncio
async def test_function_receives_tool_context():
    """Function declares ToolContext as first arg → executor injects it."""

    captured: dict[str, t.Any] = {}

    def with_ctx(ctx: ToolContext, message: str) -> str:
        """Greet with context info."""
        captured["run_id"] = ctx.run_id
        captured["session_id"] = ctx.session_id
        return f"hi: {message}"

    tool = FunctionAsTool(with_ctx, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    executor = ToolExecutor(tools=[tool], agent_name="test")
    record = make_record("with_ctx", parameters={"message": "world"})

    ctx = RunContext()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    msg = items[1]
    assert isinstance(msg, ToolMessage)
    assert msg.success is True
    assert "hi: world" in msg.content

    # Verify the ToolContext was injected with the run_id.
    assert captured["run_id"] == ctx.run_id


# -------- LOAD_FROM RESUME WORKFLOW -----------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_after_user_approves_pending_tool():
    """End-to-end resume: tool was pending in run #1, user approves, run #2 executes it."""

    def send_email(to: str, subject: str) -> str:
        """Send an email."""
        return f"email sent to {to}: {subject}"

    tool = FunctionAsTool(send_email, approval_mode=ToolApprovalMode.ASK_APPROVED)
    executor = ToolExecutor(tools=[tool], agent_name="test")

    # Run #1: tool is requested, ends up pending.
    ctx_run1 = RunContext()
    record = make_record(
        "send_email", parameters={"to": "alice@x.com", "subject": "Hi"}
    )
    ctx_run1.tool_state.add(record)

    items_run1 = await collect(executor.execute_tool_call(ctx_run1, [record]))
    assert any(isinstance(i, ToolApprovalEvent) for i in items_run1)
    assert record.is_pending_approval

    # Persist + rehydrate.
    serialized = ctx_run1.tool_state.model_dump()

    # Run #2: rehydrate state, apply approval, re-run executor on the
    # actionable record.
    from max_ai.types.tool_call import ToolCallRecord  # already imported but kept explicit
    from max_ai.core.tool_state import ToolState

    state_run2 = ToolState.load_from(
        serialized,
        approval=(record.id, True, "user said yes"),
    )
    ctx_run2 = RunContext()
    ctx_run2.tool_state = state_run2

    actionable = state_run2.actionable_calls
    assert len(actionable) == 1

    items_run2 = await collect(executor.execute_tool_call(ctx_run2, actionable))

    response = [i for i in items_run2 if isinstance(i, ToolCallResponseEvent)][0]
    assert response.tool_result is not None
    assert response.tool_result.success is True

    final_record = state_run2.get(record.id)
    assert final_record is not None
    assert final_record.is_consumed
    assert "alice@x.com" in str(final_record.result.result)  # type: ignore[union-attr]