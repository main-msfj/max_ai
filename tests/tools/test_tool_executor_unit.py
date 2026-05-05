"""
Unit tests for ToolExecutor with controlled MockTool.

Covers: tool resolution, approval flows, validation, cancellation,
error mapping, parallel execution, and result/event ordering.

Tests use a MockTool (not FunctionAsTool) so we have full control
over what the tool does — return any value, raise any exception,
sleep arbitrary durations.
"""

from __future__ import annotations

import asyncio
import typing as t

import pytest

from max_ai.base.tool_executor import ToolExecutor
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.core.event_type import (
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)
from max_ai.core.messages import ToolMessage
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode, CoreToolParameters
from max_ai.types.run_context import RunContext
from max_ai.termination import CancellationToken


# -------- MOCK TOOL -----------------------------------------------------------
class MockTool(CoreTool):
    """Tool that returns a pre-set value or raises a pre-set exception."""

    def __init__(
        self,
        name: str = "mock",
        description: str = "Mock tool",
        approval_mode: ToolApprovalMode = ToolApprovalMode.AUTO_APPROVED,
        timeout_seconds: float = 30,
        returns: t.Any = None,
        raises: Exception | None = None,
        sleep_seconds: float = 0,
        parameters_schema: dict[str, t.Any] | None = None,
        validation_passes: bool = True,
        validation_error: str | None = None,
    ):
        super().__init__(
            name=name,
            description=description,
            approval_mode=approval_mode,
            timeout_seconds=timeout_seconds,
        )
        self._returns = returns
        self._raises = raises
        self._sleep_seconds = sleep_seconds
        self._parameters_schema = parameters_schema or {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }
        self._validation_passes = validation_passes
        self._validation_error = validation_error
        self.execute_called = False
        self.execute_record: ToolCallRecord | None = None

    @property
    def parameters(self) -> dict[str, t.Any]:
        return self._parameters_schema

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        if self._validation_passes:
            return CoreToolParameters(is_tool_valid=True, msg_error=None)
        return CoreToolParameters(
            is_tool_valid=False,
            msg_error=self._validation_error or "validation failed",
        )

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        self.execute_called = True
        self.execute_record = tool_request

        if self._sleep_seconds > 0:
            await asyncio.sleep(self._sleep_seconds)

        if self._raises is not None:
            raise self._raises

        return ToolResult.success_result(
            tool_call_id=tool_request.id, result=self._returns
        )


# -------- HELPERS -----------------------------------------------------------
def make_record(tool_name: str = "mock", **overrides: t.Any) -> ToolCallRecord:
    base: dict[str, t.Any] = {"tool_name": tool_name, "parameters": {}}
    base.update(overrides)
    return ToolCallRecord(**base)


def make_executor(tools: list[CoreTool], **overrides: t.Any) -> ToolExecutor:
    defaults: dict[str, t.Any] = {
        "tools": tools,
        "agent_name": "test-agent",
        "max_concurrent_tools": 5,
    }
    defaults.update(overrides)
    return ToolExecutor(**defaults)


def make_ctx() -> RunContext:
    return RunContext()


async def collect(gen) -> list[t.Any]:
    return [item async for item in gen]


# -------- EMPTY INPUT -----------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_records_yields_nothing():
    executor = make_executor(tools=[])
    items = await collect(executor.execute_tool_call(make_ctx(), []))
    assert items == []


# -------- TOOL RESOLUTION -----------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_not_found_emits_failure():
    executor = make_executor(tools=[])
    record = make_record("missing_tool")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # Expect: ToolMessage(error) + ToolCallResponseEvent(failure)
    assert len(items) == 2
    assert isinstance(items[0], ToolMessage)
    assert items[0].success is False
    assert "not found" in (items[0].error or "").lower()

    assert isinstance(items[1], ToolCallResponseEvent)
    assert items[1].tool_result is not None
    assert items[1].tool_result.success is False

    # Record should not be consumed (never reached EXECUTING).
    assert not record.is_consumed


# -------- AUTO-APPROVED HAPPY PATH -----------------------------------------------------------
@pytest.mark.asyncio
async def test_auto_approved_success_yields_event_message_response():
    tool = MockTool(returns="42")
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # Expected order: ToolCallEvent → ToolMessage(success) → ToolCallResponseEvent
    assert len(items) == 3
    assert isinstance(items[0], ToolCallEvent)
    assert items[0].tool_name == "mock"
    assert items[0].tool_call_id == record.id

    assert isinstance(items[1], ToolMessage)
    assert items[1].success is True
    assert items[1].content == "42"

    assert isinstance(items[2], ToolCallResponseEvent)
    assert items[2].tool_result is not None
    assert items[2].tool_result.success is True

    # Record state.
    assert record.is_consumed
    assert record.was_auto_approved
    assert record.result is not None
    assert record.result.result == "42"
    assert tool.execute_called


@pytest.mark.asyncio
async def test_auto_approved_passes_record_to_tool():
    tool = MockTool(returns="ok")
    executor = make_executor(tools=[tool])
    record = make_record("mock", parameters={"x": 1})
    ctx = make_ctx()
    ctx.tool_state.add(record)

    await collect(executor.execute_tool_call(ctx, [record]))

    assert tool.execute_record is record
    assert tool.execute_record.parameters == {"x": 1}


# -------- ASK_APPROVED FLOW -----------------------------------------------------------
@pytest.mark.asyncio
async def test_ask_approved_pending_emits_approval_event_and_pauses():
    tool = MockTool(approval_mode=ToolApprovalMode.ASK_APPROVED)
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # Expected: ToolCallEvent + ToolApprovalEvent (then pause).
    assert len(items) == 2
    assert isinstance(items[0], ToolCallEvent)
    assert isinstance(items[1], ToolApprovalEvent)
    assert items[1].tool_name == "mock"
    assert items[1].tool_call_id == record.id

    # Tool was NOT executed.
    assert not tool.execute_called
    assert record.is_pending_approval


@pytest.mark.asyncio
async def test_ask_approved_already_approved_proceeds():
    tool = MockTool(approval_mode=ToolApprovalMode.ASK_APPROVED, returns="hi")
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    record.approve(reason="user said yes")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    assert len(items) == 3  # Event + Message + Response
    assert tool.execute_called
    assert record.is_consumed


@pytest.mark.asyncio
async def test_ask_approved_rejected_yields_failure_without_running():
    tool = MockTool(approval_mode=ToolApprovalMode.ASK_APPROVED)
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    record.reject(reason="user said no")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # ToolCallEvent + ToolMessage(error) + ToolCallResponseEvent(failure)
    assert len(items) == 3
    assert isinstance(items[0], ToolCallEvent)

    assert isinstance(items[1], ToolMessage)
    assert items[1].success is False
    assert "user said no" in (items[1].error or "")

    assert isinstance(items[2], ToolCallResponseEvent)
    assert items[2].tool_result is not None
    assert items[2].tool_result.success is False

    assert not tool.execute_called
    assert record.is_rejected
    assert not record.is_consumed


# -------- VALIDATION -----------------------------------------------------------
@pytest.mark.asyncio
async def test_validation_failure_yields_failure_without_running():
    tool = MockTool(
        validation_passes=False,
        validation_error="missing required field 'q'",
    )
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # ToolCallEvent + ToolMessage(error) + ToolCallResponseEvent(failure)
    assert len(items) == 3
    assert isinstance(items[1], ToolMessage)
    assert items[1].success is False
    assert "missing required field 'q'" in (items[1].error or "")

    assert not tool.execute_called
    assert not record.is_consumed


# -------- TOOL EXCEPTIONS -----------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_raises_exception_becomes_tool_failure():
    tool = MockTool(raises=RuntimeError("boom"))
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # ToolCallEvent + ToolMessage(error) + ToolCallResponseEvent(failure)
    assert len(items) == 3
    assert isinstance(items[1], ToolMessage)
    assert items[1].success is False
    assert "boom" in (items[1].error or "")

    assert isinstance(items[2], ToolCallResponseEvent)
    assert items[2].tool_result is not None
    assert items[2].tool_result.success is False

    # Record is consumed (with failure result), since EXECUTING was reached.
    assert record.is_consumed
    assert record.result is not None
    assert record.result.success is False


# -------- TIMEOUT -----------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_timeout_becomes_tool_failure():
    tool = MockTool(sleep_seconds=2, timeout_seconds=0.1)
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    response_events = [i for i in items if isinstance(i, ToolCallResponseEvent)]
    assert len(response_events) == 1
    assert response_events[0].tool_result is not None
    assert response_events[0].tool_result.success is False
    assert "timeout" in (response_events[0].tool_result.error or "").lower()

    assert record.is_consumed


# -------- CANCELLATION -----------------------------------------------------------
@pytest.mark.asyncio
async def test_cancellation_before_execution():
    tool = MockTool(returns="ok")
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    cancel = CancellationToken()
    cancel.cancel()  # canceled before we even start

    items = await collect(executor.execute_tool_call(ctx, [record], cancel))

    response_events = [i for i in items if isinstance(i, ToolCallResponseEvent)]
    assert len(response_events) == 1
    assert response_events[0].tool_result is not None
    assert response_events[0].tool_result.success is False

    # Tool should not have been called.
    assert not tool.execute_called
    assert record.is_consumed  # consumed with cancelled_before_start


# -------- STALE EXECUTION GUARD -----------------------------------------------------------
@pytest.mark.asyncio
async def test_stale_executing_record_is_refused():
    """A record that arrives in EXECUTING (from a crashed previous run)
    must NOT be re-executed. The executor refuses and emits a failure.
    """
    tool = MockTool()
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    record.auto_approve()
    record.start_execution()  # leave it in EXECUTING (simulates crash)
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # ToolMessage(error) + ToolCallResponseEvent(failure). NO ToolCallEvent
    # because we refuse before that.
    assert any(isinstance(i, ToolMessage) and not i.success for i in items)
    assert not tool.execute_called


# -------- ALREADY-CONSUMED RECORDS -----------------------------------------------------------
@pytest.mark.asyncio
async def test_consumed_record_is_skipped_silently():
    tool = MockTool()
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    record.auto_approve().start_execution()
    record.mark_consumed(ToolResult.success_result(record.id, "done"))
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # Defensive: the executor skips consumed records silently.
    assert items == []
    assert not tool.execute_called


# -------- PARALLEL EXECUTION -----------------------------------------------------------
@pytest.mark.asyncio
async def test_parallel_two_records_both_run():
    tool_a = MockTool(name="a", returns="A")
    tool_b = MockTool(name="b", returns="B")
    executor = make_executor(tools=[tool_a, tool_b])

    rec_a = make_record("a")
    rec_b = make_record("b")
    ctx = make_ctx()
    ctx.tool_state.add(rec_a)
    ctx.tool_state.add(rec_b)

    items = await collect(executor.execute_tool_call(ctx, [rec_a, rec_b]))

    # Each gets: Event + Message + ResponseEvent → 6 total.
    assert len(items) == 6
    assert tool_a.execute_called
    assert tool_b.execute_called
    assert rec_a.is_consumed
    assert rec_b.is_consumed


@pytest.mark.asyncio
async def test_parallel_one_succeeds_one_fails():
    tool_ok = MockTool(name="ok", returns="good")
    tool_bad = MockTool(name="bad", raises=RuntimeError("nope"))
    executor = make_executor(tools=[tool_ok, tool_bad])

    rec_ok = make_record("ok")
    rec_bad = make_record("bad")
    ctx = make_ctx()
    ctx.tool_state.add(rec_ok)
    ctx.tool_state.add(rec_bad)

    items = await collect(executor.execute_tool_call(ctx, [rec_ok, rec_bad]))

    # Both records should be consumed.
    assert rec_ok.is_consumed
    assert rec_bad.is_consumed
    assert rec_ok.result is not None and rec_ok.result.success is True
    assert rec_bad.result is not None and rec_bad.result.success is False

    # Each emitted Event + Message + ResponseEvent.
    assert len(items) == 6


@pytest.mark.asyncio
async def test_parallel_respects_max_concurrent():
    """With max_concurrent_tools=1, two records run sequentially."""
    tool = MockTool(returns="x", sleep_seconds=0.05)
    executor = make_executor(tools=[tool], max_concurrent_tools=1)

    rec1 = make_record("mock")
    rec2 = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(rec1)
    ctx.tool_state.add(rec2)

    items = await collect(executor.execute_tool_call(ctx, [rec1, rec2]))

    # Both should still finish, just one at a time.
    assert len(items) == 6
    assert rec1.is_consumed
    assert rec2.is_consumed


# -------- RECORD STATE TRANSITIONS -----------------------------------------------------------
@pytest.mark.asyncio
async def test_auto_approved_sets_was_auto_approved_flag():
    tool = MockTool()
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    await collect(executor.execute_tool_call(ctx, [record]))

    assert record.was_auto_approved is True
    assert record.is_consumed


@pytest.mark.asyncio
async def test_manually_approved_does_not_set_auto_flag():
    tool = MockTool(approval_mode=ToolApprovalMode.ASK_APPROVED)
    executor = make_executor(tools=[tool])
    record = make_record("mock").approve(reason="ok")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    await collect(executor.execute_tool_call(ctx, [record]))

    assert record.was_auto_approved is False
    assert record.is_consumed


@pytest.mark.asyncio
async def test_started_at_is_set_after_execution():
    tool = MockTool()
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    assert record.started_at is None
    await collect(executor.execute_tool_call(ctx, [record]))
    assert record.started_at is not None


@pytest.mark.asyncio
async def test_consumed_at_is_set_after_execution():
    tool = MockTool()
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    await collect(executor.execute_tool_call(ctx, [record]))
    assert record.consumed_at is not None


# -------- EVENT ORDERING -----------------------------------------------------------
@pytest.mark.asyncio
async def test_events_in_order_for_successful_call():
    tool = MockTool(returns="x")
    executor = make_executor(tools=[tool])
    record = make_record("mock")
    ctx = make_ctx()
    ctx.tool_state.add(record)

    items = await collect(executor.execute_tool_call(ctx, [record]))

    # Order: ToolCallEvent → ToolMessage → ToolCallResponseEvent
    assert isinstance(items[0], ToolCallEvent)
    assert isinstance(items[1], ToolMessage)
    assert isinstance(items[2], ToolCallResponseEvent)