"""Tests for Agent.resume() and its streaming variants."""

from __future__ import annotations

import typing as t
import pytest

from max_ai.base.agent import Agent
from max_ai.base.clients import CoreChatCompletionClient
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.core.messages import AssistantMessage, ToolCall, UserMessage
from max_ai.core.event_type import CoreEvent, ToolCallEvent, ToolCallResponseEvent
from max_ai.core.models import ModelConfig
from max_ai.types.agent_response import AgentResponse
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode, CoreToolParameters
from max_ai.errors.agent import AgentError


# -------- FAKE CLIENT -----------------------------------------------------------
class FakeChatClient(CoreChatCompletionClient):
    def __init__(self, results=None):
        self.model = "fake"
        self.config = ModelConfig()
        self._results: list[ChatCompletionResult] = list(results or [])

    async def run(self, ctx, prompts, tools=None, output_format=None,
                  stream=False, **kwargs):
        return self._results.pop(0)

    def format_messages(self, ctx, prompts): return []
    def build_api_messages(self, messages): return []
    def build_tool_schema(self, tools): return []
    def normalize_usage_stats(self, usage): return Usage()
    async def complete(self, *a, **kw): raise NotImplementedError
    async def stream(self, *a, **kw):
        raise NotImplementedError
        yield


# -------- FAKE TOOL -----------------------------------------------------------
class FakeApprovalTool(CoreTool):
    """Tool requiring approval; records each execution."""

    def __init__(self, name="risky_op"):
        super().__init__(
            name=name,
            description="Fake risky tool.",
            version="1.0.0",
            approval_mode=ToolApprovalMode.ASK_APPROVED,
            timeout_seconds=10,
            max_retries=0,
        )
        self.executions: list[dict] = []

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        return CoreToolParameters(is_tool_valid=True, msg_error=None)

    async def execute(self, tool_request, tool_context=None, cancellation_token=None):
        self.executions.append(dict(tool_request.parameters))
        return ToolResult.success_result(
            tool_request.id, "operation_done", {"name": self.name}
        )


# -------- HELPERS -----------------------------------------------------------
def make_result(content="", tool_calls=None, finish_reason="stop"):
    return ChatCompletionResult(
        message=AssistantMessage(
            source="fake",
            content=content,
            tool_calls=tool_calls or [],
        ),
        usage=Usage(llm_calls=1, attempts_to_call_api=1),
        model="fake",
        finish_reason=finish_reason,
    )


def make_agent(client, tools=None, **kwargs) -> Agent:
    return Agent(
        name="test_agent",
        description="Agent for resume tests.",
        instructions="Be concise.",
        client=client,
        toolset=tools,
        **kwargs,
    )


async def pause_with_pending_approval():
    """Run an agent until it pauses on a pending tool approval. Returns ctx + tool."""
    tool = FakeApprovalTool()
    tc = ToolCall(id="call_1", tool_name="risky_op", parameters={})
    client = FakeChatClient(results=[
        make_result(tool_calls=[tc], finish_reason="tool_calls"),
        make_result(content="all done"),  # used after approval
    ])
    agent = make_agent(client, tools=[tool])

    response = await agent.run(task="do the risky thing")
    assert response.finish_reason == "approval_needed"
    assert response.needs_approval
    return agent, tool, response


# -------- HAPPY PATH -----------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_happy_path():
    """Run pauses, user approves, resume executes tool and finishes."""
    agent, tool, response = await pause_with_pending_approval()

    # User approves
    pending = response.pending_approvals
    assert len(pending) == 1
    response.context.tool_state.apply_approval(
        pending[0].id, approved=True, reason="ok"
    )

    final = await agent.resume(run_context=response.context)

    assert final.finish_reason == "stop"
    assert final.usage.tool_calls >= 1
    assert tool.executions == [{}]  # tool ran once
    assert final.final_text == "all done"


# -------- VALIDATION -----------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_with_pending_approvals_raises():
    """Resume without applying decisions → AgentError."""
    agent, tool, response = await pause_with_pending_approval()

    # Don't apply approval — leave the record pending.
    with pytest.raises(AgentError):
        await agent.resume(run_context=response.context)

    # Tool must NOT have run.
    assert tool.executions == []


@pytest.mark.asyncio
async def test_resume_with_nothing_to_do_raises():
    """Resume on a clean ctx (no actionable, no stale) → AgentError."""
    client = FakeChatClient(results=[])
    agent = make_agent(client)

    ctx = RunContext()
    ctx.messages.append(UserMessage(source="user", content="hi"))

    with pytest.raises(AgentError):
        await agent.resume(run_context=ctx)


# -------- REJECTED -----------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_with_rejected_decision():
    """Run pauses, user rejects, resume reports rejection back to LLM."""
    agent, tool, response = await pause_with_pending_approval()

    pending = response.pending_approvals[0]
    response.context.tool_state.apply_approval(
        pending.id, approved=False, reason="not safe"
    )

    final = await agent.resume(run_context=response.context)

    assert final.finish_reason == "stop"
    assert tool.executions == []  # tool MUST not have run
    # The LLM saw the rejection and produced its closing reply.
    assert final.final_text == "all done"


@pytest.mark.asyncio
async def test_resume_with_mixed_approval_decisions_only_runs_approved_tool():
    """A turn with multiple pending tools resumes only the approved calls."""
    weather_tool = FakeApprovalTool(name="get_weather")
    link_tool = FakeApprovalTool(name="check_link")
    weather_call = ToolCall(
        id="call_weather",
        tool_name="get_weather",
        parameters={"city": "Tokyo"},
    )
    link_call = ToolCall(
        id="call_link",
        tool_name="check_link",
        parameters={"url": "https://example.com"},
    )
    client = FakeChatClient(results=[
        make_result(
            tool_calls=[weather_call, link_call],
            finish_reason="tool_calls",
        ),
        make_result(content="weather checked; link skipped"),
    ])
    agent = make_agent(client, tools=[weather_tool, link_tool])

    response = await agent.run(task="check weather and link")

    assert response.finish_reason == "approval_needed"
    assert len(response.pending_approvals) == 2
    response.context.tool_state.apply_approval(
        "call_weather",
        approved=True,
        reason="ok",
    )
    response.context.tool_state.apply_approval(
        "call_link",
        approved=False,
        reason="not this link",
    )

    items = [
        item async for item in agent.resume_stream_events(run_context=response.context)
    ]

    final = items[-1]
    assert isinstance(final, AgentResponse)
    assert final.finish_reason == "stop"
    assert final.final_text == "weather checked; link skipped"
    assert weather_tool.executions == [{"city": "Tokyo"}]
    assert link_tool.executions == []

    tool_call_events = [item for item in items if isinstance(item, ToolCallEvent)]
    assert [event.tool_name for event in tool_call_events] == ["get_weather"]

    tool_results = [item for item in items if isinstance(item, ToolCallResponseEvent)]
    assert len(tool_results) == 2
    result_by_id = {event.tool_call_id: event.tool_result for event in tool_results}
    assert result_by_id["call_weather"] is not None
    assert result_by_id["call_weather"].success is True
    assert result_by_id["call_link"] is not None
    assert result_by_id["call_link"].success is False


# -------- STALE -----------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_with_stale_execution_marks_failed():
    """A record left in EXECUTING is force-failed before resume continues."""
    # Build a ctx that has a stale record AND an actionable record so the
    # resume has work to do (otherwise nothing_to_resume fires).
    ctx = RunContext()
    ctx.messages.append(UserMessage(source="user", content="do things"))
    # Synthesize an assistant message with two tool calls so the stale + new
    # record are both consistent with the transcript.
    actionable_tc = ToolCall(id="call_actionable", tool_name="risky_op", parameters={})
    stale_tc = ToolCall(id="call_stale", tool_name="risky_op", parameters={})
    ctx.messages.append(AssistantMessage(
        source="fake", content="", tool_calls=[actionable_tc, stale_tc],
    ))

    actionable = ToolCallRecord(
        id="call_actionable", tool_name="risky_op", parameters={},
    )
    actionable.auto_approve()
    ctx.tool_state.add(actionable)

    stale = ToolCallRecord(id="call_stale", tool_name="risky_op", parameters={})
    stale.auto_approve()
    stale.start_execution()  # → EXECUTING; will look stale to resume()
    ctx.tool_state.add(stale)

    tool = FakeApprovalTool()
    # After resume, executor runs the actionable, then the LLM closes the turn.
    client = FakeChatClient(results=[
        make_result(content="finished after recovery"),
    ])
    agent = make_agent(client, tools=[tool])

    final = await agent.resume(run_context=ctx)

    # Stale record was force-failed.
    stale_after = ctx.tool_state.get("call_stale")
    assert stale_after is not None
    assert stale_after.is_consumed
    assert stale_after.result is not None
    assert stale_after.result.success is False

    # Actionable record ran.
    assert tool.executions == [{}]
    assert final.finish_reason == "stop"


# -------- STREAM_EVENTS -----------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_stream_events_terminates_with_response():
    """resume_stream_events yields events, final item is AgentResponse."""
    agent, tool, response = await pause_with_pending_approval()
    response.context.tool_state.apply_approval(
        response.pending_approvals[0].id, approved=True
    )

    items: list = [
        item async for item in agent.resume_stream_events(run_context=response.context)
    ]

    assert isinstance(items[-1], AgentResponse)
    assert all(isinstance(x, CoreEvent) for x in items[:-1])
    response_count = sum(1 for x in items if isinstance(x, AgentResponse))
    assert response_count == 1
