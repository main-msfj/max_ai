"""Tests for ReActLoopSelfDirected + UpdatePlanTool (planning-as-tool)."""

from __future__ import annotations

import pytest

from max_ai.reasoning.react_self_directed import (
    ReActLoopSelfDirected,
    ReActLoopState,
)
from max_ai.tools.update_plan import UpdatePlanTool
from max_ai.reasoning.plan import AgentPlan
from max_ai.base.reasoning import BaseLoopState
from max_ai.core.messages import AssistantMessage, ToolMessage, ToolCall
from max_ai.core.event_type import PlanningEvent
from max_ai.core.models import ModelConfig
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.base.clients import CoreChatCompletionClient


# -------- FAKES ---------------------------------------------------------------
class FakeChatClient(CoreChatCompletionClient):
    def __init__(self, results):
        self.model = "fake"
        self.config = ModelConfig()
        self._results = list(results)

    async def run(self, ctx, prompts, tools=None, output_format=None, stream=False, **kw):
        return self._results.pop(0)

    def format_messages(self, ctx, prompts): return []
    def build_api_messages(self, messages): return []
    def build_tool_schema(self, tools): return []
    def normalize_usage_stats(self, usage): return Usage()
    async def complete(self, *a, **kw): raise NotImplementedError
    async def stream(self, *a, **kw): raise NotImplementedError


class FakeToolExecutor:
    """Runs the real UpdatePlanTool; any other tool returns a generic ok."""

    def __init__(self, tool: UpdatePlanTool | None = None):
        self._tool = tool
        self.tools: dict = {}

    async def execute_tool_call(self, ctx, records, cancellation_token=None):
        for record in records:
            if self._tool and record.tool_name == UpdatePlanTool.TOOL_NAME:
                result = await self._tool.execute(record)
                yield ToolMessage(
                    source="tool", tool_call_id=record.id, tool_name=record.tool_name,
                    success=result.success, content=result.result or "",
                )
            else:
                yield ToolMessage(
                    source="tool", tool_call_id=record.id, tool_name=record.tool_name,
                    success=True, content="ok",
                )


class FakeMiddlewareChain:
    async def execute(self, action, ctx, data, func, metadata=None):
        yield await func(ctx)

    async def execute_stream(self, action, ctx, data, stream_func, metadata=None):
        async for chunk in stream_func(ctx):
            yield chunk


# -------- HELPERS -------------------------------------------------------------
def make_result(content="done", tool_calls=None):
    return ChatCompletionResult(
        message=AssistantMessage(source="fake", content=content, tool_calls=tool_calls or []),
        usage=Usage(llm_calls=1, attempts_to_call_api=1),
        model="fake",
        finish_reason="stop" if not tool_calls else "tool_calls",
    )


def make_update_plan_call(steps, rationale="because", call_id="call_1"):
    """LLM response that calls update_plan with the given steps.

    steps: list of (id, description, status) tuples.
    """
    return make_result(
        content="",
        tool_calls=[ToolCall(
            id=call_id,
            tool_name=UpdatePlanTool.TOOL_NAME,
            parameters={
                "steps": [
                    {"id": i, "description": d, "status": s} for (i, d, s) in steps
                ],
                "rationale": rationale,
            },
        )],
    )


def make_loop(client, tool=None):
    loop = ReActLoopSelfDirected(max_loop_iterations=5)
    loop.bind(
        name="test_agent",
        client=client,
        tool_executor=FakeToolExecutor(tool=tool),
        middleware_chain=FakeMiddlewareChain(),
    )
    return loop


@pytest.fixture
def ctx():
    return RunContext()


@pytest.fixture
def prompts():
    return PromptCtx.model_construct(stack=None, variables={}, rendered_layers={})


async def collect(gen):
    return [ev async for ev in gen]


# -------- UpdatePlanTool unit tests -------------------------------------------
@pytest.mark.asyncio
async def test_update_plan_tool_writes_plan_to_loop_state():
    """execute() drafts an AgentPlan onto loop_state and flags plan_updated."""
    state = BaseLoopState()
    tool = UpdatePlanTool(loop_state=state)
    record = ToolCallRecord(
        id="c1",
        tool_name=UpdatePlanTool.TOOL_NAME,
        parameters={
            "steps": [
                {"id": 1, "description": "search", "status": "active"},
                {"id": 2, "description": "summarize", "status": "pending"},
            ],
            "rationale": "broad then narrow",
        },
    )

    result = await tool.execute(record)

    assert result.success is True
    assert state.plan_updated is True
    assert isinstance(state.plan_draft, AgentPlan)
    assert [s.id for s in state.plan_draft.steps] == [1, 2]
    assert state.plan_draft.steps[0].status == "active"
    assert state.plan_draft.rationale == "broad then narrow"


@pytest.mark.asyncio
async def test_update_plan_tool_is_auto_approved():
    """The plan tool runs without asking for approval."""
    from max_ai.types.tools import ToolApprovalMode
    tool = UpdatePlanTool(loop_state=BaseLoopState())
    assert tool.approval_mode == ToolApprovalMode.AUTO_APPROVED


# -------- Loop integration tests ----------------------------------------------
@pytest.mark.asyncio
async def test_loop_syncs_plan_to_ctx_and_emits_event(ctx, prompts):
    """When the model calls update_plan, the loop copies the plan to ctx.plan
    and emits a PlanningEvent carrying that same plan."""
    state = ReActLoopState()
    tool = UpdatePlanTool(loop_state=state)
    client = FakeChatClient(results=[
        make_update_plan_call([(1, "search", "active"), (2, "summarize", "pending")]),
        make_result(content="final answer"),
    ])
    loop = make_loop(client, tool=tool)

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    # Plan landed on ctx.plan with the model's steps.
    assert ctx.plan is not None
    assert [s.id for s in ctx.plan.steps] == [1, 2]
    assert ctx.plan.steps[0].status == "active"

    # The UI got a PlanningEvent carrying that plan.
    progress = [e for e in events if isinstance(e, PlanningEvent) and e.phase == "progress"]
    assert len(progress) == 1
    assert progress[0].plan is ctx.plan


@pytest.mark.asyncio
async def test_loop_does_not_plan_when_tool_unused(ctx, prompts):
    """No update_plan call → ctx.plan stays None, no PlanningEvent. Planning is
    optional in this loop: the model plans only if it chooses to."""
    state = ReActLoopState()
    tool = UpdatePlanTool(loop_state=state)
    client = FakeChatClient(results=[make_result(content="just answering")])
    loop = make_loop(client, tool=tool)

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    assert ctx.plan is None
    assert not any(isinstance(e, PlanningEvent) for e in events)


@pytest.mark.asyncio
async def test_loop_reflects_plan_revision(ctx, prompts):
    """A second update_plan call revises ctx.plan and emits a second event —
    the model manages the plan over time."""
    state = ReActLoopState()
    tool = UpdatePlanTool(loop_state=state)
    client = FakeChatClient(results=[
        make_update_plan_call([(1, "search", "active")], call_id="c1"),
        make_update_plan_call(
            [(1, "search", "done"), (2, "write", "active")], call_id="c2"
        ),
        make_result(content="done"),
    ])
    loop = make_loop(client, tool=tool)

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    # Final plan reflects the revision.
    assert [s.id for s in ctx.plan.steps] == [1, 2]
    assert ctx.plan.steps[0].status == "done"
    assert ctx.plan.steps[1].status == "active"

    # Two progress events: one per update_plan call.
    progress = [e for e in events if isinstance(e, PlanningEvent) and e.phase == "progress"]
    assert len(progress) == 2
