"""Tests for loop guards (max_ai.reasoning.guards).

Each guard is a pure check over (ctx, loop_state, guard_ctx) — unit-tested
with synthetic state — plus integration tests proving the loop delivers
guard steering transiently (the model sees it; the transcript doesn't).
"""

from __future__ import annotations

import pytest

from max_ai.reasoning.guards import (
    GuardContext,
    SchemaRetryGuard,
    RepetitionGuard,
    BudgetGuard,
    NoProgressGuard,
)
from max_ai.reasoning.react_self_directed import (
    ReActLoopSelfDirected as ReActLoop,
    ReActLoopState,
)
from max_ai.base.reasoning import BaseLoopState
from max_ai.core.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from max_ai.core.models import ModelConfig
from max_ai.core.primitives import FailureReason
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.tool_call import ToolCallRecord, ToolResult


# -------- FAKES / HELPERS -------------------------------------------------------
class FakeChatClient:
    def __init__(self, results: list):
        self.model = "fake"
        self.config = ModelConfig()
        self._results = list(results)
        self.seen_messages: list[list] = []

    async def run(self, ctx, prompts, tools=None, output_format=None, stream=False, **kw):
        self.seen_messages.append(list(ctx.messages))
        return self._results.pop(0)

    def format_messages(self, ctx, prompts): return []
    def build_api_messages(self, messages): return []
    def build_tool_schema(self, tools): return []
    def normalize_usage_stats(self, usage): return Usage()
    async def complete(self, *a, **kw): raise NotImplementedError
    async def stream(self, *a, **kw): raise NotImplementedError


class FakeToolExecutor:
    def __init__(self):
        self.tools: dict = {}

    async def execute_tool_call(self, ctx, records, cancellation_token=None):
        for record in records:
            yield ToolMessage(
                source="tool",
                tool_call_id=record.id,
                tool_name=record.tool_name,
                success=True,
                content="same result",
            )


class FakeMiddlewareChain:
    async def execute(self, action, ctx, data, func, metadata=None):
        yield await func(ctx)

    async def execute_stream(self, action, ctx, data, stream_func, metadata=None):
        async for chunk in stream_func(ctx):
            yield chunk


class SchemaTool:
    """Minimal object exposing .parameters for GuardContext.tools."""

    def __init__(self, name="ping"):
        self.name = name

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }


def make_result(content="done", tool_calls=None, finish_reason=None):
    return ChatCompletionResult(
        message=AssistantMessage(
            source="fake", content=content, tool_calls=tool_calls or []
        ),
        usage=Usage(llm_calls=1, attempts_to_call_api=1),
        model="fake",
        finish_reason=finish_reason
        or ("stop" if not tool_calls else "tool_calls"),
    )


def make_loop(client, guards=None, max_iter=6):
    loop = ReActLoop(
        max_loop_iterations=max_iter,
        guards=guards,
        enable_human_input=False,
    )
    loop.bind(
        name="test_agent",
        client=client,
        tool_executor=FakeToolExecutor(),
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


def guard_msgs(messages):
    return [
        m for m in messages
        if isinstance(m, SystemMessage) and m.source == "loop-guard"
    ]


# -------- RepetitionGuard ------------------------------------------------------
def _round_with_call(ctx: RunContext, tool_name="ping", params=None, call_id="c"):
    """Append one assistant tool round to the transcript."""
    tc = ToolCall(id=call_id, tool_name=tool_name, parameters=params or {"x": 1})
    ctx.messages.append(
        AssistantMessage(source="a", content="", tool_calls=[tc])
    )
    ctx.messages.append(
        ToolMessage(
            source="t", tool_call_id=tc.id, tool_name=tool_name,
            success=True, content="same",
        )
    )


def test_repetition_guard_fires_on_third_identical_call(ctx):
    guard = RepetitionGuard(max_repeats=3)
    state = BaseLoopState()
    gctx = GuardContext()

    _round_with_call(ctx, call_id="c1")
    assert guard.after_tool_round(ctx, state, gctx) is None
    _round_with_call(ctx, call_id="c2")
    assert guard.after_tool_round(ctx, state, gctx) is None
    _round_with_call(ctx, call_id="c3")
    steering = guard.after_tool_round(ctx, state, gctx)

    assert steering is not None
    assert "ping" in steering
    assert "Do not call these tools again" in steering


def test_repetition_guard_fires_on_second_call_by_default_with_result_echo(ctx):
    """Default fires on the 2nd identical call, and the steering echoes the
    cached result so the model has no reason to re-fetch."""
    guard = RepetitionGuard()
    state = BaseLoopState()
    gctx = GuardContext()

    def round_with_record(call_id):
        _round_with_call(ctx, call_id=call_id)
        record = ToolCallRecord(
            id=call_id, tool_name="ping", parameters={"x": 1}, session_id="s",
        )
        record.result = ToolResult.success_result(call_id, "{'temp': 18}")
        ctx.tool_state.add(record)

    round_with_record("r1")
    assert guard.after_tool_round(ctx, state, gctx) is None
    round_with_record("r2")
    steering = guard.after_tool_round(ctx, state, gctx)

    assert steering is not None
    assert "already have the results" in steering
    assert "{'temp': 18}" in steering  # cached result echoed back


def test_repetition_guard_ignores_different_arguments(ctx):
    guard = RepetitionGuard(max_repeats=2)
    state = BaseLoopState()
    gctx = GuardContext()

    _round_with_call(ctx, params={"x": 1}, call_id="c1")
    assert guard.after_tool_round(ctx, state, gctx) is None
    _round_with_call(ctx, params={"x": 2}, call_id="c2")
    assert guard.after_tool_round(ctx, state, gctx) is None


# -------- BudgetGuard ----------------------------------------------------------
def test_budget_guard_warns_once_at_threshold(ctx):
    guard = BudgetGuard(threshold=0.75)
    state = BaseLoopState()
    gctx = GuardContext(max_loop_iterations=10)

    state.iteration = 7
    assert guard.after_tool_round(ctx, state, gctx) is None

    state.iteration = 8
    steering = guard.after_tool_round(ctx, state, gctx)
    assert steering is not None
    assert "2 reasoning iteration(s) left" in steering

    # Fires once per turn.
    state.iteration = 9
    assert guard.after_tool_round(ctx, state, gctx) is None


# -------- NoProgressGuard ------------------------------------------------------
def _state_with_answer(content: str) -> BaseLoopState:
    state = BaseLoopState()
    state.last_result = make_result(content=content)
    return state


def test_no_progress_guard_vetoes_empty_answer_once(ctx):
    guard = NoProgressGuard()
    gctx = GuardContext(tools={"ping": SchemaTool()})
    state = _state_with_answer("")

    steering = guard.on_final_answer(ctx, state, gctx)
    assert steering is not None
    assert "ping" in steering  # the action menu lists the tools

    # A second empty answer passes through — never live-locks.
    state.last_result = make_result(content="")
    assert guard.on_final_answer(ctx, state, gctx) is None


def test_no_progress_guard_allows_real_answer(ctx):
    guard = NoProgressGuard()
    state = _state_with_answer("here is your answer")
    assert guard.on_final_answer(ctx, state, GuardContext()) is None


# -------- SchemaRetryGuard -----------------------------------------------------
def test_schema_retry_guard_echoes_schema_on_invalid_params(ctx):
    guard = SchemaRetryGuard()
    state = BaseLoopState()
    gctx = GuardContext(tools={"ping": SchemaTool()})

    record = ToolCallRecord(id="c1", tool_name="ping", parameters={"wrong": True})
    ctx.tool_state.add(record)
    record.force_consume(
        ToolResult.invalid_parameters("c1", "missing required 'x'")
    )
    ctx.messages.append(
        AssistantMessage(
            source="a", content="",
            tool_calls=[ToolCall(id="c1", tool_name="ping", parameters={"wrong": True})],
        )
    )
    ctx.messages.append(
        ToolMessage(
            source="t", tool_call_id="c1", tool_name="ping",
            success=False, content="Invalid parameters",
        )
    )

    steering = guard.after_tool_round(ctx, state, gctx)
    assert steering is not None
    assert "ping" in steering
    assert '"required": ["x"]' in steering
    assert "missing required 'x'" in steering


def test_schema_retry_guard_ignores_other_failures(ctx):
    guard = SchemaRetryGuard()
    gctx = GuardContext(tools={"ping": SchemaTool()})

    record = ToolCallRecord(id="c1", tool_name="ping", parameters={})
    ctx.tool_state.add(record)
    record.force_consume(
        ToolResult.tool_failure("c1", "boom", reason=FailureReason.EXECUTION_ERROR)
    )
    ctx.messages.append(
        AssistantMessage(
            source="a", content="",
            tool_calls=[ToolCall(id="c1", tool_name="ping", parameters={})],
        )
    )
    ctx.messages.append(
        ToolMessage(
            source="t", tool_call_id="c1", tool_name="ping",
            success=False, content="boom",
        )
    )

    assert guard.after_tool_round(ctx, BaseLoopState(), gctx) is None


# -------- Integration: steering is transient -------------------------------------
def _tool_round_result(call_id: str):
    return make_result(
        content="",
        tool_calls=[ToolCall(id=call_id, tool_name="ping", parameters={"x": 1})],
    )


@pytest.mark.asyncio
async def test_loop_delivers_repetition_steering_transiently(ctx, prompts):
    """Three identical calls → the 4th LLM call sees the guard message; the
    durable transcript never contains it."""
    client = FakeChatClient(results=[
        _tool_round_result("c1"),
        _tool_round_result("c2"),
        _tool_round_result("c3"),
        make_result(content="ok, answering now"),
    ])
    loop = make_loop(client, guards=[RepetitionGuard(max_repeats=3)])
    state = ReActLoopState()

    await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    assert state.finish_reason == "stop"
    # Calls 1-3 clean; call 4 carries the steering.
    assert [len(guard_msgs(seen)) for seen in client.seen_messages] == [0, 0, 0, 1]
    assert "Do not call these tools again" in guard_msgs(client.seen_messages[3])[0].content
    assert guard_msgs(ctx.messages) == []


@pytest.mark.asyncio
async def test_loop_vetoes_empty_final_answer_once(ctx, prompts):
    """An empty tool-free answer is sent back with steering; the retry's real
    answer ends the turn."""
    client = FakeChatClient(results=[
        make_result(content=""),
        make_result(content="real answer"),
    ])
    loop = make_loop(client, guards=[NoProgressGuard()])
    state = ReActLoopState()

    await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    assert state.finish_reason == "stop"
    assert len(client.seen_messages) == 2
    assert len(guard_msgs(client.seen_messages[1])) == 1
    assert ctx.messages[-1].text() == "real answer"
    assert guard_msgs(ctx.messages) == []


@pytest.mark.asyncio
async def test_loop_with_guards_disabled_finishes_on_empty_answer(ctx, prompts):
    """guards=[] restores the raw behaviour — an empty answer ends the turn."""
    client = FakeChatClient(results=[make_result(content="")])
    loop = make_loop(client, guards=[])
    state = ReActLoopState()

    await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    assert state.finish_reason == "stop"
    assert len(client.seen_messages) == 1


# -------- PlanCompletionGuard --------------------------------------------------
def _plan(statuses: list[str]):
    from max_ai.capabilities.tools.plan import AgentPlan, PlanStep

    return AgentPlan(
        steps=[
            PlanStep(id=i + 1, description=f"step {i + 1}", status=s)
            for i, s in enumerate(statuses)
        ],
        rationale="test",
    )


def test_plan_completion_guard_vetoes_while_steps_remain(ctx):
    from max_ai.reasoning.guards import PlanCompletionGuard

    guard = PlanCompletionGuard(max_nudges=2)
    gctx = GuardContext(tools={"update_plan": SchemaTool()})
    ctx.plan = _plan(["done", "active", "pending"])
    state = _state_with_answer("Next: I'll do step 2")

    steering = guard.on_final_answer(ctx, state, gctx)
    assert steering is not None
    assert "step 2" in steering and "step 3" in steering
    assert "step 1" not in steering  # finished steps are not re-listed

    # Second veto still fires, third is allowed through (cap reached).
    assert guard.on_final_answer(ctx, state, gctx) is not None
    assert guard.on_final_answer(ctx, state, gctx) is None


def test_plan_completion_guard_silent_when_plan_done(ctx):
    from max_ai.reasoning.guards import PlanCompletionGuard

    guard = PlanCompletionGuard()
    gctx = GuardContext(tools={"update_plan": SchemaTool()})
    state = _state_with_answer("all done")

    assert guard.on_final_answer(ctx, state, gctx) is None  # no plan at all
    ctx.plan = _plan(["done", "done"])
    assert guard.on_final_answer(ctx, state, gctx) is None


def test_plan_completion_guard_silent_without_update_plan_tool(ctx):
    from max_ai.reasoning.guards import PlanCompletionGuard

    guard = PlanCompletionGuard()
    ctx.plan = _plan(["active", "pending"])
    state = _state_with_answer("stopping early")

    assert guard.on_final_answer(ctx, state, GuardContext()) is None


@pytest.mark.asyncio
async def test_loop_continues_past_premature_final_answer(ctx, prompts):
    """Model narrates 'Next: ...' and stops mid-plan; the guard sends it
    back (transiently) and the turn only ends on the follow-up answer.
    max_nudges=1 so the second answer passes even though the fake model
    never updates the plan — the cap is the live-lock escape hatch."""
    from max_ai.reasoning.guards import PlanCompletionGuard

    ctx.plan = _plan(["done", "active"])
    client = FakeChatClient(
        results=[
            make_result(content="Next: I'll do step 2."),
            make_result(content="Step 2 done; here is the full answer."),
        ]
    )
    loop = make_loop(client, guards=[PlanCompletionGuard(max_nudges=1)])
    state = ReActLoopState()

    await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state)
    )

    assert state.finish_reason == "stop"
    assert len(client.seen_messages) == 2
    # The steering reached the second call transiently...
    assert guard_msgs(client.seen_messages[1])
    # ...but never entered the durable transcript.
    assert not guard_msgs(ctx.messages)
    assert ctx.messages[-1].text() == "Step 2 done; here is the full answer."
    # The vetoed draft stays in the transcript but is marked interim so
    # UIs hide it; the accepted final answer is not.
    drafts = [
        m for m in ctx.messages
        if isinstance(m, AssistantMessage) and m.text().startswith("Next:")
    ]
    assert len(drafts) == 1 and drafts[0].interim is True
    assert ctx.messages[-1].interim is False


def test_plan_completion_guard_points_at_ask_tool_when_available(ctx):
    """When the ask-the-user tool is registered, the steering offers it as
    the escape for input-blocked steps; without it, it does not."""
    from max_ai.reasoning.guards import PlanCompletionGuard

    ctx.plan = _plan(["done", "active"])
    state = _state_with_answer("I need your check-in date to continue.")

    with_ask = PlanCompletionGuard().on_final_answer(
        ctx, state, GuardContext(tools={
            "update_plan": SchemaTool(), "ask_user": SchemaTool(),
        }),
    )
    assert "ask_user" in with_ask

    state2 = _state_with_answer("I need your check-in date to continue.")
    without_ask = PlanCompletionGuard().on_final_answer(
        ctx, state2, GuardContext(tools={"update_plan": SchemaTool()}),
    )
    assert "ask_user" not in without_ask
