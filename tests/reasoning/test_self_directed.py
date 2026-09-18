"""Tests for ReActLoopSelfDirected + UpdatePlanTool (planning-as-tool)."""

from __future__ import annotations

import pytest

from max_ai.reasoning.react_self_directed import (
    ReActLoopSelfDirected,
    ReActLoopState,
)
from max_ai.capabilities.tools.plan import AgentPlan, UpdatePlanTool
from max_ai.base.reasoning import BaseLoopState
from max_ai.core.messages import AssistantMessage, ToolMessage, ToolCall, SystemMessage
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
        # Messages the model actually saw on each call — includes transient
        # steering that never enters the durable ctx.messages.
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


def make_loop(client, tool=None, guards=None):
    # guards=[] by default: these tests exercise plan SYNC mechanics with
    # scripted models that may stop mid-plan; PlanCompletionGuard would
    # veto those finals and demand extra scripted results. The guard has
    # its own suite in test_guards.py.
    loop = ReActLoopSelfDirected(
        max_loop_iterations=5,
        guards=guards if guards is not None else [],
    )
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


# -------- STRUCTURAL PLAN NUDGE -----------------------------------------------
def make_generic_tool_call(tool_name="send_email", call_id="call_x"):
    """LLM response that calls some non-plan tool."""
    return make_result(
        content="",
        tool_calls=[ToolCall(id=call_id, tool_name=tool_name, parameters={})],
    )


def _nudges_in(messages):
    return [
        m for m in messages
        if isinstance(m, SystemMessage) and m.source == "plan-progress"
    ]


@pytest.mark.asyncio
async def test_plan_nudge_fires_when_model_skips_update(ctx, prompts):
    """The model makes a plan, then runs a tool WITHOUT updating the plan while
    steps remain -> the NEXT LLM call carries a transient 'plan-progress'
    reminder, and the durable transcript stays clean."""
    state = ReActLoopState()
    tool = UpdatePlanTool(loop_state=state)
    client = FakeChatClient(results=[
        make_update_plan_call(
            [(1, "search", "active"), (2, "email", "pending")], call_id="c1"
        ),
        make_generic_tool_call("send_email", call_id="c2"),  # progress, no update
        make_result(content="all done"),
    ])
    loop = make_loop(client, tool=tool)

    await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    # Call 1: plan created. Call 2: no nudge yet. Call 3: the round after the
    # non-updating tool run — the model sees the reminder.
    assert len(client.seen_messages) == 3
    assert _nudges_in(client.seen_messages[0]) == []
    assert _nudges_in(client.seen_messages[1]) == []
    nudges = _nudges_in(client.seen_messages[2])
    assert len(nudges) == 1
    assert "update_plan" in nudges[0].content

    # Transient: the nudge never lands in the durable transcript.
    assert _nudges_in(ctx.messages) == []


@pytest.mark.asyncio
async def test_plan_nudge_silent_when_plan_complete(ctx, prompts):
    """No nudge once every step is done — the reminder is not spammed."""
    state = ReActLoopState()
    tool = UpdatePlanTool(loop_state=state)
    client = FakeChatClient(results=[
        make_update_plan_call([(1, "search", "active")], call_id="c1"),
        make_update_plan_call([(1, "search", "done")], call_id="c2"),  # all done
        make_generic_tool_call("send_email", call_id="c3"),  # runs after completion
        make_result(content="done"),
    ])
    loop = make_loop(client, tool=tool)

    await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    # No call ever saw a nudge, and the transcript has none either.
    for seen in client.seen_messages:
        assert _nudges_in(seen) == []
    assert _nudges_in(ctx.messages) == []


# -------- ONE-SHOT FINISH -------------------------------------------------------
@pytest.mark.asyncio
async def test_final_answer_with_closing_plan_update_ends_turn(ctx, prompts):
    """The model marks the last step done AND delivers its answer in the
    same message: the loop accepts that text as the final answer instead
    of forcing another LLM round to repeat it (where weak models stall)."""
    state = ReActLoopState()
    tool = UpdatePlanTool(loop_state=state)
    closing = make_update_plan_call(
        [(1, "search", "done"), (2, "write", "done")], call_id="c_close"
    )
    closing.message = closing.message.model_copy(
        update={"content": "Full summary of steps 1 and 2."}
    )
    client = FakeChatClient(results=[
        make_update_plan_call(
            [(1, "search", "active"), (2, "write", "pending")], call_id="c_open"
        ),
        closing,
    ])
    loop = make_loop(client, tool=tool)

    await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state)
    )

    # Exactly two LLM calls — no third round to re-state the answer.
    assert state.finish_reason == "stop"
    assert len(client.seen_messages) == 2
    assert not ctx.plan.has_unfinished_steps()
    # The closing message's text is the last assistant text in the transcript.
    assistant_texts = [
        m.text() for m in ctx.messages
        if isinstance(m, AssistantMessage) and m.text().strip()
    ]
    assert assistant_texts[-1] == "Full summary of steps 1 and 2."


@pytest.mark.asyncio
async def test_bare_closing_plan_update_promotes_vetoed_answer(ctx, prompts):
    """Split finish: the model writes the final answer (vetoed by
    PlanCompletionGuard because the plan is unfinished), then closes the
    plan with a bare update_plan. The loop promotes the vetoed answer to
    final instead of forcing a 4th LLM round that would visibly repeat it."""
    from max_ai.reasoning.guards import PlanCompletionGuard

    state = ReActLoopState()
    tool = UpdatePlanTool(loop_state=state)
    client = FakeChatClient(results=[
        make_update_plan_call(
            [(1, "search", "active"), (2, "write", "pending")], call_id="c_open"
        ),
        make_result(content="Complete consolidated answer."),  # vetoed
        make_update_plan_call(
            [(1, "search", "done"), (2, "write", "done")], call_id="c_close"
        ),  # bare close-out, no text
    ])
    loop = make_loop(client, tool=tool, guards=[PlanCompletionGuard()])

    await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state)
    )

    # Three LLM calls — the loop did NOT go back for a repeat of the answer.
    assert state.finish_reason == "stop"
    assert len(client.seen_messages) == 3
    assert not ctx.plan.has_unfinished_steps()
    # The vetoed answer was promoted: no longer interim, so UIs and
    # AgentResponse.final_message surface it as the final answer.
    answer_msgs = [
        m for m in ctx.messages
        if isinstance(m, AssistantMessage)
        and m.text() == "Complete consolidated answer."
    ]
    assert len(answer_msgs) == 1
    assert answer_msgs[0].interim is False


@pytest.mark.asyncio
async def test_closing_plan_update_without_text_keeps_looping(ctx, prompts):
    """Marking the plan done with NO answer text must not end the turn —
    the model still owes the user a final answer."""
    state = ReActLoopState()
    tool = UpdatePlanTool(loop_state=state)
    client = FakeChatClient(results=[
        make_update_plan_call([(1, "search", "done")], call_id="c1"),  # no text
        make_result(content="here is the real answer"),
    ])
    loop = make_loop(client, tool=tool)

    await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state)
    )

    assert state.finish_reason == "stop"
    assert len(client.seen_messages) == 2
    assert ctx.messages[-1].text() == "here is the real answer"


@pytest.mark.asyncio
async def test_update_plan_noop_does_not_reflag():
    """Re-sending an identical plan is accepted but does not re-trigger a
    PlanningEvent (plan_updated stays False) and tells the model to move on."""
    state = BaseLoopState()
    tool = UpdatePlanTool(loop_state=state)
    params = {
        "steps": [
            {"id": 1, "description": "search", "status": "active"},
            {"id": 2, "description": "write", "status": "pending"},
        ],
        "rationale": "first",
    }

    first = await tool.execute(ToolCallRecord(
        id="c1", tool_name=UpdatePlanTool.TOOL_NAME, parameters=params,
    ))
    assert first.success and state.plan_updated is True
    state.plan_updated = False  # the loop's _sync_plan consumes the flag

    # Identical steps (rationale may differ — it is not user-visible).
    second = await tool.execute(ToolCallRecord(
        id="c2", tool_name=UpdatePlanTool.TOOL_NAME,
        parameters={**params, "rationale": "rephrased"},
    ))
    assert second.success is True
    assert state.plan_updated is False
    assert "unchanged" in str(second.result).lower()

    # A real status change flags again.
    third = await tool.execute(ToolCallRecord(
        id="c3", tool_name=UpdatePlanTool.TOOL_NAME,
        parameters={
            "steps": [
                {"id": 1, "description": "search", "status": "done"},
                {"id": 2, "description": "write", "status": "active"},
            ],
            "rationale": "progress",
        },
    ))
    assert third.success and state.plan_updated is True


@pytest.mark.asyncio
async def test_update_plan_warns_on_wholesale_replacement():
    """Renumbering/rewriting the plan (losing which steps were done) is
    accepted but the result warns not to re-execute completed work."""
    state = BaseLoopState()
    tool = UpdatePlanTool(loop_state=state)

    await tool.execute(ToolCallRecord(
        id="c1", tool_name=UpdatePlanTool.TOOL_NAME,
        parameters={
            "steps": [
                {"id": 1, "description": "check Taipei weather", "status": "done"},
                {"id": 2, "description": "check Tokyo weather", "status": "done"},
                {"id": 3, "description": "decide destination", "status": "active"},
                {"id": 4, "description": "find lodging", "status": "pending"},
            ],
            "rationale": "original",
        },
    ))

    # Model rewrites the plan with new ids — completed history lost.
    replaced = await tool.execute(ToolCallRecord(
        id="c2", tool_name=UpdatePlanTool.TOOL_NAME,
        parameters={
            "steps": [
                {"id": 10, "description": "recommend destination", "status": "active"},
                {"id": 11, "description": "suggest lodging", "status": "pending"},
            ],
            "rationale": "rewritten",
        },
    ))

    assert replaced.success is True
    assert state.plan_updated is True  # rewrite is accepted, plan syncs
    text = str(replaced.result)
    assert "WARNING" in text
    assert "check Taipei weather" in text  # done work is named explicitly
    assert "check Tokyo weather" in text


@pytest.mark.asyncio
async def test_update_plan_status_change_has_no_replacement_warning():
    """A normal status update (same ids) never triggers the warning."""
    state = BaseLoopState()
    tool = UpdatePlanTool(loop_state=state)
    base = [
        {"id": 1, "description": "a", "status": "active"},
        {"id": 2, "description": "b", "status": "pending"},
    ]
    await tool.execute(ToolCallRecord(
        id="c1", tool_name=UpdatePlanTool.TOOL_NAME,
        parameters={"steps": base, "rationale": "r"},
    ))
    progressed = await tool.execute(ToolCallRecord(
        id="c2", tool_name=UpdatePlanTool.TOOL_NAME,
        parameters={
            "steps": [
                {"id": 1, "description": "a", "status": "done"},
                {"id": 2, "description": "b", "status": "active"},
            ],
            "rationale": "r",
        },
    ))
    assert "WARNING" not in str(progressed.result)
