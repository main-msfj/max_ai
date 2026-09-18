"""Tests for durable Human-in-the-Loop (elicitation as record state).

Elicitation works exactly like approvals: the executor never runs the
ask-the-user tool. A fresh call becomes ``INPUT_NEEDED`` on the record
plus a ``UserInputRequestEvent``, and the turn ends with
``finish_reason='input_needed'``. The consumer answers via
``ctx.tool_state.apply_user_answer(...)`` and resumes; the executor then
completes the call from the stored answer. A pending question serializes
with the ``RunContext`` and survives process death.
"""

from __future__ import annotations

import pytest

from max_ai.reasoning.react_self_directed import (
    ReActLoopSelfDirected as ReActLoop,
    ReActLoopState,
)
from max_ai.base.tool_executor import ToolExecutor, USER_INPUT_TOOL_NAME
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage
from max_ai.core.event_type import (
    ReasoningCompleteEvent,
    ToolApprovalEvent,
    UserInputRequestEvent,
)
from max_ai.core.models import ModelConfig
from max_ai.core.primitives import ToolCallStatus
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode, CoreToolParameters
from max_ai.capabilities.tools.ask_user import AskUserTool
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.termination import CancellationToken


# -------- FAKES ---------------------------------------------------------------
class FakeChatClient:
    def __init__(self, results: list):
        self.model = "fake"
        self.config = ModelConfig()
        self._results = list(results)

    async def run(
        self, ctx, prompts, tools=None, output_format=None, stream=False, **kw
    ):
        return self._results.pop(0)

    def format_messages(self, ctx, prompts):
        return []

    def build_api_messages(self, messages):
        return []

    def build_tool_schema(self, tools):
        return []

    def normalize_usage_stats(self, usage):
        return Usage()

    async def complete(self, *a, **kw):
        raise NotImplementedError

    async def stream(self, *a, **kw):
        raise NotImplementedError


class FakeMiddlewareChain:
    async def execute(self, action, ctx, data, func, metadata=None):
        yield await func(ctx)

    async def execute_stream(self, action, ctx, data, stream_func, metadata=None):
        async for chunk in stream_func(ctx):
            yield chunk


class EchoTool(CoreTool):
    """Simple tool for mixed-batch tests. Optionally approval-gated."""

    def __init__(self, name="echo", approval_mode=ToolApprovalMode.AUTO_APPROVED):
        super().__init__(name=name, description="Echo", approval_mode=approval_mode)
        self.execute_called = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": True}

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        return CoreToolParameters(is_tool_valid=True, msg_error=None)

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        self.execute_called = True
        return ToolResult.success_result(tool_request.id, "echoed")


# -------- HELPERS -------------------------------------------------------------
def make_result(content="done", tool_calls=None):
    return ChatCompletionResult(
        message=AssistantMessage(
            source="fake",
            content=content,
            tool_calls=tool_calls or [],
        ),
        usage=Usage(llm_calls=1, attempts_to_call_api=1),
        model="fake",
        finish_reason="stop" if not tool_calls else "tool_calls",
    )


def make_question_call(
    question="Which format?", options=None, call_id="call_q1"
):
    return make_result(
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                tool_name=USER_INPUT_TOOL_NAME,
                parameters={"question": question, "options": options}
                if options
                else {"question": question},
            )
        ],
    )


def make_loop(client, executor=None):
    loop = ReActLoop(max_loop_iterations=5)
    loop.bind(
        name="test_agent",
        client=client,
        tool_executor=executor or ToolExecutor(agent_name="test_agent"),
        middleware_chain=FakeMiddlewareChain(),
    )
    return loop


def make_question_record(
    record_id="c1", question="Which format?", options=None
) -> ToolCallRecord:
    return ToolCallRecord(
        id=record_id,
        tool_name=USER_INPUT_TOOL_NAME,
        parameters={"question": question, "options": options},
        session_id="s1",
    )


@pytest.fixture
def ctx():
    return RunContext()


@pytest.fixture
def prompts():
    return PromptCtx.model_construct(stack=None, variables={}, rendered_layers={})


async def collect(gen) -> list:
    return [ev async for ev in gen]


# -------- Contract glue ---------------------------------------------------------
def test_executor_constant_matches_tool_name():
    """The executor's short-circuit key must stay in sync with the tool."""
    assert USER_INPUT_TOOL_NAME == AskUserTool.TOOL_NAME


def test_user_input_tool_parameters_schema():
    """Tool exposes correct JSON schema with question required, options optional."""
    tool = AskUserTool()
    schema = tool.parameters
    assert schema["type"] == "object"
    assert "question" in schema["properties"]
    assert "options" in schema["properties"]
    assert "question" in schema["required"]
    assert "options" not in schema.get("required", [])


# -------- Record-state unit tests ----------------------------------------------
def test_await_user_input_transition():
    """PENDING_APPROVAL → INPUT_NEEDED carrying question/options."""
    record = make_question_record()
    record.await_user_input("Which format?", ["JSON", "CSV"])

    assert record.status == ToolCallStatus.INPUT_NEEDED
    assert record.is_awaiting_input
    assert record.input_question == "Which format?"
    assert record.input_options == ["JSON", "CSV"]
    assert not record.is_actionable


def test_apply_user_answer_transition():
    """INPUT_NEEDED → APPROVED with the answer stored (actionable again)."""
    record = make_question_record()
    record.await_user_input("Which format?")
    record.apply_user_answer("JSON")

    assert record.status == ToolCallStatus.APPROVED
    assert record.user_answer == "JSON"
    assert record.answered_at is not None
    assert record.is_actionable


def test_apply_user_answer_requires_input_needed():
    record = make_question_record()
    with pytest.raises(ValueError, match="must be INPUT_NEEDED"):
        record.apply_user_answer("JSON")


def test_await_user_input_requires_pending_approval():
    record = make_question_record()
    record.await_user_input("q")
    with pytest.raises(ValueError, match="cannot await user input"):
        record.await_user_input("q again")


def test_tool_state_pending_and_apply():
    """ToolState surfaces INPUT_NEEDED records and applies answers."""
    ctx = RunContext()
    record = make_question_record()
    ctx.tool_state.add(record)
    record.await_user_input("Which tone?", ["formal", "casual"])

    assert ctx.tool_state.waiting_for_input is True
    assert ctx.tool_state.pending_user_input == [record]

    ctx.tool_state.apply_user_answer(record.id, "formal")
    assert ctx.tool_state.waiting_for_input is False
    assert ctx.tool_state.pending_user_input == []
    assert record.user_answer == "formal"


def test_tool_state_apply_answer_unknown_id():
    ctx = RunContext()
    with pytest.raises(KeyError):
        ctx.tool_state.apply_user_answer("nope", "answer")


# -------- Executor short-circuit -------------------------------------------------
@pytest.mark.asyncio
async def test_executor_pauses_fresh_question(ctx):
    """Fresh record → INPUT_NEEDED + UserInputRequestEvent, tool never runs."""
    executor = ToolExecutor(
        tools=[AskUserTool()], agent_name="t"
    )
    record = make_question_record(question="Which format?", options=["JSON", "CSV"])
    ctx.tool_state.add(record)

    items = [i async for i in executor.execute_tool_call(ctx, [record])]

    input_events = [i for i in items if isinstance(i, UserInputRequestEvent)]
    assert len(input_events) == 1
    assert input_events[0].question == "Which format?"
    assert input_events[0].options == ["JSON", "CSV"]
    assert input_events[0].tool_call_id == record.id
    # No ToolMessage — nothing executed.
    assert not any(isinstance(i, ToolMessage) for i in items)
    assert record.status == ToolCallStatus.INPUT_NEEDED


@pytest.mark.asyncio
async def test_executor_completes_answered_question(ctx):
    """Answered record → normal ToolMessage carrying the answer, CONSUMED."""
    executor = ToolExecutor(tools=[AskUserTool()], agent_name="t")
    record = make_question_record()
    ctx.tool_state.add(record)
    record.await_user_input("Which format?")
    record.apply_user_answer("JSON")

    items = [i async for i in executor.execute_tool_call(ctx, [record])]

    tool_msgs = [i for i in items if isinstance(i, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].success is True
    # The answer is framed with the question so the model gets a
    # self-describing result, not a bare string.
    assert "'JSON'" in tool_msgs[0].text()
    assert "Which format?" in tool_msgs[0].text()
    assert record.status == ToolCallStatus.CONSUMED
    # The raw answer stays untouched on the durable record.
    assert record.user_answer == "JSON"
    assert record.result is not None and "'JSON'" in str(record.result.result)


@pytest.mark.asyncio
async def test_executor_frames_free_form_answer_as_valid(ctx):
    """An answer that is not one of the offered options is explicitly
    labeled a valid free-form reply — small models otherwise misread it
    as invalid and claim it 'blocks' them."""
    executor = ToolExecutor(tools=[AskUserTool()], agent_name="t")
    record = make_question_record()
    ctx.tool_state.add(record)
    record.await_user_input(
        "Which destination?", options=["Tokyo — warmer", "New York — sunny"]
    )
    record.apply_user_answer("you should have deduced that already")

    items = [i async for i in executor.execute_tool_call(ctx, [record])]

    tool_msgs = [i for i in items if isinstance(i, ToolMessage)]
    assert len(tool_msgs) == 1
    text = tool_msgs[0].text()
    assert "you should have deduced that already" in text
    assert "free-form reply IS the answer" in text
    assert "do not say it blocks you" in text


@pytest.mark.asyncio
async def test_executor_frames_skipped_question(ctx):
    """An empty answer (Skip) tells the model to proceed, not to re-ask."""
    executor = ToolExecutor(tools=[AskUserTool()], agent_name="t")
    record = make_question_record()
    ctx.tool_state.add(record)
    record.await_user_input("Which format?")
    record.apply_user_answer("")

    items = [i async for i in executor.execute_tool_call(ctx, [record])]

    tool_msgs = [i for i in items if isinstance(i, ToolMessage)]
    assert len(tool_msgs) == 1
    text = tool_msgs[0].text()
    assert "skipped this question" in text
    assert "Do not re-ask" in text


@pytest.mark.asyncio
async def test_executor_frames_option_label_answer(ctx):
    """Picking an option label frames it as the chosen answer with the
    question restated — no free-form disclaimer."""
    executor = ToolExecutor(tools=[AskUserTool()], agent_name="t")
    record = make_question_record()
    ctx.tool_state.add(record)
    record.await_user_input(
        "Which destination?", options=["Tokyo — warmer", "New York — sunny"]
    )
    record.apply_user_answer("Tokyo")  # UI submits the parsed label

    items = [i async for i in executor.execute_tool_call(ctx, [record])]

    tool_msgs = [i for i in items if isinstance(i, ToolMessage)]
    text = tool_msgs[0].text()
    assert "'Tokyo'" in text
    assert "Which destination?" in text
    assert "free-form" not in text


@pytest.mark.asyncio
async def test_executor_reemits_event_for_unanswered_record(ctx):
    """An INPUT_NEEDED record dispatched again without an answer re-pauses."""
    executor = ToolExecutor(tools=[AskUserTool()], agent_name="t")
    record = make_question_record()
    ctx.tool_state.add(record)
    record.await_user_input("Still there?")

    items = [i async for i in executor.execute_tool_call(ctx, [record])]

    assert any(isinstance(i, UserInputRequestEvent) for i in items)
    assert record.status == ToolCallStatus.INPUT_NEEDED


# -------- Loop integration: pause → answer → resume ------------------------------
@pytest.mark.asyncio
async def test_loop_pauses_on_question_and_resumes_with_answer(ctx, prompts):
    """Full cycle: model asks → turn ends input_needed → answer applied →
    second segment folds the answer in as a ToolMessage and finishes."""
    client = FakeChatClient(
        results=[
            make_question_call(question="Which format?", options=["JSON", "CSV"]),
            make_result(content="Using JSON, here you go"),
        ]
    )
    executor = ToolExecutor(agent_name="test_agent")  # loop auto-registers the tool
    loop = make_loop(client, executor)
    state = ReActLoopState()

    # Segment 1: the model asks; the turn pauses.
    events = await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state)
    )

    assert state.finish_reason == "input_needed"
    input_events = [e for e in events if isinstance(e, UserInputRequestEvent)]
    assert len(input_events) == 1
    assert input_events[0].question == "Which format?"
    complete = [e for e in events if isinstance(e, ReasoningCompleteEvent)]
    assert complete[-1].finish_reason == "input_needed"

    record_id = input_events[0].tool_call_id
    assert ctx.tool_state.waiting_for_input is True

    # The consumer answers on the durable state and resumes.
    ctx.tool_state.apply_user_answer(record_id, "JSON")

    state2 = ReActLoopState()
    events2 = await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state2)
    )

    assert state2.finish_reason == "stop"
    # The answer entered the transcript as the tool's result.
    tool_msgs = [m for m in ctx.messages if isinstance(m, ToolMessage)]
    assert any("JSON" in m.text() for m in tool_msgs)
    assert ctx.messages[-1].text() == "Using JSON, here you go"
    assert not ctx.tool_state.waiting_for_input
    assert not any(isinstance(e, UserInputRequestEvent) for e in events2)


@pytest.mark.asyncio
async def test_pending_question_survives_process_death(ctx, prompts):
    """Serialize the paused ctx, reload it in a 'new process', answer, resume."""
    client = FakeChatClient(results=[make_question_call(question="Deploy where?")])
    loop = make_loop(client)
    state = ReActLoopState()

    await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state)
    )
    assert state.finish_reason == "input_needed"

    # Process dies: round-trip the context through JSON.
    revived = RunContext.model_validate_json(ctx.model_dump_json())

    pending = revived.tool_state.pending_user_input
    assert len(pending) == 1
    assert pending[0].input_question == "Deploy where?"
    assert pending[0].status == ToolCallStatus.INPUT_NEEDED

    # New process answers and resumes with a fresh loop instance.
    revived.tool_state.apply_user_answer(pending[0].id, "staging")

    client2 = FakeChatClient(results=[make_result(content="Deployed to staging")])
    loop2 = make_loop(client2)
    state2 = ReActLoopState()
    await collect(
        loop2.execute_reasoning_loop(
            ctx=revived, prompts=prompts, loop_state=state2
        )
    )

    assert state2.finish_reason == "stop"
    tool_msgs = [m for m in revived.messages if isinstance(m, ToolMessage)]
    assert any("staging" in m.text() for m in tool_msgs)


@pytest.mark.asyncio
async def test_same_batch_question_and_approval(ctx, prompts):
    """Model emits a question AND an approval-gated tool in one batch: both
    pauses surface together, both resolve on the same resume — no dangling
    state (the edge case from the framework evaluation §1.5)."""
    gated = EchoTool(name="dangerous", approval_mode=ToolApprovalMode.ASK_APPROVED)
    executor = ToolExecutor(tools=[gated], agent_name="test_agent")

    batch = make_result(
        content="",
        tool_calls=[
            ToolCall(
                id="c_q",
                tool_name=USER_INPUT_TOOL_NAME,
                parameters={"question": "Proceed how?"},
            ),
            ToolCall(id="c_d", tool_name="dangerous", parameters={}),
        ],
    )
    client = FakeChatClient(
        results=[batch, make_result(content="all done")]
    )
    loop = make_loop(client, executor)
    state = ReActLoopState()

    events = await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state)
    )

    # Both pauses arrived in the same turn-ending batch.
    assert any(isinstance(e, UserInputRequestEvent) for e in events)
    assert any(isinstance(e, ToolApprovalEvent) for e in events)
    assert state.finish_reason == "approval_needed"
    assert ctx.tool_state.waiting_for_input
    assert ctx.tool_state.waiting_for_approval

    # Resolve both on the durable state, then resume once.
    ctx.tool_state.apply_approval("c_d", approved=True)
    ctx.tool_state.apply_user_answer("c_q", "carefully")

    state2 = ReActLoopState()
    await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state2)
    )

    assert state2.finish_reason == "stop"
    assert gated.execute_called is True
    tool_msgs = {m.tool_call_id: m for m in ctx.messages if isinstance(m, ToolMessage)}
    assert "carefully" in tool_msgs["c_q"].text()
    assert "echoed" in tool_msgs["c_d"].text()
    assert not ctx.tool_state.waiting_for_input
    assert not ctx.tool_state.waiting_for_approval
    assert ctx.tool_state.stale_executions == []


@pytest.mark.asyncio
async def test_input_pause_alone_sets_input_needed(ctx, prompts):
    """A question with no approvals in the batch → finish_reason='input_needed'."""
    client = FakeChatClient(results=[make_question_call()])
    loop = make_loop(client)
    state = ReActLoopState()

    events = await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state)
    )

    assert state.finish_reason == "input_needed"
    assert events[-1].finish_reason == "input_needed"


# -------- Runtime tool registration ----------------------------------------------
@pytest.mark.asyncio
async def test_loop_auto_registers_human_input_tool(ctx, prompts):
    executor = ToolExecutor(agent_name="t")
    loop = make_loop(FakeChatClient([make_result("hi")]), executor)
    await collect(
        loop.execute_reasoning_loop(
            ctx=ctx, prompts=prompts, loop_state=ReActLoopState()
        )
    )
    assert AskUserTool.TOOL_NAME in executor.tools


@pytest.mark.asyncio
async def test_disable_human_input_skips_registration(ctx, prompts):
    executor = ToolExecutor(agent_name="t")
    loop = ReActLoop(max_loop_iterations=5, enable_human_input=False)
    loop.bind(
        name="t",
        client=FakeChatClient([make_result("hi")]),
        tool_executor=executor,
        middleware_chain=FakeMiddlewareChain(),
    )
    await collect(
        loop.execute_reasoning_loop(
            ctx=ctx, prompts=prompts, loop_state=ReActLoopState()
        )
    )
    assert AskUserTool.TOOL_NAME not in executor.tools


@pytest.mark.asyncio
async def test_loop_registers_both_runtime_tools(ctx, prompts):
    from max_ai.capabilities.tools.plan import UpdatePlanTool

    executor = ToolExecutor(agent_name="t")
    loop = make_loop(FakeChatClient([make_result("hi")]), executor)
    await collect(
        loop.execute_reasoning_loop(
            ctx=ctx, prompts=prompts, loop_state=ReActLoopState()
        )
    )
    assert AskUserTool.TOOL_NAME in executor.tools
    assert UpdatePlanTool.TOOL_NAME in executor.tools


# -------- Direct execute() fallback ----------------------------------------------
@pytest.mark.asyncio
async def test_direct_execute_returns_stored_answer():
    """Custom executors that bypass the short-circuit still complete answered
    calls — and never block on unanswered ones."""
    tool = AskUserTool()
    record = make_question_record()
    record.await_user_input("Which?")
    record.apply_user_answer("this one")

    result = await tool.execute(record)
    assert result.success is True
    assert result.result == "this one"


@pytest.mark.asyncio
async def test_direct_execute_without_answer_fails_fast():
    tool = AskUserTool()
    record = make_question_record()

    result = await tool.execute(record)
    assert result.success is False
    assert "pending" in (result.error or "").lower()


# -------- Plan drafted in the same round as a pause ------------------------------
@pytest.mark.asyncio
async def test_plan_synced_before_input_pause(ctx, prompts):
    """update_plan + question in ONE round: the plan must be synced to
    ctx.plan (and its PlanningEvent emitted) BEFORE the pause, because
    loop_state.plan_draft does not survive the pause/resume boundary."""
    from max_ai.core.event_type import PlanningEvent

    plan_and_question = make_result(
        content="",
        tool_calls=[
            ToolCall(
                id="c_plan",
                tool_name="update_plan",
                parameters={
                    "steps": [
                        {"id": 1, "description": "gather", "status": "active"},
                        {"id": 2, "description": "answer", "status": "pending"},
                    ],
                    "rationale": "two steps",
                },
            ),
            ToolCall(
                id="c_q",
                tool_name=USER_INPUT_TOOL_NAME,
                parameters={"question": "Which format?"},
            ),
        ],
    )
    client = FakeChatClient(results=[plan_and_question])
    loop = make_loop(client)
    state = ReActLoopState()

    events = await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state)
    )

    assert state.finish_reason == "input_needed"
    plan_events = [e for e in events if isinstance(e, PlanningEvent)]
    assert len(plan_events) == 1
    assert ctx.plan is not None
    assert len(ctx.plan.steps) == 2
    # The plan event precedes the pause events in the stream.
    input_index = next(
        i for i, e in enumerate(events) if isinstance(e, UserInputRequestEvent)
    )
    plan_index = next(
        i for i, e in enumerate(events) if isinstance(e, PlanningEvent)
    )
    assert plan_index < input_index


# -------- Turn-scoped call dedup --------------------------------------------------
class CountingTool(CoreTool):
    """Auto-approved tool that counts real executions."""

    def __init__(self, name="fetch"):
        super().__init__(
            name=name, description="Fetch", approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        self.calls = 0

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": True}

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        return CoreToolParameters(is_tool_valid=True, msg_error=None)

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        self.calls += 1
        return ToolResult.success_result(tool_request.id, "data-123")


@pytest.mark.asyncio
async def test_identical_call_answered_from_turn_cache(ctx, prompts):
    """The second identical (tool, args) call in one turn never re-executes:
    the framework answers it from the turn cache — deterministically, even
    if compaction evicted the first result from the model's context."""
    tool = CountingTool()
    executor = ToolExecutor(tools=[tool], agent_name="test_agent")
    client = FakeChatClient(results=[
        make_result(content="", tool_calls=[
            ToolCall(id="a1", tool_name="fetch", parameters={"q": 1}),
        ]),
        make_result(content="", tool_calls=[
            ToolCall(id="a2", tool_name="fetch", parameters={"q": 1}),  # identical
        ]),
        make_result(content="final"),
    ])
    loop = make_loop(client, executor)
    state = ReActLoopState()

    await collect(
        loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state)
    )

    assert state.finish_reason == "stop"
    assert tool.calls == 1  # executed exactly once

    tool_msgs = {m.tool_call_id: m for m in ctx.messages if isinstance(m, ToolMessage)}
    # Both calls got a ToolMessage (API contract), the second from the cache.
    assert tool_msgs["a1"].text() == "data-123"
    assert "[framework]" in tool_msgs["a2"].text()
    assert "data-123" in tool_msgs["a2"].text()


@pytest.mark.asyncio
async def test_different_arguments_are_not_deduped(ctx, prompts):
    """Same tool, different args → both execute normally."""
    tool = CountingTool()
    executor = ToolExecutor(tools=[tool], agent_name="test_agent")
    client = FakeChatClient(results=[
        make_result(content="", tool_calls=[
            ToolCall(id="b1", tool_name="fetch", parameters={"q": 1}),
        ]),
        make_result(content="", tool_calls=[
            ToolCall(id="b2", tool_name="fetch", parameters={"q": 2}),
        ]),
        make_result(content="final"),
    ])
    loop = make_loop(client, executor)

    await collect(
        loop.execute_reasoning_loop(
            ctx=ctx, prompts=prompts, loop_state=ReActLoopState()
        )
    )

    assert tool.calls == 2
