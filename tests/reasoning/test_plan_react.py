"""Tests for ReActLoop with planning and self-eval enabled."""

from __future__ import annotations

import pytest
import typing as t

from max_ai.reasoning.react_planning import (
    ReActLoopPlanning as ReActLoop,
    ReActLoopPlanningState as ReActLoopState,
)
from max_ai.reasoning.plan import AgentPlan, PlanStep
from max_ai.reasoning.eval import EvalResult, EvalCheck, EvalConfig
from max_ai.core.messages import AssistantMessage, ToolMessage, ToolCall, UserMessage, SystemMessage
from max_ai.core.event_type import (
    PlanningEvent,
    EvalEvent,
    ReasoningCompleteEvent,
)
from max_ai.core.models import ModelConfig
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.completions import ChatCompletionResult, Usage


# -------- FAKES ---------------------------------------------------------------
class FakeChatClient:
    def __init__(self, results: list):
        self.model = "fake"
        self.config = ModelConfig()
        self._results = list(results)
        self._call_count = 0
        # Spy: messages seen on the eval call (output_format=EvalResult).
        # Lets a test assert which criteria reached the eval prompt.
        self.eval_messages: list | None = None

    async def run(self, ctx, prompts, tools=None, output_format=None, stream=False, **kw):
        self._call_count += 1
        if output_format is not None and output_format.__name__ == "EvalResult":
            self.eval_messages = list(ctx.messages)
        result = self._results.pop(0)
        print(f"\n  [FakeClient call #{self._call_count}] output_format={output_format.__name__ if output_format else None} → structured_output={type(result.message.structured_output).__name__ if result.message.structured_output else None}")
        return result

    def format_messages(self, ctx, prompts): return []
    def build_api_messages(self, messages): return []
    def build_tool_schema(self, tools): return []
    def normalize_usage_stats(self, usage): return Usage()
    async def complete(self, *a, **kw): raise NotImplementedError
    async def stream(self, *a, **kw): raise NotImplementedError


class FakeToolExecutor:
    def __init__(self):
        self.tools = {}

    async def execute_tool_call(self, ctx, records, cancellation_token=None):
        return
        yield


class FakeFailingToolExecutor:
    """Executor whose tools always fail — yields an error ToolMessage
    per requested record. Drives the intermediate-eval check."""

    def __init__(self):
        self.tools = {}

    async def execute_tool_call(self, ctx, records, cancellation_token=None):
        for record in records:
            yield ToolMessage.error_message(
                tool_call_id=record.id,
                tool_name=record.tool_name,
                error="boom: tool blew up",
                source="fake-tool",
            )


class FakeSuccessToolExecutor:
    """Executor whose tools always succeed — yields one successful
    ToolMessage per requested record. Drives piece 4 rule A."""

    def __init__(self):
        self.tools = {}

    async def execute_tool_call(self, ctx, records, cancellation_token=None):
        for record in records:
            yield ToolMessage.success_message(
                tool_call_id=record.id,
                tool_name=record.tool_name,
                content="ok: tool ran",
                source="fake-tool",
            )


class FakeMiddlewareChain:
    async def execute(self, action, ctx, data, func, metadata=None):
        yield await func(ctx)

    async def execute_stream(self, action, ctx, data, stream_func, metadata=None):
        async for chunk in stream_func(ctx):
            yield chunk


# -------- HELPERS -------------------------------------------------------------
def make_result(content="done", structured_output=None, tokens=0):
    return ChatCompletionResult(
        message=AssistantMessage(
            source="fake",
            content=content,
            structured_output=structured_output,
        ),
        usage=Usage(
            llm_calls=1,
            attempts_to_call_api=1,
            tokens_output=tokens,
        ),
        model="fake",
        finish_reason="stop",
    )


def make_tool_result(tool_name="do_thing", call_id="call_1", tokens=0):
    """LLM response that requests one tool call (drives the tool path)."""
    return ChatCompletionResult(
        message=AssistantMessage(
            source="fake",
            content="",
            tool_calls=[ToolCall(id=call_id, tool_name=tool_name, parameters={})],
        ),
        usage=Usage(llm_calls=1, attempts_to_call_api=1, tokens_output=tokens),
        model="fake",
        finish_reason="tool_calls",
    )


def make_plan():
    return AgentPlan(
        steps=[
            PlanStep(id=1, description="Search for info", tool_hint="web_search"),
            PlanStep(id=2, description="Summarize findings"),
        ],
        rationale="Start broad then synthesize",
    )


def empty_plan():
    """A plan with no steps. Present (so the loop won't generate one) but
    inert — active_step()/next_pending() return None, so no plan-progress
    fires. Used by eval tests that don't care about plan mechanics."""
    return AgentPlan(steps=[], rationale="no steps")


def make_eval(passed_checks: int, total_checks: int, issues: list[str] | None = None):
    checks = [
        EvalCheck(
            criterion=f"criterion_{i}",
            passed=(i < passed_checks),
            reason="ok" if i < passed_checks else "missing",
        )
        for i in range(total_checks)
    ]
    return EvalResult(
        checks=checks,
        issues=issues or [],
        suggestions=[],
        summary="eval summary",
    )


def make_loop(client, enable_planning=False, enable_self_eval=False,
              eval_threshold=0.8, max_eval_retries=1,
              eval_max_extra_tokens=None, enable_intermediate_eval=False,
              tool_executor=None, max_step_retries=2, max_loop_iterations=3):
    """Build a bound ReActLoopPlanning for tests.

    Planning is now intrinsic to the loop (it always plans when ctx.plan is
    None), so ``enable_planning`` no longer maps to a constructor flag — the
    planning-focused tests control behaviour by seeding ``ctx.plan`` (or not).
    The old eval flags are translated into an ``EvalConfig``: any eval flag
    being set builds one, otherwise ``eval`` stays None (no self-eval).
    """
    eval_cfg = None
    if enable_self_eval or enable_intermediate_eval:
        eval_cfg = EvalConfig(
            self_eval=enable_self_eval,
            intermediate=enable_intermediate_eval,
            threshold=eval_threshold,
            max_retries=max_eval_retries,
            max_extra_tokens=eval_max_extra_tokens,
        )
    loop = ReActLoop(
        max_loop_iterations=max_loop_iterations,
        eval=eval_cfg,
        max_step_retries=max_step_retries,
    )
    loop.bind(
        name="test_agent",
        client=client,
        tool_executor=tool_executor or FakeToolExecutor(),
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


def print_events(events: list, label: str = "") -> None:
    print(f"\n{'='*55}")
    if label:
        print(f"  {label}")
    print(f"{'='*55}")
    for ev in events:
        print(f"  {ev.event_type:35s} | {getattr(ev, 'phase', getattr(ev, 'finish_reason', ''))}")
    print(f"{'='*55}")


def print_messages(ctx: RunContext, label: str = "") -> None:
    print(f"\n  --- messages in ctx ({label}) ---")
    for i, m in enumerate(ctx.messages):
        role = getattr(m, "role", "?")
        source = getattr(m, "source", "?")
        preview = (m.content or "")[:80].replace("\n", " ")
        print(f"  [{i}] role={role} source={source}: {preview}")
    print()


# -------- PLANNING TESTS ------------------------------------------------------
@pytest.mark.asyncio
async def test_planning_emits_events(ctx, prompts):
    """enable_planning=True → PlanningEvent(start) y PlanningEvent(complete)."""
    client = FakeChatClient(results=[
        make_result(structured_output=make_plan()),
        make_result(content="final answer"),
    ])
    loop = make_loop(client, enable_planning=True)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_planning_emits_events")

    # This test focuses on the planning step's own events (start/complete).
    # The phase="progress" event from plan-progress injection has its own
    # dedicated test below.
    planning_events = [
        e
        for e in events
        if isinstance(e, PlanningEvent) and e.phase in ("start", "complete")
    ]
    assert len(planning_events) == 2
    assert planning_events[0].phase == "start"
    assert planning_events[1].phase == "complete"
    assert planning_events[1].plan is not None

    print(f"\n  Plan steps:")
    for s in planning_events[1].plan.steps:
        print(f"    Step {s.id}: {s.description} [{s.tool_hint}]")
    print(f"  Rationale: {planning_events[1].plan.rationale}")


@pytest.mark.asyncio
async def test_planning_injects_system_message(ctx, prompts):
    """El plan se inyecta en ctx.messages como SystemMessage."""
    client = FakeChatClient(results=[
        make_result(structured_output=make_plan()),
        make_result(content="answer"),
    ])
    loop = make_loop(client, enable_planning=True)
    state = ReActLoopState()

    await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_messages(ctx, "after planning")

    # Filter by source: the planning step emits source="planning"; piece 3's
    # plan-progress injection adds a separate source="plan-progress" message.
    plan_msgs = [
        m
        for m in ctx.messages
        if isinstance(m, SystemMessage) and m.source == "planning"
    ]
    assert len(plan_msgs) == 1
    assert "Execution plan" in plan_msgs[0].content
    assert "Search for info" in plan_msgs[0].content


@pytest.mark.asyncio
async def test_planning_failed_continues(ctx, prompts):
    """Si planning falla (structured_output=None), el loop continúa igual."""
    client = FakeChatClient(results=[
        make_result(structured_output=None),
        make_result(content="answer anyway"),
    ])
    loop = make_loop(client, enable_planning=True)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_planning_failed_continues")

    planning_events = [e for e in events if isinstance(e, PlanningEvent)]
    assert any(e.phase == "failed" for e in planning_events)
    assert state.finish_reason == "stop"
    print(f"\n  finish_reason={state.finish_reason} (loop continued after planning failure)")


@pytest.mark.asyncio
async def test_planning_always_runs_when_no_plan(ctx, prompts):
    """Planning is intrinsic to this loop: with no ctx.plan it always plans
    (no enable flag), emitting start/complete PlanningEvents."""
    client = FakeChatClient(results=[
        make_result(structured_output=make_plan()),
        make_result(content="answer"),
    ])
    loop = make_loop(client)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_planning_always_runs_when_no_plan")

    phases = [e.phase for e in events if isinstance(e, PlanningEvent)]
    assert "start" in phases and "complete" in phases


@pytest.mark.asyncio
async def test_planning_skipped_when_plan_already_present(ctx, prompts):
    """If ctx.plan is already set (e.g. a resume), the loop does NOT replan —
    no PlanningEvent(start) is emitted, the existing plan is used as-is."""
    ctx.plan = make_plan()
    client = FakeChatClient(results=[make_result(content="answer")])
    loop = make_loop(client)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    assert not any(
        isinstance(e, PlanningEvent) and e.phase == "start" for e in events
    )


# -------- PLAN-PROGRESS (piece 3) TESTS ---------------------------------------
@pytest.mark.asyncio
async def test_plan_progress_activates_and_injects(ctx, prompts):
    """ctx.plan set → first iteration activates step 1, injects a
    plan-progress SystemMessage and emits PlanningEvent(phase="progress")."""
    # Plan is set directly (no _planning_step) to isolate piece 3.
    ctx.plan = make_plan()
    client = FakeChatClient(results=[make_result(content="answer")])
    loop = make_loop(client, enable_planning=False)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_plan_progress_activates_and_injects")
    print_messages(ctx, "after plan-progress")

    # Two progress events: one when piece 3 activates step 1, one when
    # piece 4 (rule C) marks it done at the final answer.
    progress = [e for e in events if isinstance(e, PlanningEvent) and e.phase == "progress"]
    assert len(progress) == 2
    assert progress[0].plan is ctx.plan

    # Step 1 was activated, then closed by piece 4 (rule C: the final
    # answer had no tool calls). Step 2 was never reached.
    assert ctx.plan.steps[0].status == "done"
    assert ctx.plan.steps[1].status == "pending"

    # A plan-progress message naming step 1 was injected.
    pp = [m for m in ctx.messages if isinstance(m, SystemMessage) and m.source == "plan-progress"]
    assert len(pp) == 1
    assert "step 1" in pp[0].content
    assert "Search for info" in pp[0].content


@pytest.mark.asyncio
async def test_plan_progress_advances_after_successful_tool(ctx, prompts):
    """piece 4 (rule A + C, the research case): step 1 uses a tool that
    succeeds → marked done; step 2 is then activated and finishes as pure
    LLM output → marked done. The plan advances end to end."""
    ctx.plan = make_plan()
    # iter 1: step 1 runs a tool (browser-like) that succeeds -> done.
    # iter 2: step 2 is the summary (no tool) -> done, loop ends.
    client = FakeChatClient(results=[
        make_tool_result(),
        make_result(content="here is the summary"),
    ])
    loop = make_loop(
        client, enable_planning=False, tool_executor=FakeSuccessToolExecutor()
    )
    state = ReActLoopState()

    await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_messages(ctx, "after research plan")

    # Both steps fully done — the plan advanced end to end.
    assert ctx.plan.steps[0].status == "done"
    assert ctx.plan.steps[1].status == "done"

    # Step 2 was activated only after step 1 closed: a plan-progress message
    # naming step 2 must have been injected.
    pp = [m for m in ctx.messages if isinstance(m, SystemMessage) and m.source == "plan-progress"]
    assert any("step 2" in m.content for m in pp)


@pytest.mark.asyncio
async def test_plan_progress_only_advances_on_success(ctx, prompts):
    """piece 4 (rule A) guards on all(m.success): a successful tool round
    closes the active step, but the rule never fires unless tools succeeded.
    Verified by checking the active step advances exactly once per success."""
    ctx.plan = make_plan()
    # iter 1: tool succeeds -> step 1 done. iter 2: final answer -> step 2 done.
    client = FakeChatClient(results=[
        make_tool_result(),
        make_result(content="summary"),
    ])
    loop = make_loop(
        client, enable_planning=False, tool_executor=FakeSuccessToolExecutor()
    )
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    # Step 1 closed by rule A (tool success), step 2 by rule C (final answer).
    assert [s.status for s in ctx.plan.steps] == ["done", "done"]
    # No step was left active/pending at the end — the whole plan completed.
    assert all(s.status == "done" for s in ctx.plan.steps)


# -------- REPLAN (piece 5) TESTS ----------------------------------------------
@pytest.mark.asyncio
async def test_replan_after_step_fails(ctx, prompts):
    """piece 5: when a step fails max_step_retries times it is marked failed
    and the loop replans — ctx.plan is replaced by the new plan."""
    ctx.plan = make_plan()  # steps 1, 2
    recovery_plan = AgentPlan(
        steps=[PlanStep(id=99, description="recovery step")],
        rationale="retry differently",
    )
    client = FakeChatClient(results=[
        make_tool_result(),                          # iter 1: step 1 -> tool (fails)
        make_result(structured_output=recovery_plan),  # replan -> new plan
        make_result(content="done"),                 # iter 2: step 99 -> final answer
    ])
    loop = make_loop(
        client,
        enable_planning=False,
        tool_executor=FakeFailingToolExecutor(),
        max_step_retries=1,          # replan on the first failure
        max_loop_iterations=5,
    )
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_replan_after_step_fails")

    # The failure was counted for step 1.
    assert state.step_failures.get(1) == 1
    # ctx.plan was replaced by the recovery plan and it completed.
    assert [s.id for s in ctx.plan.steps] == [99]
    assert ctx.plan.steps[0].status == "done"

    # A replan emitted its own start/complete planning events.
    phases = [e.phase for e in events if isinstance(e, PlanningEvent)]
    assert "start" in phases and "complete" in phases


@pytest.mark.asyncio
async def test_no_replan_before_threshold(ctx, prompts):
    """piece 5: a single failure under max_step_retries does NOT replan —
    the step stays in the original plan and is retried."""
    ctx.plan = make_plan()  # steps 1, 2
    # One failing tool turn, then the LLM gives up with a final answer. With
    # max_step_retries=2, the single failure must not trigger a replan.
    client = FakeChatClient(results=[
        make_tool_result(),
        make_result(content="answer"),
    ])
    loop = make_loop(
        client,
        enable_planning=False,
        tool_executor=FakeFailingToolExecutor(),
        max_step_retries=2,
        max_loop_iterations=5,
    )
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    # Failure counted but below threshold → original plan kept (ids 1, 2).
    assert state.step_failures.get(1) == 1
    assert [s.id for s in ctx.plan.steps] == [1, 2]
    # No replan: no PlanningEvent(phase="start") was emitted (planning was off).
    assert not any(
        isinstance(e, PlanningEvent) and e.phase == "start" for e in events
    )


@pytest.mark.asyncio
async def test_plan_progress_skipped_when_planning_fails(ctx, prompts):
    """If planning fails (no plan produced), ctx.plan stays None and the loop
    injects no plan-progress messages or progress events."""
    client = FakeChatClient(results=[
        make_result(structured_output=None),  # planning fails
        make_result(content="answer"),
    ])
    loop = make_loop(client)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_plan_progress_skipped_when_planning_fails")

    assert ctx.plan is None
    assert not any(isinstance(e, PlanningEvent) and e.phase == "progress" for e in events)
    assert not any(
        isinstance(m, SystemMessage) and m.source == "plan-progress" for m in ctx.messages
    )


# -------- EVAL TESTS ----------------------------------------------------------
@pytest.mark.asyncio
async def test_eval_passes_on_high_score(ctx, prompts):
    """Todos los checks pasan → passed=True, score=1.0."""
    ctx.plan = empty_plan()  # plan present (so the loop won't plan) but inert
    client = FakeChatClient(results=[
        make_result(content="good answer"),
        make_result(structured_output=make_eval(passed_checks=3, total_checks=3)),
    ])
    loop = make_loop(client, enable_self_eval=True, eval_threshold=0.8)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_eval_passes_on_high_score")

    eval_events = [e for e in events if isinstance(e, EvalEvent)]
    complete_ev = next(e for e in eval_events if e.phase == "complete")
    print(f"\n  score={complete_ev.score:.2f}  passed={complete_ev.passed}")
    print(f"  checks: {[c.criterion + '=' + str(c.passed) for c in complete_ev.result.checks]}")

    assert complete_ev.passed is True
    assert complete_ev.score == pytest.approx(1.0)
    assert state.finish_reason == "stop"


@pytest.mark.asyncio
async def test_eval_retries_on_low_score(ctx, prompts):
    """Score bajo → feedback inyectado → segundo intento → pasa."""
    ctx.plan = empty_plan()
    client = FakeChatClient(results=[
        make_result(content="weak answer"),
        make_result(structured_output=make_eval(
            passed_checks=1, total_checks=3,
            issues=["Missing section A", "No sources cited"],
        )),
        make_result(content="better answer"),
        make_result(structured_output=make_eval(passed_checks=3, total_checks=3)),
    ])
    loop = make_loop(client, enable_self_eval=True, eval_threshold=0.8, max_eval_retries=2)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_eval_retries_on_low_score")
    print_messages(ctx, "after retry")

    eval_events = [e for e in events if isinstance(e, EvalEvent) and e.phase == "complete"]
    print(f"\n  Eval attempt 1: score={eval_events[0].score:.2f} passed={eval_events[0].passed}")
    print(f"  Eval attempt 2: score={eval_events[1].score:.2f} passed={eval_events[1].passed}")

    assert len(eval_events) == 2
    assert eval_events[0].passed is False
    assert eval_events[1].passed is True

    feedback_msgs = [m for m in ctx.messages
                     if isinstance(m, UserMessage) and m.source == "self-eval"]
    assert len(feedback_msgs) == 1
    assert "Missing section A" in feedback_msgs[0].content
    print(f"\n  Feedback injected: {feedback_msgs[0].content[:100]}")


@pytest.mark.asyncio
async def test_eval_score_calculated_from_checks(ctx, prompts):
    """2/3 checks pasados = score 0.666, bajo threshold 0.8 → falla primer intento."""
    ctx.plan = empty_plan()
    client = FakeChatClient(results=[
        make_result(content="partial answer"),
        make_result(structured_output=make_eval(passed_checks=2, total_checks=3)),
        make_result(content="retried answer"),
        make_result(structured_output=make_eval(passed_checks=3, total_checks=3)),
    ])
    loop = make_loop(client, enable_self_eval=True, eval_threshold=0.8, max_eval_retries=2)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    eval_events = [e for e in events if isinstance(e, EvalEvent) and e.phase == "complete"]
    print(f"\n  score={eval_events[0].score:.4f} (expected ~0.6667)")
    assert eval_events[0].score == pytest.approx(2 / 3)
    assert eval_events[0].passed is False


@pytest.mark.asyncio
async def test_eval_disabled_no_events(ctx, prompts):
    """eval=None → cero EvalEvents."""
    ctx.plan = empty_plan()
    client = FakeChatClient(results=[make_result(content="answer")])
    loop = make_loop(client, enable_self_eval=False)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_eval_disabled_no_events")

    assert not any(isinstance(e, EvalEvent) for e in events)
    print("  No EvalEvents — correct")


# -------- COMBINED ------------------------------------------------------------
@pytest.mark.asyncio
async def test_planning_and_eval_together(ctx, prompts):
    """Planning + eval activos → orden: plan → react → eval → complete."""
    client = FakeChatClient(results=[
        make_result(structured_output=make_plan()),
        make_result(content="answer"),
        make_result(structured_output=make_eval(passed_checks=3, total_checks=3)),
    ])
    loop = make_loop(client, enable_self_eval=True)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_planning_and_eval_together")
    print_messages(ctx, "combined")

    types = [type(e).__name__ for e in events]
    planning_idx = next(i for i, t in enumerate(types) if t == "PlanningEvent")
    eval_idx = next(i for i, t in enumerate(types) if t == "EvalEvent")
    complete_idx = next(i for i, t in enumerate(types) if t == "ReasoningCompleteEvent")

    print(f"\n  PlanningEvent at idx={planning_idx}")
    print(f"  EvalEvent     at idx={eval_idx}")
    print(f"  CompleteEvent at idx={complete_idx}")

    assert planning_idx < eval_idx < complete_idx
    assert state.finish_reason == "stop"


# -------- COST CAP ------------------------------------------------------------
@pytest.mark.asyncio
async def test_eval_budget_cap_stops_retries(ctx, prompts):
    """eval_max_extra_tokens bajo → el cap corta los reintentos antes que
    max_eval_retries, y emite un EvalEvent(phase='skipped') visible.

    Cada respuesta reporta tokens=100, así que tras el 1er intento ya se
    gastaron 200 tokens (respuesta + eval). Con un cap de 150 y eval que
    siempre falla (0/3 checks), el cap muerde en el primer check —
    mucho antes de agotar los 5 retries permitidos.
    """
    # Holgura de results: si el cap NO cortara, max_eval_retries=5 pediría
    # hasta 12 calls. Proveemos de sobra para que un fallo del cap se
    # manifieste como "demasiados eval rounds", no como IndexError.
    ctx.plan = empty_plan()
    results = []
    for _ in range(6):
        results.append(make_result(content="weak answer", tokens=100))
        results.append(make_result(
            structured_output=make_eval(passed_checks=0, total_checks=3),
            tokens=100,
        ))
    client = FakeChatClient(results=results)
    loop = make_loop(
        client,
        enable_self_eval=True,
        eval_threshold=0.8,
        max_eval_retries=5,        # alto: NO queremos que esto sea el freno
        eval_max_extra_tokens=150, # bajo: queremos que ESTO sea el freno
    )
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_eval_budget_cap_stops_retries")

    # El cap se hizo visible.
    skipped = [e for e in events if isinstance(e, EvalEvent) and e.phase == "skipped"]
    assert len(skipped) == 1, "esperaba exactamente un EvalEvent(phase='skipped')"

    # Cortó MUY pronto: como mucho un eval completo corrió antes del corte,
    # nada que ver con los 5 retries que max_eval_retries habría permitido.
    completed_evals = [e for e in events if isinstance(e, EvalEvent) and e.phase == "complete"]
    print(f"\n  eval rounds run: {len(completed_evals)} (max_eval_retries=5 habría permitido 6)")
    assert len(completed_evals) < 5


@pytest.mark.asyncio
async def test_eval_no_cap_is_unaffected(ctx, prompts):
    """eval_max_extra_tokens=None (default) → el cap nunca actúa; el flujo
    de retries se comporta como siempre. Test de no-regresión."""
    ctx.plan = empty_plan()
    client = FakeChatClient(results=[
        make_result(content="weak answer", tokens=100),
        make_result(structured_output=make_eval(passed_checks=1, total_checks=3),
                    tokens=100),
        make_result(content="better answer", tokens=100),
        make_result(structured_output=make_eval(passed_checks=3, total_checks=3),
                    tokens=100),
    ])
    loop = make_loop(
        client,
        enable_self_eval=True,
        eval_threshold=0.8,
        max_eval_retries=2,
        eval_max_extra_tokens=None,  # sin cap
    )
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_eval_no_cap_is_unaffected")

    # Sin cap: ningún 'skipped', y el retry normal llevó a un pase final.
    assert not any(isinstance(e, EvalEvent) and e.phase == "skipped" for e in events)
    eval_completes = [e for e in events if isinstance(e, EvalEvent) and e.phase == "complete"]
    assert eval_completes[-1].passed is True
    assert state.finish_reason == "stop"


# -------- PER-RUN CRITERIA ----------------------------------------------------
@pytest.mark.asyncio
async def test_eval_per_run_criteria_overrides_constructor(ctx, prompts):
    """eval_criteria pasado por-run gana sobre el del constructor.

    El criterio por-run debe llegar al prompt de evaluación; el del
    constructor no debe aparecer. Lo verificamos espiando los mensajes
    que el _eval_step entrega al client.run().
    """
    ctx.plan = empty_plan()
    client = FakeChatClient(results=[
        make_result(content="answer"),
        make_result(structured_output=make_eval(passed_checks=3, total_checks=3)),
    ])
    loop = make_loop(client, enable_self_eval=True)
    loop.eval.criteria = ["CONSTRUCTOR_CRITERION"]  # set como si viniera del __init__
    state = ReActLoopState()

    await collect(loop.execute_reasoning_loop(
        ctx=ctx, prompts=prompts, loop_state=state,
        eval_criteria=["PER_RUN_CRITERION"],
    ))

    assert client.eval_messages is not None, "el eval debió correr y ser espiado"
    eval_text = " ".join((m.content or "") for m in client.eval_messages)
    print(f"\n  eval prompt contiene PER_RUN? {'PER_RUN_CRITERION' in eval_text}")
    print(f"  eval prompt contiene CONSTRUCTOR? {'CONSTRUCTOR_CRITERION' in eval_text}")

    assert "PER_RUN_CRITERION" in eval_text
    assert "CONSTRUCTOR_CRITERION" not in eval_text


@pytest.mark.asyncio
async def test_eval_falls_back_to_constructor_criteria(ctx, prompts):
    """Sin eval_criteria por-run → usa el del constructor."""
    ctx.plan = empty_plan()
    client = FakeChatClient(results=[
        make_result(content="answer"),
        make_result(structured_output=make_eval(passed_checks=3, total_checks=3)),
    ])
    loop = make_loop(client, enable_self_eval=True)
    loop.eval.criteria = ["CONSTRUCTOR_CRITERION"]
    state = ReActLoopState()

    await collect(loop.execute_reasoning_loop(
        ctx=ctx, prompts=prompts, loop_state=state,
        # sin eval_criteria → None → cae al constructor
    ))

    assert client.eval_messages is not None
    eval_text = " ".join((m.content or "") for m in client.eval_messages)
    assert "CONSTRUCTOR_CRITERION" in eval_text


# -------- INTERMEDIATE EVAL ---------------------------------------------------
@pytest.mark.asyncio
async def test_intermediate_eval_fires_on_tool_failure(ctx, prompts):
    """enable_intermediate_eval=True + un tool que falla → se emite un
    EvalEvent(phase='intermediate') y se inyecta feedback en ctx.messages.

    Secuencia: la 1ª respuesta del LLM pide un tool (que falla), la 2ª es
    la respuesta final sin tools. El check intermedio debe dispararse
    entre ambas.
    """
    ctx.plan = empty_plan()  # inert plan: no step mechanics interfere
    client = FakeChatClient(results=[
        make_tool_result(tool_name="do_thing"),   # pide tool → fallará
        make_result(content="final answer"),      # respuesta final
    ])
    loop = make_loop(
        client,
        enable_intermediate_eval=True,
        tool_executor=FakeFailingToolExecutor(),
    )
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_intermediate_eval_fires_on_tool_failure")

    # Se emitió el evento intermedio.
    intermediate = [e for e in events if isinstance(e, EvalEvent) and e.phase == "intermediate"]
    assert len(intermediate) == 1

    # Se inyectó el feedback con la fuente correcta y el error del tool.
    feedback = [m for m in ctx.messages
                if isinstance(m, UserMessage) and m.source == "intermediate-eval"]
    assert len(feedback) == 1
    assert "do_thing" in feedback[0].content
    print(f"\n  feedback inyectado: {feedback[0].content[:80]}")


@pytest.mark.asyncio
async def test_intermediate_eval_silent_when_tools_succeed(ctx, prompts):
    """Con el flag ON pero el tool que SÍ tiene éxito → no se dispara nada.
    El check solo reacciona a fallos."""
    client = FakeChatClient(results=[
        make_tool_result(tool_name="do_thing"),
        make_result(content="final answer"),
    ])
    # FakeToolExecutor por defecto no produce ToolMessages de error
    # (de hecho no produce ninguno), así que no hay fallo que detectar.
    loop = make_loop(client, enable_intermediate_eval=True)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    assert not any(isinstance(e, EvalEvent) and e.phase == "intermediate" for e in events)
    assert not any(isinstance(m, UserMessage) and m.source == "intermediate-eval"
                   for m in ctx.messages)


@pytest.mark.asyncio
async def test_intermediate_eval_off_by_default(ctx, prompts):
    """Sin el flag (default) → aunque un tool falle, no se dispara el check."""
    client = FakeChatClient(results=[
        make_tool_result(tool_name="do_thing"),
        make_result(content="final answer"),
    ])
    loop = make_loop(client, tool_executor=FakeFailingToolExecutor())
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))

    assert not any(isinstance(e, EvalEvent) and e.phase == "intermediate" for e in events)
