"""Tests for ReActLoop with planning and self-eval enabled."""

from __future__ import annotations

import pytest
import typing as t

from max_ai.reasoning.react_planning import (
    ReActLoopPlanning as ReActLoop,
    ReActLoopPlanningState as ReActLoopState,
)
from max_ai.reasoning.plan import AgentPlan, PlanStep
from max_ai.reasoning.eval import EvalResult, EvalCheck
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
              tool_executor=None):
    loop = ReActLoop(
        max_loop_iterations=3,
        enable_planning=enable_planning,
        enable_self_eval=enable_self_eval,
        eval_threshold=eval_threshold,
        max_eval_retries=max_eval_retries,
        eval_max_extra_tokens=eval_max_extra_tokens,
        enable_intermediate_eval=enable_intermediate_eval,
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

    planning_events = [e for e in events if isinstance(e, PlanningEvent)]
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

    system_msgs = [m for m in ctx.messages if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1
    assert "Execution plan" in system_msgs[0].content
    assert "Search for info" in system_msgs[0].content


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
async def test_planning_disabled_no_events(ctx, prompts):
    """enable_planning=False → cero PlanningEvents."""
    client = FakeChatClient(results=[make_result(content="answer")])
    loop = make_loop(client, enable_planning=False)
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=state))
    print_events(events, "test_planning_disabled_no_events")

    assert not any(isinstance(e, PlanningEvent) for e in events)
    print("  No PlanningEvents — correct")


# -------- EVAL TESTS ----------------------------------------------------------
@pytest.mark.asyncio
async def test_eval_passes_on_high_score(ctx, prompts):
    """Todos los checks pasan → passed=True, score=1.0."""
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
    """enable_self_eval=False → cero EvalEvents."""
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
    loop = make_loop(client, enable_planning=True, enable_self_eval=True)
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
    client = FakeChatClient(results=[
        make_result(content="answer"),
        make_result(structured_output=make_eval(passed_checks=3, total_checks=3)),
    ])
    loop = make_loop(client, enable_self_eval=True)
    loop.eval_criteria = ["CONSTRUCTOR_CRITERION"]  # set como si viniera del __init__
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
    client = FakeChatClient(results=[
        make_result(content="answer"),
        make_result(structured_output=make_eval(passed_checks=3, total_checks=3)),
    ])
    loop = make_loop(client, enable_self_eval=True)
    loop.eval_criteria = ["CONSTRUCTOR_CRITERION"]
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
