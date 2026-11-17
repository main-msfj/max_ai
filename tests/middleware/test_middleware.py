"""Every middleware sees the turn, each model call and each tool call, and may change their text."""

from __future__ import annotations

import pytest

from max_ai.agents import Agent
from max_ai.base import component
from max_ai.base.middleware import CoreMiddleware, MiddlewareConfig
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.middleware import BudgetMiddleware, LoggingMiddleware
from max_ai.core.messages import AssistantMessage, ToolMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.errors.middleware import StopRun
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolResult

from .fakes import RAN, FakeClient, call, say, weather_agent


@pytest.fixture(autouse=True)
def clean():
    RAN.clear()


async def run(agent: Agent, text: str | None = "clima en Lima", ctx: RunContext | None = None):
    ctx = ctx or RunContext(user_id="u", session_id="s")
    state = await agent.run(text, ctx)
    return state, ctx


def tool_answers(ctx: RunContext) -> list[str]:
    return [m.text() for m in ctx.messages if isinstance(m, ToolMessage)]


# -------- EVERY PLACE ----------------------------------------------------------------
class Recorder(CoreMiddleware):
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def turn(self, ctx, call_next):
        self.seen.append("turn start")
        async for event in call_next():
            yield event
        self.seen.append("turn end")

    async def model(self, ctx, call, call_next):
        self.seen.append("model")
        return await call_next(call)

    async def stream(self, ctx, call, call_next):
        async for chunk in call_next(call):
            self.seen.append(f"chunk:{chunk.content}")
            yield chunk

    async def tool(self, ctx, record, call_next):
        result = await call_next(record)
        self.seen.append(f"tool {record.tool_name}: {'waits' if result is None else result.success}")
        return result


async def test_every_place_goes_through_the_middlewares():
    recorder = Recorder()
    state, _ = await run(weather_agent(FakeClient(call("Lima"), say("24 grados")), recorder))
    assert state.message is not None and state.message.text() == "24 grados"
    assert recorder.seen == ["turn start", "model", "tool get_weather: True", "model", "turn end"]


async def test_a_stream_reaches_the_middlewares_line_by_line():
    recorder = Recorder()
    agent = weather_agent(FakeClient(say("hace sol\nen Lima\nhoy")), recorder)
    ctx = RunContext(user_id="u", session_id="s")
    text = "".join([piece async for piece in agent.run_stream("clima", ctx)])
    chunks = [seen for seen in recorder.seen if seen.startswith("chunk:")]
    # Cut every 3 characters by the provider; whole lines for the middleware.
    assert chunks == ["chunk:hace sol\n", "chunk:en Lima\n", "chunk:hoy", "chunk:"]
    assert text == "hace sol\nen Lima\nhoy"


async def test_every_tool_call_passes_even_one_that_waits_for_approval():
    recorder = Recorder()
    agent = weather_agent(FakeClient(call("Lima"), say("24 grados")), recorder, ask=True)
    state, ctx = await run(agent)
    assert state.finish_reason == "waiting_for_user" and RAN == []
    # The person approves: the same call passes again, now with its result.
    await agent.approve(ctx, "c-Lima")
    state, _ = await run(agent, None, ctx)
    assert RAN == ["Lima"] and state.message is not None and state.message.text() == "24 grados"
    tools = [seen for seen in recorder.seen if seen.startswith("tool")]
    assert tools == ["tool get_weather: waits", "tool get_weather: True"]


# -------- WHAT A MIDDLEWARE MAY DO --------------------------------------------------
class Upper(CoreMiddleware):
    async def model(self, ctx, call, call_next):
        result = await call_next(call)
        result.message = result.message.model_copy(update={"content": result.message.text().upper()})
        return result


async def test_the_text_can_be_changed_and_the_changed_text_is_kept():
    state, ctx = await run(weather_agent(FakeClient(say("hace sol")), Upper()))
    assert state.message is not None and state.message.text() == "HACE SOL"
    last = ctx.messages[-1]
    assert isinstance(last, AssistantMessage) and last.text() == "HACE SOL"


class NoTools(CoreMiddleware):
    async def tool(self, ctx, record, call_next):
        return ToolResult.execution_error(record.id, "tools are off")


async def test_a_tool_can_be_answered_without_running():
    _, ctx = await run(weather_agent(FakeClient(call("Lima"), say("no pude")), NoTools()))
    assert RAN == [] and "tools are off" in tool_answers(ctx)[0]


class Changer(CoreMiddleware):
    async def tool(self, ctx, record, call_next):
        record.parameters["city"] = "Quito"
        return await call_next(record)


async def test_a_tool_never_runs_with_parameters_a_middleware_changed():
    _, ctx = await run(weather_agent(FakeClient(call("Lima"), say("no pude")), Changer()))
    assert RAN == [] and "changed this call's parameters" in tool_answers(ctx)[0]


class Fallback(CoreMiddleware):
    async def model(self, ctx, call, call_next):
        try:
            return await call_next(call)
        except Exception:
            return ChatCompletionResult(message=say("respaldo"), usage=Usage(), model="backup", finish_reason="stop")


async def test_a_failed_model_call_can_be_recovered():
    state, _ = await run(weather_agent(FakeClient(fail=1), Fallback()))
    assert state.message is not None and state.message.text() == "respaldo"


class Broken(CoreMiddleware):
    async def model(self, ctx, call, call_next):
        await call_next(call)
        raise RuntimeError("bug in the middleware")


async def test_a_broken_middleware_is_skipped_and_the_model_runs_once():
    client = FakeClient(say("hace sol"))
    state, _ = await run(weather_agent(client, Broken()))
    assert state.message is not None and state.message.text() == "hace sol"
    assert client.calls == 1


class Refuse(CoreMiddleware):
    async def turn(self, ctx, call_next):
        raise StopRun("Fuera de horario.")
        yield  # an async generator, like every turn()


async def test_stop_run_ends_the_turn_cleanly():
    client = FakeClient(say("nunca"))
    state, ctx = await run(weather_agent(client, Refuse()))
    assert state.finish_reason == "stopped"
    assert state.message is not None and state.message.text() == "Fuera de horario."
    assert client.calls == 0 and ctx.inbox == []  # the model was never called; the task is not kept


# -------- BUDGET -----------------------------------------------------------------------
async def test_budget_stops_before_the_next_model_call():
    state, ctx = await run(weather_agent(FakeClient(call("Lima"), say("nunca llega")), BudgetMiddleware(max_model_calls=1)))
    assert state.finish_reason == "budget_exceeded"
    assert state.message is not None and "model-call budget (1)" in state.message.text()
    assert RAN == []  # the model can't read their results any more: they don't run
    assert "Not run" in tool_answers(ctx)[0]  # every call has its result


async def test_budget_refuses_tools_over_the_limit():
    state, ctx = await run(weather_agent(FakeClient(call("Lima", "Quito"), say("x")), BudgetMiddleware(max_tool_calls=1)))
    assert RAN == ["Lima"]
    assert "Not run" in tool_answers(ctx)[1]
    assert state.finish_reason == "budget_exceeded"


async def test_budget_counts_tokens_and_cost():
    prices = ModelConfig(input_cost_per_mtok=2.0, output_cost_per_mtok=10.0)
    budget = BudgetMiddleware(max_cost_usd=1.0)
    _, ctx = await run(weather_agent(FakeClient(call("Lima"), say("ok"), config=prices), budget))
    spent = budget.spent(ctx)
    assert spent["tokens"] == 2200 and spent["model_calls"] == 2 and spent["tool_calls"] == 1
    assert spent["cost_usd"] == pytest.approx(2 * (1000 * 2 + 100 * 10) / 1e6)

    # Without the prices a cost limit can't be kept: the turn stops, it never spends blind.
    client = FakeClient(say("x"))
    state, _ = await run(weather_agent(client, BudgetMiddleware(max_cost_usd=1.0)))
    assert state.finish_reason == "budget_exceeded" and client.calls == 0
    assert state.message is not None and "input_cost_per_mtok" in state.message.text()


async def test_a_new_task_starts_a_new_budget():
    agent = weather_agent(FakeClient(say("uno"), say("dos")), BudgetMiddleware(max_model_calls=1))
    first, ctx = await run(agent, "hola")
    second, _ = await run(agent, "otra", ctx)
    assert first.message is not None and second.message is not None
    assert (first.message.text(), second.message.text(), second.finish_reason) == ("uno", "dos", "stop")


# -------- SERIALIZATION --------------------------------------------------------------
class TagConfig(MiddlewareConfig):
    tag: str = "x"


class Tag(CoreMiddleware):
    component_schema = TagConfig

    def __init__(self, tag: str = "x") -> None:
        self.tag = tag


def test_middlewares_travel_with_the_agent(monkeypatch):
    monkeypatch.setattr(component, "_ALLOWED_PREFIXES", {"max_ai."})
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    agent = Agent("a", "d", "i", OpenRouterChatCompletionClient(model="m:free"),
                  middlewares=[BudgetMiddleware(max_tokens=5000), LoggingMiddleware("debug"), Tag("t1")])
    row = agent.serialize().model_dump_json()
    component.allow_providers(Tag.__module__ + ".")
    back = Agent.deserialize(row)
    tracing, budget, logging_mw, tag = back.middlewares  # the harness puts tracing first
    assert type(tracing).__name__ == "TracingMiddleware"
    assert (budget.max_tokens, logging_mw.level, tag.tag) == (5000, "debug", "t1")
