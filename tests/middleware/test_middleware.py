"""Middleware sees and can change the run, its model calls and its tool calls."""

from __future__ import annotations

import pytest

from max_ai.agents import Agent
from max_ai.base import component
from max_ai.base.middleware import CoreMiddleware, MiddlewareConfig, StopRun
from max_ai.capabilities.middleware import BudgetMiddleware, LoggingMiddleware
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.model.llm import ModelConfig
from max_ai.types.completions import ChatCompletionChunk, ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolResult
from max_ai.types.tools import ToolApprovalMode

RAN: list[str] = []


def get_weather(city: str) -> dict:
    """Current weather for a city."""
    RAN.append(city)
    return {"city": city, "temp_c": 24}


def call(*cities: str) -> AssistantMessage:
    return AssistantMessage(source="llm", content="", tool_calls=[
        ToolCall(id=f"c-{city}", tool_name="get_weather", parameters={"city": city}) for city in cities
    ])


def say(text: str) -> AssistantMessage:
    return AssistantMessage(source="llm", content=text)


class FakeLLM:
    model = "fake"

    def __init__(self, *replies: AssistantMessage, config: ModelConfig | None = None, fail: int = 0):
        self.replies = list(replies)
        self.config = config or ModelConfig()
        self.generation_options = {"max_tokens": 500}
        self.fail = fail
        self.options: list[dict] = []

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.options.append(kwargs)
        if self.fail:
            self.fail -= 1
            raise RuntimeError("provider down")
        message = self.replies.pop(0)
        usage = Usage(tokens_input=1000, tokens_output=100)
        if not stream:
            return ChatCompletionResult(message=message, usage=usage, model="fake", finish_reason="stop")

        async def chunks():
            for word in message.content.split(" ") if message.content else []:
                yield ChatCompletionChunk(content=word + " ", is_complete=False)
            for tc in message.tool_calls:
                yield ChatCompletionChunk(content="", is_complete=False, tool_call_chunk={
                    "id": tc.id, "function": {"name": tc.tool_name, "arguments": '{"city": "%s"}' % tc.parameters["city"]}})
            yield ChatCompletionChunk(content="", is_complete=True, usage=usage)
        return chunks()


def agent(tmp_path, llm, *middlewares) -> Agent:
    tool = FunctionAsTool(get_weather, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    return Agent(name="a", description="d", instructions="i", client=llm, toolset=[tool],
                 workspace=LocalWorkspace(root=tmp_path), middlewares=list(middlewares))


async def run(a: Agent, text: str = "clima en Lima", **kwargs):
    async with a:
        return await a.run(text, run_context=RunContext(user_id="u"), **kwargs)


@pytest.fixture(autouse=True)
def clean():
    RAN.clear()


# -------- HOOKS -----------------------------------------------------------
class Recorder(CoreMiddleware):
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def on_run_start(self, mw, task):
        self.seen.append(f"run_start:{'new' if task else 'resume'}")

    async def on_model_request(self, mw, request):
        self.seen.append("model_request")
        request.options["temperature"] = 0  # a hook may change the call
        return request

    async def on_model_chunk(self, mw, request, chunk):
        self.seen.append("chunk")
        return chunk

    async def on_model_response(self, mw, request, result):
        self.seen.append("model_response")
        return result

    async def on_tool_request(self, mw, request):
        self.seen.append(f"tool_request:{request.tool_name}")

    async def on_tool_response(self, mw, request, result):
        self.seen.append(f"tool_response:{result.success}")
        return result

    async def on_final_response(self, mw, message):
        self.seen.append("final_response")
        return message

    async def on_run_end(self, mw, response):
        self.seen.append(f"run_end:{response.finish_reason}")


async def test_every_step_goes_through_the_hooks(tmp_path):
    recorder, llm = Recorder(), FakeLLM(call("Lima"), say("24 grados"))
    await run(agent(tmp_path, llm, recorder))
    assert recorder.seen == [
        "run_start:new",
        "model_request", "model_response",
        "tool_request:get_weather", "tool_response:True",
        "model_request", "model_response",
        "final_response",
        "run_end:stop",
    ]
    assert llm.options == [{"temperature": 0}, {"temperature": 0}]


async def test_streaming_chunks_go_through_the_hooks(tmp_path):
    recorder = Recorder()
    await run(agent(tmp_path, FakeLLM(say("hace sol hoy"), ), recorder), stream_tokens=True)
    assert recorder.seen.count("chunk") == 4  # three words + the closing chunk
    assert "final_response" in recorder.seen


class Mapper(CoreMiddleware):
    async def on_final_response(self, mw, message):
        return message.model_copy(update={"content": message.text().upper()})


async def test_the_final_answer_can_be_mapped(tmp_path):
    response = await run(agent(tmp_path, FakeLLM(say("hace sol")), Mapper()))
    assert response.final_text == "HACE SOL"


class NoTools(CoreMiddleware):
    async def on_tool_request(self, mw, request):
        return ToolResult.execution_error(request.record.id, "tools are off")


async def test_a_tool_can_be_blocked_without_running(tmp_path):
    await run(agent(tmp_path, FakeLLM(call("Lima"), say("no pude")), NoTools()))
    assert RAN == []


class Fallback(CoreMiddleware):
    async def on_model_error(self, mw, request, error):
        return ChatCompletionResult(message=say("respaldo"), usage=Usage(), model="backup", finish_reason="stop")


async def test_a_failed_model_call_can_be_recovered(tmp_path):
    llm = FakeLLM(fail=1)
    llm.generation_options = {}
    a = agent(tmp_path, llm, Fallback())
    a.reasoning.max_connection_retries = 0
    assert (await run(a)).final_text == "respaldo"


class Refuse(CoreMiddleware):
    async def on_run_start(self, mw, task):
        raise StopRun("Fuera de horario.")


async def test_stop_run_ends_the_turn_cleanly(tmp_path):
    llm = FakeLLM(say("nunca"))
    response = await run(agent(tmp_path, llm, Refuse()))
    assert (response.finish_reason, response.stop_message) == ("stopped", "Fuera de horario.")
    assert llm.replies  # the model was never called


# -------- BUDGET -----------------------------------------------------------
async def test_budget_stops_before_the_next_model_call(tmp_path):
    llm = FakeLLM(call("Lima"), say("nunca llega"))
    response = await run(agent(tmp_path, llm, BudgetMiddleware(max_model_calls=1)))
    assert response.finish_reason == "budget_exceeded"
    assert "model-call budget (1)" in response.stop_message
    assert RAN == []  # the model can't read their results any more: they don't run
    last = response.context.messages[-1]
    assert last.role == "tool" and "Not run" in last.content  # every call has its result


async def test_budget_refuses_tools_over_the_limit(tmp_path):
    response = await run(agent(tmp_path, FakeLLM(call("Lima", "Quito"), say("x")),
                               BudgetMiddleware(max_tool_calls=1)))
    assert RAN == ["Lima"]
    assert "Not run" in response.context.messages[-1].content
    assert response.finish_reason == "budget_exceeded"


async def test_budget_counts_tokens_and_cost(tmp_path):
    prices = ModelConfig(input_cost_per_mtok=2.0, output_cost_per_mtok=10.0)
    budget = BudgetMiddleware(max_cost_usd=1.0)
    response = await run(agent(tmp_path, FakeLLM(call("Lima"), say("ok"), config=prices), budget))
    spent = budget.spent(response.context)
    assert spent["tokens"] == 2200 and spent["model_calls"] == 2 and spent["tool_calls"] == 1
    assert spent["cost_usd"] == pytest.approx(2 * (1000 * 2 + 100 * 10) / 1e6)

    with pytest.raises(ValueError, match="input_cost_per_mtok"):
        await run(agent(tmp_path, FakeLLM(say("x")), BudgetMiddleware(max_cost_usd=1.0)))


async def test_a_new_task_starts_a_new_budget(tmp_path):
    budget = BudgetMiddleware(max_model_calls=1)
    a = agent(tmp_path, FakeLLM(say("uno"), say("dos")), budget)
    ctx = RunContext(user_id="u")
    async with a:
        first = (await a.run("hola", run_context=ctx)).final_text
        second = await a.run("otra", run_context=ctx)
    assert (first, second.final_text, second.finish_reason) == ("uno", "dos", "stop")


# -------- SERIALIZATION -----------------------------------------------------------
class TagConfig(MiddlewareConfig):
    tag: str = "x"


class Tag(CoreMiddleware):
    component_schema = TagConfig

    def __init__(self, tag: str = "x") -> None:
        self.tag = tag


def test_middlewares_travel_with_the_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(component, "_ALLOWED_PREFIXES", {"max_ai."})
    llm_free = Agent(name="a", description="d", instructions="i",
                     client=_openai(monkeypatch), workspace=LocalWorkspace(root=tmp_path),
                     middlewares=[BudgetMiddleware(max_tokens=5000), LoggingMiddleware("debug"), Tag("t1")])
    row = llm_free.serialize().model_dump_json()
    component.allow_providers(Tag.__module__ + ".")
    back = Agent.deserialize(row)
    budget, logging_mw, tag = back.middlewares
    assert (budget.max_tokens, logging_mw.level, tag.tag) == (5000, "debug", "t1")


def _openai(monkeypatch):
    from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    return OpenAIChatCompletionClient(model="gpt-4o-mini")
