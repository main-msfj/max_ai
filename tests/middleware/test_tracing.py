"""TracingMiddleware turns a run into an OpenTelemetry trace Langfuse understands."""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from max_ai.agents import Agent
from max_ai.capabilities.middleware import TracingMiddleware, configure_langfuse
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.model.llm import ModelConfig
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode


def get_weather(city: str) -> dict:
    """Current weather for a city."""
    if city == "Atlantis":
        raise ValueError("unknown city")
    return {"city": city, "temp_c": 24}


class FakeLLM:
    model = "fake-model"

    def __init__(self, *replies, config: ModelConfig | None = None, fail: bool = False):
        self.replies = list(replies)
        self.config = config or ModelConfig()
        self.generation_options = {"max_tokens": 500}
        self.fail = fail

    async def run(self, **kwargs):
        if self.fail:
            raise RuntimeError("provider down")
        return ChatCompletionResult(message=self.replies.pop(0), usage=Usage(tokens_input=1000, tokens_output=100),
                                    model="fake-model-2026", finish_reason="stop")


def call(city: str) -> AssistantMessage:
    return AssistantMessage(source="llm", content="", tool_calls=[
        ToolCall(id=f"c-{city}", tool_name="get_weather", parameters={"city": city})])


def say(text: str) -> AssistantMessage:
    return AssistantMessage(source="llm", content=text)


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def traced_agent(tmp_path, llm, exporter, **options) -> Agent:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tool = FunctionAsTool(get_weather, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    agent = Agent(name="weather", description="d", instructions="i", client=llm, toolset=[tool],
                  workspace=LocalWorkspace(root=tmp_path),
                  middlewares=[TracingMiddleware(tracer_provider=provider, **options)])
    agent.reasoning.max_connection_retries = 0
    return agent


async def run(agent: Agent, text: str = "clima en Lima"):
    async with agent:
        return await agent.run(text, run_context=RunContext(user_id="ana", session_id="chat-1"))


def by_name(exporter: InMemorySpanExporter) -> dict[str, list]:
    spans: dict[str, list] = {}
    for span in exporter.get_finished_spans():
        spans.setdefault(span.name, []).append(span)
    return spans


async def test_a_run_is_one_trace_with_model_and_tool_spans(tmp_path, exporter):
    prices = ModelConfig(input_cost_per_mtok=2.0, output_cost_per_mtok=10.0)
    await run(traced_agent(tmp_path, FakeLLM(call("Lima"), say("24 grados"), config=prices), exporter))
    spans = by_name(exporter)
    root = spans["invoke_agent weather"][0]
    chats, tool = spans["chat fake-model"], spans["execute_tool get_weather"][0]

    assert len(chats) == 2
    assert {s.context.trace_id for s in exporter.get_finished_spans()} == {root.context.trace_id}
    assert all(s.parent.span_id == root.context.span_id for s in [*chats, tool])

    assert root.attributes["user.id"] == "ana" and root.attributes["session.id"] == "chat-1"
    assert root.attributes["langfuse.observation.type"] == "agent"
    assert root.attributes["langfuse.trace.input"] == "clima en Lima"
    assert root.attributes["langfuse.trace.output"] == "24 grados"
    assert root.attributes["max_ai.finish_reason"] == "stop"

    first = chats[0].attributes
    assert first["langfuse.observation.type"] == "generation"
    assert (first["gen_ai.request.model"], first["gen_ai.response.model"]) == ("fake-model", "fake-model-2026")
    assert (first["gen_ai.usage.input_tokens"], first["gen_ai.usage.output_tokens"]) == (1000, 100)
    assert json.loads(first["langfuse.observation.output"]) == {
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Lima"}}]}
    assert json.loads(first["langfuse.observation.cost_details"])["total"] == pytest.approx(0.003)

    assert json.loads(tool.attributes["gen_ai.tool.call.arguments"]) == {"city": "Lima"}
    assert json.loads(tool.attributes["gen_ai.tool.call.result"])["temp_c"] == 24


async def test_failed_tools_and_crashed_runs_are_marked_as_errors(tmp_path, exporter):
    await run(traced_agent(tmp_path, FakeLLM(call("Atlantis"), say("no sé")), exporter))
    tool = by_name(exporter)["execute_tool get_weather"][0]
    assert tool.status.status_code == StatusCode.ERROR

    exporter.clear()
    with pytest.raises(Exception, match="provider down"):
        await run(traced_agent(tmp_path, FakeLLM(fail=True), exporter))
    spans = by_name(exporter)
    assert spans["chat fake-model"][0].status.status_code == StatusCode.ERROR
    root = spans["invoke_agent weather"][0]  # still ended, with the error
    assert root.status.status_code == StatusCode.ERROR and root.events[0].name == "exception"


async def test_content_can_be_left_out(tmp_path, exporter):
    await run(traced_agent(tmp_path, FakeLLM(call("Lima"), say("24 grados")), exporter, capture_content=False))
    for span in exporter.get_finished_spans():
        keys = set(span.attributes)
        assert not keys & {"langfuse.trace.input", "langfuse.observation.input", "langfuse.observation.output",
                           "gen_ai.tool.call.arguments", "gen_ai.tool.call.result"}
    assert by_name(exporter)["chat fake-model"][0].attributes["gen_ai.usage.input_tokens"] == 1000


def test_tracing_config_travels_with_the_agent():
    back = TracingMiddleware.deserialize(TracingMiddleware(capture_content=False).serialize())
    assert back.capture_content is False


def test_configure_langfuse_needs_keys_and_builds_the_exporter(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        configure_langfuse(set_global=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-2")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000/")
    provider = configure_langfuse(set_global=False)
    exporter = provider._active_span_processor._span_processors[0].span_exporter
    assert exporter._endpoint == "http://localhost:3000/api/public/otel/v1/traces"
    assert exporter._headers["Authorization"].startswith("Basic ")
    provider.shutdown()
