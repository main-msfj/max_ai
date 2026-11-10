"""Deferred tools stay out of the model's tool list but still run when called."""

from __future__ import annotations

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.tools.decorator import tool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.model.llm import ModelConfig
from max_ai.core.tool.registry import ToolRegistry
from max_ai.errors.tools import ToolRegistryError
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext

WEATHER: list[str] = []


def weather(city: str) -> str:
    """Current forecast for a city."""
    WEATHER.append(city)
    return "sunny"


def hello(name: str) -> str:
    """Say hello to someone by name."""
    return f"hola {name}"


class RecordingLLM:
    model = "scripted"
    config = ModelConfig()
    generation_options = {"max_tokens": 200}

    def __init__(self, *replies: AssistantMessage) -> None:
        self.replies = list(replies)
        self.sent: list[list[str]] = []

    async def run(self, **kwargs):
        self.sent.append([t.name for t in kwargs.get("tools") or []])
        message = self.replies.pop(0)
        return ChatCompletionResult(
            message=message, usage=Usage(), model="scripted", finish_reason="stop"
        )


def say(text: str) -> AssistantMessage:
    return AssistantMessage(source="llm", content=text)


def agent(tmp_path, llm) -> Agent:
    toolset = [tool(hello), tool(weather, deferred=True)]
    a = Agent(
        name="a",
        description="d",
        instructions="i",
        client=llm,
        toolset=toolset,
        workspace=LocalWorkspace(root=tmp_path),
    )
    a.reasoning.max_connection_retries = 0
    return a


@pytest.fixture(autouse=True)
def reset():
    WEATHER.clear()


def test_the_registry_splits_loaded_and_deferred():
    registry = ToolRegistry([tool(hello), tool(weather, deferred=True)])
    assert [t.name for t in registry.loaded_tools()] == ["hello"]
    assert [t.name for t in registry.deferred_tools()] == ["weather"]
    assert registry.get("weather") is not None


async def test_the_model_never_receives_a_deferred_tool(tmp_path):
    llm = RecordingLLM(say("ok"))
    async with agent(tmp_path, llm) as a:
        await a.run("hola", run_context=RunContext(user_id="u"))
    assert "hello" in llm.sent[0] and "weather" not in llm.sent[0]


async def test_a_deferred_tool_called_by_name_still_runs(tmp_path):
    call = AssistantMessage(
        source="llm",
        content="",
        tool_calls=[ToolCall(id="w", tool_name="weather", parameters={"city": "Lima"})],
    )
    async with agent(tmp_path, RecordingLLM(call, say("listo"))) as a:
        done = await a.run("¿clima?", run_context=RunContext(user_id="u"))
    assert done.finish_reason == "stop" and WEATHER == ["Lima"]


def test_the_registry_refuses_duplicates_and_non_tools():
    registry = ToolRegistry([tool(hello)])
    with pytest.raises(ToolRegistryError, match="Duplicate tool name: 'hello'") as duplicate:
        registry.register(tool(hello))
    with pytest.raises(ToolRegistryError, match="got int") as wrong:
        registry.register(42)  # type: ignore[arg-type]
    assert (duplicate.value.kind, wrong.value.kind) == ("duplicate_name", "not_a_tool")
