"""call_tool is a door: the deferred tool it names is what runs, and
Policy, approval and the transcript all see that tool."""

from __future__ import annotations

import pytest

from max_ai.agents import Agent, Policy
from max_ai.capabilities.tools.call_tool import CallToolTool
from max_ai.capabilities.tools.decorator import tool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.harness import messages as harness
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext

WEATHER: list[str] = []


@tool(deferred=True, read_only=True)
def weather(city: str) -> str:
    """Current forecast for a city."""
    WEATHER.append(city)
    return "sunny"


@tool
def hello(name: str) -> str:
    """Say hello to someone by name."""
    return f"hola {name}"


class ScriptedLLM:
    model = "scripted"
    config = ModelConfig()
    generation_options = {"max_tokens": 200}

    def __init__(self, *replies: AssistantMessage) -> None:
        self.replies = list(replies)

    async def run(self, **kwargs):
        return ChatCompletionResult(
            message=self.replies.pop(0), usage=Usage(), model="scripted", finish_reason="stop"
        )


def call(**parameters) -> AssistantMessage:
    return AssistantMessage(
        source="llm",
        content="",
        tool_calls=[ToolCall(id="c", tool_name="call_tool", parameters=parameters)],
    )


def say(text: str) -> AssistantMessage:
    return AssistantMessage(source="llm", content=text)


def agent(tmp_path, *replies, policy: Policy | None = None) -> Agent:
    a = Agent(
        name="a",
        description="d",
        instructions="i",
        client=ScriptedLLM(*replies),
        toolset=[hello, weather],
        policy=policy,
        workspace=LocalWorkspace(root=tmp_path),
    )
    a._registry.register(CallToolTool(a._registry))
    return a


async def run(tmp_path, *replies, policy: Policy | None = None) -> RunContext:
    ctx = RunContext(user_id="u")
    async with agent(tmp_path, *replies, policy=policy) as a:
        await a.run("hola", run_context=ctx)
    return ctx


def answer(ctx: RunContext) -> ToolMessage:
    return next(m for m in ctx.messages if isinstance(m, ToolMessage))


@pytest.fixture(autouse=True)
def reset():
    WEATHER.clear()


async def test_the_deferred_tool_runs_and_the_answer_keeps_the_called_name(tmp_path):
    ctx = await run(tmp_path, call(tool="weather", arguments={"city": "Lima"}), say("ok"))
    record = ctx.tool_state.records["c"]
    assert WEATHER == ["Lima"]
    assert (record.tool_name, record.parameters, record.called_as) == (
        "weather", {"city": "Lima"}, "call_tool")
    assert answer(ctx).tool_name == "call_tool" and answer(ctx).success


async def test_policy_sees_the_real_tool(tmp_path):
    ctx = await run(
        tmp_path,
        call(tool="weather", arguments={"city": "Lima"}),
        say("ok"),
        say("ok"),
        policy=Policy(deny=["weather"]),
    )
    assert WEATHER == [] and "Denied by policy: weather" in answer(ctx).content


async def test_arguments_sent_as_a_json_string_work(tmp_path):
    await run(tmp_path, call(tool="weather", arguments='{"city": "Cusco"}'), say("ok"))
    assert WEATHER == ["Cusco"]


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ({"tool": "hello", "arguments": {"name": "x"}}, harness.call_tool_loaded("hello")),
        ({"tool": "nope"}, harness.call_tool_unknown("nope", "search_tools")),
        ({"tool": "weather", "arguments": "not json"}, "is not of type 'object'"),
    ],
)
async def test_a_wrong_call_tells_the_model_what_to_fix(tmp_path, parameters, expected):
    ctx = await run(tmp_path, call(**parameters), say("ok"))
    assert ctx.tool_state.records["c"].tool_name == "call_tool"
    assert expected in answer(ctx).content and WEATHER == []
