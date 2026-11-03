"""Some providers derive tool call ids from the tool name, so the same id
comes back in later rounds or twice in one reply. The run renames them
instead of failing, and every call keeps its own answer."""

from types import SimpleNamespace

from max_ai.agents.agent import Agent
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.tools import ToolApprovalMode


def get_weather(city: str) -> str:
    """Return the weather in a city."""
    return f"Sunny in {city}"


class SameIds:
    """Replies with ``rounds`` of tool calls, all named ``call_get_weather``."""

    model = "fake"
    config = SimpleNamespace(tokenizer_base="o200k_base")

    def __init__(self, *rounds: list[str]):
        self.rounds = list(rounds)

    async def run(self, ctx, prompts, tools=None, **kwargs):
        cities = self.rounds.pop(0) if self.rounds else []
        calls = [ToolCall(id="call_get_weather", tool_name="get_weather", parameters={"city": c})
                 for c in cities]
        message = AssistantMessage(source="fake", content="" if calls else "done", tool_calls=calls)
        return ChatCompletionResult(message=message, usage=Usage(), model=self.model,
                                    finish_reason="tool_calls" if calls else "stop")


async def test_repeated_ids_are_renamed_and_each_call_is_answered(tmp_path):
    client = SameIds(["Tokyo"], ["Lima", "Oslo"])  # across rounds, and twice in one reply
    tool = FunctionAsTool(get_weather, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    async with Agent(name="t", description="d", instructions="i", client=client, toolset=[tool],
                     workspace=LocalWorkspace(root=tmp_path)) as agent:
        response = await agent.run("weather?")

    assert response.final_text == "done"
    messages = response.context.messages
    ids = [c.id for m in messages if isinstance(m, AssistantMessage) for c in m.tool_calls]
    assert ids == ["call_get_weather", "call_get_weather_2", "call_get_weather_3"]
    answered = {m.tool_call_id: m.text() for m in messages if isinstance(m, ToolMessage)}
    assert [answered[i] for i in ids] == ['"Sunny in Tokyo"', '"Sunny in Lima"', '"Sunny in Oslo"']
