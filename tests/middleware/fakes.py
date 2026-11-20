"""A real client whose provider is faked: the middlewares run as they do live."""

from __future__ import annotations

from max_ai.agents import Agent
from max_ai.base.middleware import CoreMiddleware
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.model.llm import ModelConfig
from max_ai.types.completions import LLMChunk, LLMResponse, Usage
from max_ai.types.tools import ToolApprovalMode

RAN: list[str] = []  # cities the tool really ran for


def get_weather(city: str) -> dict[str, object]:
    """Current weather for a city."""
    if city == "Atlantis":
        raise ValueError("unknown city")
    RAN.append(city)
    return {"city": city, "temp_c": 24}


def call(*cities: str) -> AssistantMessage:
    """The model asks for the weather of each city."""
    return AssistantMessage(source="llm", content="", tool_calls=[
        ToolCall(id=f"c-{city}", tool_name="get_weather", parameters={"city": city}) for city in cities
    ])


def say(text: str) -> AssistantMessage:
    return AssistantMessage(source="llm", content=text)


class FakeClient(OpenRouterChatCompletionClient):
    """Every call answers the next reply and costs 1,000 + 100 tokens."""

    def __init__(self, *replies: AssistantMessage, config: ModelConfig | None = None, fail: int = 0) -> None:
        super().__init__(model="fake-model", api_key="x", config=config)
        self.replies = list(replies)
        self.fail = fail  # calls that fail before the answers start
        self.calls = 0

    async def complete(self, messages, tools, output_format):  # type: ignore[override]
        self.calls += 1
        if self.fail:
            self.fail -= 1
            raise RuntimeError("provider down")
        message = self.replies.pop(0) if self.replies else say(f"respuesta {self.calls}")
        reason = "tool_calls" if message.tool_calls else "stop"
        usage = Usage(tokens_input=1000, tokens_output=100)
        return LLMResponse(message=message, usage=usage, model="fake-model-2026", finish_reason=reason)

    async def complete_stream(self, messages, tools, output_format):  # type: ignore[override]
        # The text comes in pieces cut anywhere, as providers send it.
        result = await self.complete(messages, tools, output_format)
        text = result.message.text()
        for start in range(0, len(text), 3):
            yield LLMChunk(content=text[start:start + 3], is_complete=False)
        yield LLMChunk(
            content="", is_complete=True, message=result.message, usage=result.usage,
            finish_reason=result.finish_reason,
        )


def weather_agent(client: FakeClient, *middlewares: CoreMiddleware, ask: bool = False) -> Agent:
    """An agent with the weather tool; with ``ask`` the tool waits for approval."""
    mode = ToolApprovalMode.ASK_APPROVED if ask else ToolApprovalMode.AUTO_APPROVED
    tool = FunctionAsTool(get_weather, approval_mode=mode)
    return Agent("weather", "Tells the weather.", "Be brief.", client, toolset=[tool],
                 middlewares=list(middlewares))
