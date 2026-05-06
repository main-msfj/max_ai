"""End-to-end integration test against a real Ollama instance.

Requires:
  - Ollama reachable at $OLLAMA_HOST (default http://ollama:11434).
  - qwen3:4b-thinking-2507-q4_K_M pulled.

Run with: pytest -m integration tests/integration/test_ollama_agent.py
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest

from max_ai.base.agent import Agent
from max_ai.clients.ollama.client import OllamaChatCompletionClient
from max_ai.tools.function_as_tool import FunctionAsTool
from max_ai.core.models import ModelConfig
from max_ai.core.messages import AssistantMessage, ToolMessage, UserMessage
from max_ai.types.tools import ToolApprovalMode


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL = "qwen3:4b-instruct-2507-q4_K_M"


# -------- HEALTH CHECK -----------------------------------------------------------
def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason=f"Ollama not reachable at {OLLAMA_HOST}",
    ),
]


# -------- FIXTURES -----------------------------------------------------------
@pytest.fixture
def client() -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model=MODEL,
        host=OLLAMA_HOST,
        config=ModelConfig(
            supports_function_calling=True,
            supports_thinking=False,
        ),
        think=False,
    )


def _make_weather_tool() -> FunctionAsTool:
    def get_weather(city: str) -> str:
        """Return the current weather in the given city."""
        return f"Sunny, 22°C in {city}"

    return FunctionAsTool(
        func=get_weather,
        name="get_weather",
        description="Get the current weather for a city.",
        approval_mode=ToolApprovalMode.AUTO_APPROVED,
    )


# -------- TESTS -----------------------------------------------------------
@pytest.mark.asyncio
async def test_simple_no_tools(client):
    """Direct factual question with no tools available."""
    agent = Agent(
        name="math_agent",
        description="Answers math questions.",
        instructions="You are a concise math assistant. Reply with the number only.",
        client=client,
    )

    response = await agent.run(task="What is 2 + 2?")

    assert response.finish_reason == "stop"
    assert response.usage.llm_calls >= 1
    assert response.usage.tool_calls == 0

    messages = response.context.messages
    assert isinstance(messages[0], UserMessage)
    assert isinstance(messages[-1], AssistantMessage)
    assert "4" in messages[-1].text()


@pytest.mark.asyncio
async def test_tool_call_flow(client):
    """LLM → tool → LLM. Validates the full ReAct cycle end-to-end."""
    agent = Agent(
        name="weather_agent",
        description="Reports weather for cities.",
        instructions=(
            "You have a get_weather tool. When asked about weather, "
            "call the tool and report the result to the user."
        ),
        client=client,
        toolset=[_make_weather_tool()],
    )

    response = await agent.run(task="What is the weather in Tokyo?")

    assert response.finish_reason == "stop"
    assert response.usage.tool_calls >= 1
    assert response.usage.llm_calls >= 2  # at least one before tool, one after

    messages = response.context.messages
    assert isinstance(messages[0], UserMessage)

    # Must contain at least one tool call and its result.
    has_tool_call = any(
        isinstance(m, AssistantMessage) and m.tool_calls for m in messages
    )
    has_tool_result = any(isinstance(m, ToolMessage) for m in messages)
    assert has_tool_call, "no AssistantMessage with tool_calls in transcript"
    assert has_tool_result, "no ToolMessage in transcript"

    # Final message is an assistant text reply mentioning the result.
    final = messages[-1]
    assert isinstance(final, AssistantMessage)
    assert not final.tool_calls
    text = final.text().lower()
    assert "tokyo" in text or "22" in text or "sunny" in text

    # Tool state should have one consumed record.
    records = list(response.context.tool_state.records.values())
    assert len(records) >= 1
    assert all(r.is_consumed for r in records)
