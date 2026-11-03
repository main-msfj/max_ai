"""Steering reaches real models and doesn't derail them (#26).

A guard steers once after the first tool round: "end with PINEAPPLE". It
goes as a ``[harness]`` user message; a late system message made Gemma drop
the user's question. A model passes when it answers the question (22°C)
and obeys. Live, so off by default; each provider runs when its key is set:

    MAXAI_LIVE=1 uv run pytest -v -s -m integration tests/integration/test_steering_live.py
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.reasoning.guards import LoopGuard
from max_ai.capabilities.reasoning.react import ReactLoop
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.core.event_type import ModelCallEvent, ModelRetryEvent
from max_ai.core.messages import AssistantMessage
from max_ai.types.agent_response import AgentResponse
from max_ai.types.tools import ToolApprovalMode

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not os.getenv("MAXAI_LIVE"), reason="Set MAXAI_LIVE=1 to call real models",
)]

# One per family; the paid ones cost cents for the whole set.
MODELS = [
    ("openai", "gpt-4o-mini"),
    ("openrouter", "nvidia/nemotron-3.5-lightning:free"),
    ("openrouter", "anthropic/claude-haiku-5.5"),
    ("openrouter", "google/gemma-4-26b-a4b-it"),
    ("openrouter", "mistralai/mistral-small-3.2-24b-instruct"),
    ("openrouter", "qwen/qwen3-32b"),
]
CLIENTS = {
    "openai": ("OPENAI_API_KEY", OpenAIChatCompletionClient),
    "openrouter": ("OPENROUTER_API_KEY", OpenRouterChatCompletionClient),
}
WORD = "PINEAPPLE"
# Providers queue and rate-limit: a model that takes longer fails, not hangs.
LIMIT_SECONDS = 240


class SteerOnce(LoopGuard):
    """Steers after the first tool round only."""

    def after_tool_round(self, ctx, state, guard_ctx):
        rounds = sum(isinstance(m, AssistantMessage) and bool(m.tool_calls) for m in ctx.messages)
        return f"End your final answer with the word {WORD}." if rounds == 1 else None


def get_weather(city: str) -> str:
    """Return the current weather in the given city."""
    return f"Sunny, 22°C in {city}"


@pytest.mark.parametrize(("provider", "model"), MODELS)
async def test_the_model_follows_steering(provider, model):
    key, client_cls = CLIENTS[provider]
    if not os.getenv(key):
        pytest.skip(f"{key} not set")
    middlewares = []
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from max_ai.capabilities.middleware import TracingMiddleware, configure_langfuse
        configure_langfuse()
        middlewares = [TracingMiddleware()]
    agent = Agent(
        name="steering_probe", description="Reports the weather.",
        instructions="Use get_weather for weather questions, then answer in one sentence.",
        client=client_cls(model=model, max_tokens=4_000),
        toolset=[FunctionAsTool(func=get_weather, name="get_weather",
                                description="Get the current weather for a city.",
                                approval_mode=ToolApprovalMode.AUTO_APPROVED)],
        reasoning=ReactLoop(guards=[SteerOnce()]),
        middlewares=middlewares,
    )
    started = time.monotonic()

    def say(text):
        print(f"  {time.monotonic() - started:5.0f}s {text}", flush=True)

    async def run():
        async for item in agent.run_stream_events(task="What is the weather in Tokyo?"):
            if isinstance(item, ModelCallEvent):
                say("calling the model")
            elif isinstance(item, ModelRetryEvent):
                say(f"retry in {item.delay:.0f}s ({item.reason}, try {item.attempt + 1}/{item.max_attempts})")
            elif isinstance(item, AgentResponse):
                return item
        raise RuntimeError("no response")

    print(flush=True)
    async with agent:
        response = await asyncio.wait_for(run(), LIMIT_SECONDS)
    answer = next((m.text() for m in reversed(response.context.messages)
                   if isinstance(m, AssistantMessage) and m.text().strip()), "")
    print(f"\n{model} {response.finish_reason}: {answer[-120:]!r}")
    assert response.finish_reason == "stop", response.finish_reason
    assert "22" in answer, "lost the user's question"
    assert WORD in answer.upper(), "ignored the steering"
