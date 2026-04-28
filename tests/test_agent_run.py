from __future__ import annotations

import typing as t

import pytest

from max_ai.base.agent import Agent
from max_ai.base.clients import CoreChatCompletionClient
from max_ai.core.messages import AssistantMessage
from max_ai.core.models import ModelConfig
from max_ai.types.completions import ChatCompletionResult, Usage


class RunClient(CoreChatCompletionClient):
    def __init__(self) -> None:
        super().__init__(model="run-client", config=ModelConfig())

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        return Usage()

    def format_messages(self, ctx, prompts):
        return list(ctx.messages)

    def build_api_messages(self, messages):
        return []

    def build_tool_schema(self, tools):
        return []

    async def complete(self, messages, tools, output_format, **kwargs):
        return ChatCompletionResult(
            message=AssistantMessage(source="run-client", content="done"),
            usage=Usage(llm_calls=1, attempts_to_call_api=1, tokens_input=1),
            model="run-client",
            finish_reason="stop",
        )

    async def stream(self, messages, tools, output_format, **kwargs):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_core_agent_run_returns_response():
    agent = Agent(
        name="runner",
        description="test agent",
        instructions="be concise",
        client=RunClient(),
    )

    response = await agent.run("hello")

    assert response.finish_reason == "stop"
    assert response.final_text == "done"
    assert response.context is not None
    assert response.context.messages[-1].text() == "done"
