from __future__ import annotations

import typing as t

import pytest

from max_ai.base.agent import Agent
from max_ai.base.clients import CoreChatCompletionClient
from max_ai.base.compaction import CompactionResult, CoreCompaction
from max_ai.core.messages import AssistantMessage
from max_ai.core.messages import UserMessage
from max_ai.core.models import ModelConfig
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext


class RunClient(CoreChatCompletionClient):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(model="run-client", config=config or ModelConfig())

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


class RecordingCompaction(CoreCompaction):
    called: bool = False

    async def compact(self, *, ctx, prompts, max_context_tokens):
        self.called = True
        return CompactionResult(
            changed=False,
            recent_messages=list(ctx.messages),
            max_context_tokens=max_context_tokens,
        )


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


@pytest.mark.asyncio
async def test_agent_applies_default_compaction_when_model_context_window_is_known():
    agent = Agent(
        name="runner",
        description="test agent",
        instructions="be concise",
        client=RunClient(config=ModelConfig(max_context_window=32_000)),
    )
    ctx = RunContext(
        messages=[
            UserMessage(source="user", content="old", token_count=2_000),
            UserMessage(source="user", content="recent", token_count=1),
        ]
    )

    response = await agent.run(run_context=ctx)

    assert response.context is not None
    assert [message.text() for message in response.context.messages] == [
        "recent",
        "done",
    ]


@pytest.mark.asyncio
async def test_agent_uses_custom_compaction_strategy():
    compaction = RecordingCompaction()
    agent = Agent(
        name="runner",
        description="test agent",
        instructions="be concise",
        client=RunClient(config=ModelConfig(max_context_window=32_000)),
        compaction=compaction,
    )

    await agent.run("hello")

    assert compaction.called is True
