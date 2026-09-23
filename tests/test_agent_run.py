from __future__ import annotations

import typing as t

import pytest

from max_ai.base.agent import Agent
from max_ai.base.clients import CoreChatCompletionClient
from max_ai.base.compaction import CompactionResult, CoreCompaction
from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.capabilities.stacks.memory_layer import MemoryLayer
from max_ai.core.event_type import CompactionEvent
from max_ai.core.messages import AssistantMessage, UserMessage
from max_ai.core.model.llm import ModelConfig
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

    async def compact(self, *, ctx, prompts, max_context_tokens, client):
        self.called = True
        return CompactionResult(
            changed=False,
            recent_messages=list(ctx.messages),
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
async def test_agent_applies_default_compaction_when_model_context_window_is_known(
    monkeypatch,
):
    # This test exercises eviction with a 2-message transcript; the
    # production min-keep floor (3 atomic groups) would keep both.
    from max_ai.config import setting

    monkeypatch.setattr(setting, "compaction_min_keep_groups", 1)
    agent = Agent(
        name="runner",
        description="test agent",
        instructions="be concise",
        client=RunClient(config=ModelConfig(max_context_window=32_000)),
    )
    ctx = RunContext(
        messages=[
            # Must exceed the live-message threshold, which is now derived
            # from the agent's real rendered prompt size (small for this
            # bare test agent), so use a clearly oversized transcript.
            UserMessage(source="user", content="old", token_count=30_000),
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
async def test_final_message_skips_interim_and_toolcall_shells():
    """AgentResponse.final_message must surface the user-facing answer:
    not a guard-vetoed interim draft, not a text-less tool-call shell."""
    from max_ai.core.messages import ToolCall

    ctx = RunContext(
        messages=[
            UserMessage(source="user", content="hi"),
            AssistantMessage(source="a", content="draft answer", interim=True),
            AssistantMessage(source="a", content="real answer"),
            AssistantMessage(
                source="a",
                content="",
                tool_calls=[
                    ToolCall(id="c1", tool_name="update_plan", parameters={})
                ],
            ),
        ]
    )
    from max_ai.types.agent_response import AgentResponse

    response = AgentResponse(
        source="a", usage=Usage(), finish_reason="stop", context=ctx
    )
    assert response.final_text == "real answer"


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


@pytest.mark.asyncio
async def test_agent_stream_does_not_emit_compaction_events_below_live_threshold():
    compaction = RecordingCompaction()
    agent = Agent(
        name="runner",
        description="test agent",
        instructions="be concise",
        client=RunClient(config=ModelConfig(max_context_window=32_000)),
        compaction=compaction,
    )

    items = [item async for item in agent.run_stream_events("hello")]
    events = [item for item in items if isinstance(item, CompactionEvent)]

    assert events == []
    assert compaction.called is True


@pytest.mark.asyncio
async def test_agent_stream_emits_compaction_start_and_end_events_above_live_threshold():
    compaction = RecordingCompaction()
    agent = Agent(
        name="runner",
        description="test agent",
        instructions="be concise",
        client=RunClient(config=ModelConfig(max_context_window=100)),
        compaction=compaction,
    )
    ctx = RunContext(
        messages=[UserMessage(source="user", content="large live message", token_count=50)]
    )

    items = [item async for item in agent.run_stream_events(run_context=ctx)]
    events = [item for item in items if isinstance(item, CompactionEvent)]

    assert [event.phase for event in events] == ["start", "end"]
    assert events[0].total_token_count == 50
    assert events[-1].changed is False
    assert compaction.called is True


@pytest.mark.asyncio
async def test_agent_refreshes_memory_layer_before_each_run(tmp_path):
    memory = LocalMemoryRegistry(user_id="u1", base_path=tmp_path)
    agent = Agent(
        name="runner",
        description="test agent",
        instructions="be concise",
        client=RunClient(),
        memory=memory,
    )

    await agent.prepare()
    assert "No persistent memories" in agent.rendered_layers[MemoryLayer]

    async with memory:
        await memory.update_fact("user_context", "User is Marin and works on NASA.")

    await agent.run("hello")

    assert "User is Marin and works on NASA." in agent.rendered_layers[MemoryLayer]


class QuestionClient(RunClient):
    """Emits an ask_user call, then a plain answer."""

    def __init__(self) -> None:
        super().__init__()
        self.turn = 0

    async def complete(self, messages, tools, output_format, **kwargs):
        from max_ai.core.messages import ToolCall

        self.turn += 1
        if self.turn == 1:
            return ChatCompletionResult(
                message=AssistantMessage(
                    source="run-client",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="q1",
                            tool_name="ask_user",
                            parameters={"question": "Which one?"},
                        )
                    ],
                ),
                usage=Usage(llm_calls=1, attempts_to_call_api=1),
                model="run-client",
                finish_reason="tool_calls",
            )
        return await super().complete(messages, tools, output_format, **kwargs)


@pytest.mark.asyncio
async def test_agent_run_reports_input_needed_finish_reason():
    """A paused-for-input turn must surface finish_reason='input_needed',
    not be coerced to 'error' by the response whitelist."""
    agent = Agent(
        name="asker",
        description="test agent",
        instructions="be concise",
        client=QuestionClient(),
    )

    response = await agent.run("hello")

    assert response.finish_reason == "input_needed"
    assert response.needs_input is True
    assert len(response.pending_questions) == 1

    ctx = response.context
    ctx.tool_state.apply_user_answer(response.pending_questions[0].id, "that one")
    resumed = await agent.resume(run_context=ctx)
    assert resumed.finish_reason == "stop"
    assert resumed.final_text == "done"
