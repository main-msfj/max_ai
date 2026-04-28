"""Agent tests with fake client; ToolExecutor real with empty middlewares."""

from __future__ import annotations

import asyncio
import pytest

from max_ai.base.agent import Agent
from max_ai.base.clients import CoreChatCompletionClient
from max_ai.core.messages import AssistantMessage, UserMessage
from max_ai.core.event_type import (
    CoreEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ReasoningCompleteEvent,
    ReasoningIterationEvent,
    ErrorEvent,
)
from max_ai.core.models import ModelConfig
from max_ai.types.agent_response import AgentResponse
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.errors.client import ClientError


# -------- FAKE CLIENT -----------------------------------------------------------
class FakeChatClient(CoreChatCompletionClient):
    """Bypasses run() pipeline; returns scripted results from a queue."""

    def __init__(self, results=None, raise_first=None):
        self.model = "fake"
        self.config = ModelConfig()
        self._results: list[ChatCompletionResult] = list(results or [])
        self._raise_first = raise_first

    async def run(self, ctx, prompts, tools=None, output_format=None,
                  stream=False, **kwargs):
        if self._raise_first is not None:
            err = self._raise_first
            self._raise_first = None
            raise err
        return self._results.pop(0)

    def format_messages(self, ctx, prompts): return []
    def build_api_messages(self, messages): return []
    def build_tool_schema(self, tools): return []
    def normalize_usage_stats(self, usage): return Usage()
    async def complete(self, *a, **kw): raise NotImplementedError
    async def stream(self, *a, **kw):
        raise NotImplementedError
        yield


# -------- HELPERS -----------------------------------------------------------
def make_result(content="", tool_calls=None, finish_reason="stop"):
    return ChatCompletionResult(
        message=AssistantMessage(
            source="fake",
            content=content,
            tool_calls=tool_calls or [],
        ),
        usage=Usage(llm_calls=1, attempts_to_call_api=1),
        model="fake",
        finish_reason=finish_reason,
    )


def make_agent(client, **kwargs) -> Agent:
    return Agent(
        name="test_agent",
        description="Agent for unit tests.",
        instructions="Be concise.",
        client=client,
        **kwargs,
    )


# -------- TESTS -----------------------------------------------------------
@pytest.mark.asyncio
async def test_run_happy_path():
    """run() returns AgentResponse with finish_reason=stop and usable messages."""
    client = FakeChatClient(results=[make_result(content="hello")])
    agent = make_agent(client)

    response = await agent.run(task="hi")

    assert isinstance(response, AgentResponse)
    assert response.finish_reason == "stop"
    assert response.source == "test_agent"
    assert response.usage.llm_calls >= 1
    assert response.final_text == "hello"
    assert isinstance(response.context.messages[0], UserMessage)


@pytest.mark.asyncio
async def test_run_stream_events_terminates_with_response():
    """run_stream_events yields CoreEvent items then exactly one AgentResponse last."""
    client = FakeChatClient(results=[make_result(content="hi")])
    agent = make_agent(client)

    items = [item async for item in agent.run_stream_events(task="hi")]

    assert len(items) >= 2  # at least one event + the response
    assert isinstance(items[-1], AgentResponse)
    assert all(isinstance(x, CoreEvent) for x in items[:-1])
    response_count = sum(1 for x in items if isinstance(x, AgentResponse))
    assert response_count == 1


@pytest.mark.asyncio
async def test_model_call_event_emitted_before_response():
    """ModelCallEvent precedes ModelResponseEvent in the stream."""
    client = FakeChatClient(results=[make_result(content="hi")])
    agent = make_agent(client)

    events = [item async for item in agent.run_stream_events(task="hi")
              if isinstance(item, CoreEvent)]

    types = [type(e) for e in events]
    assert ModelCallEvent in types
    assert ModelResponseEvent in types
    assert types.index(ModelCallEvent) < types.index(ModelResponseEvent)

    # Sanity: iteration event opens the turn, complete event closes it.
    assert types.index(ReasoningIterationEvent) < types.index(ModelCallEvent)
    assert ReasoningCompleteEvent in types


@pytest.mark.asyncio
async def test_run_stream_yields_only_text():
    """run_stream filters to assistant text chunks; no events, no response."""
    # Even with stream_tokens=True, our fake client returns a non-streaming
    # result. The reasoning loop in stream mode then synthesizes chunks
    # only if the underlying call yields ChatCompletionChunk — our fake
    # doesn't, so we expect zero text chunks. That's still a valid signal:
    # run_stream must not leak non-string items.
    client = FakeChatClient(results=[make_result(content="hi")])
    agent = make_agent(client)

    items = [item async for item in agent.run_stream(task="hi")]

    # Whatever it yields, every item must be a str.
    assert all(isinstance(x, str) for x in items)


@pytest.mark.asyncio
async def test_error_yields_error_event_and_response_marked_error():
    """Non-transient ClientError → ErrorEvent yielded, response.finish_reason=error."""
    client = FakeChatClient(
        results=[],
        raise_first=ClientError.authentication_failed("bad key"),
    )
    agent = make_agent(client, config=_no_retry_config())

    items = [item async for item in agent.run_stream_events(task="hi")]

    assert any(isinstance(x, ErrorEvent) for x in items)
    response = items[-1]
    assert isinstance(response, AgentResponse)
    assert response.finish_reason == "error"


@pytest.mark.asyncio
async def test_cancellation_propagates_without_response():
    """Cancelled token → CancelledError raised, no AgentResponse yielded."""
    class Cancelled:
        def is_cancelled(self): return True
        def link_future(self, fut): pass

    client = FakeChatClient(results=[make_result(content="never")])
    agent = make_agent(client)

    items: list = []
    with pytest.raises(asyncio.CancelledError):
        async for item in agent.run_stream_events(
            task="hi", cancellation_token=Cancelled(),
        ):
            items.append(item)

    assert not any(isinstance(x, AgentResponse) for x in items)


# -------- INTERNAL -----------------------------------------------------------
def _no_retry_config():
    """AgentConfig with retries disabled so error tests run fast."""
    from max_ai.core.models import AgentConfig
    return AgentConfig(max_connection_retries=0)