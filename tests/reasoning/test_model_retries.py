"""The harness calls the model again on passing errors, never twice into the same answer."""

from __future__ import annotations

from contextlib import nullcontext

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.event_type import ErrorEvent, ModelRetryEvent, ModelStreamChunkEvent
from max_ai.core.messages import AssistantMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.core.retry import RetryPolicy
from max_ai.errors.client import ClientError
from max_ai.types.completions import ChatCompletionChunk, ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(RetryPolicy, "delay", lambda self, attempt, retry_after=None: 0.0)


class FlakyLLM:
    """Fails with ``errors`` first, one per call; ``started`` streams before failing."""

    model = "fake"

    def __init__(self, errors: list[ClientError], started: str = ""):
        self.config = ModelConfig()
        self.generation_options: dict = {}
        self.errors = list(errors)
        self.started = started
        self.calls = 0

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.calls += 1
        error = self.errors.pop(0) if self.errors else None
        if not stream:
            if error:
                raise error
            return ChatCompletionResult(message=AssistantMessage(source="llm", content="hola"),
                                        usage=Usage(), model="fake", finish_reason="stop")

        async def chunks():
            if self.started and error:
                yield ChatCompletionChunk(content=self.started, is_complete=False)
            if error:
                raise error
            yield ChatCompletionChunk(content="hola", is_complete=False)
            yield ChatCompletionChunk(content="", is_complete=True, finish_reason="stop")

        return chunks()


async def _events(llm: FlakyLLM, tmp_path) -> list:
    events = []
    async with Agent(name="a", description="d", instructions="i", client=llm,
                     workspace=LocalWorkspace(root=tmp_path)) as agent:
        with pytest.raises(ClientError) if llm.started else nullcontext():
            async for event in agent.run_stream_events(
                "hi", run_context=RunContext(user_id="u"), stream_tokens=True
            ):
                events.append(event)
    return events


async def test_rate_limit_is_retried_and_shown(tmp_path):
    llm = FlakyLLM([ClientError.rate_limit_exceeded(7), ClientError.api_error("x", 503)])
    events = await _events(llm, tmp_path)
    retries = [e for e in events if isinstance(e, ModelRetryEvent)]
    assert [(r.attempt, r.reason) for r in retries] == [(1, "rate_limit"), (2, "api_error")]
    assert llm.calls == 3
    assert "".join(e.chunk for e in events if isinstance(e, ModelStreamChunkEvent)) == "hola"


async def test_a_bad_request_is_not_retried(tmp_path):
    llm = FlakyLLM([ClientError.api_error("x", 400, "bad tool schema")])
    async with Agent(name="a", description="d", instructions="i", client=llm,
                     workspace=LocalWorkspace(root=tmp_path)) as agent:
        with pytest.raises(ClientError):
            await agent.run("hi", run_context=RunContext(user_id="u"))
    assert llm.calls == 1


async def test_a_stream_cut_after_text_is_not_streamed_twice(tmp_path):
    llm = FlakyLLM([ClientError.stream_interrupted()], started="Hol")
    events = await _events(llm, tmp_path)
    assert llm.calls == 1
    assert [e.chunk for e in events if isinstance(e, ModelStreamChunkEvent)] == ["Hol"]
    assert "after the answer started" in next(e for e in events if isinstance(e, ErrorEvent)).error_message


def test_client_errors_know_whether_calling_again_may_work():
    assert ClientError.rate_limit_exceeded(3).transient
    assert ClientError.rate_limit_exceeded(3).retry_after == 3
    assert ClientError.api_error("x", 502).transient
    assert ClientError.api_error("x").transient  # no status: a connection error
    assert not ClientError.api_error("x", 404).transient
    assert not ClientError.authentication_failed().transient
