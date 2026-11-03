"""A stream that breaks and is retried tells consumers to drop what they showed."""

from types import SimpleNamespace

import pytest

from max_ai.agents.agent import STREAM_RESTARTED, Agent
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.event_type import ModelRetryEvent, ModelStreamChunkEvent
from max_ai.core.retry import RetryPolicy
from max_ai.errors.client import ClientError
from max_ai.types.completions import ChatCompletionChunk, Usage
from max_ai.types.run_context import RunContext


class FlakyStream:
    """The first stream drops after ``sent`` words; the retry sends the full answer."""

    model = "fake"
    config = SimpleNamespace(tokenizer_base="o200k_base")

    def __init__(self, sent: int):
        self.sent = sent
        self.calls = 0

    async def run(self, ctx, prompts, tools=None, stream=False, **kwargs):
        self.calls += 1
        first = self.calls == 1

        async def chunks():
            for word in ["Hola, ", "el ", "resultado "][: self.sent if first else 3]:
                yield ChatCompletionChunk(content=word, is_complete=False)
            if first:
                raise ClientError.stream_interrupted("connection reset")
            yield ChatCompletionChunk(content="es 42.", is_complete=False)
            yield ChatCompletionChunk(content="", is_complete=True, usage=Usage(), finish_reason="stop")

        return chunks()


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(RetryPolicy, "delay", lambda self, attempt, retry_after=None, **_: 0.0)


def make_agent(tmp_path, client):
    return Agent(name="t", description="d", instructions="i", client=client,
                 workspace=LocalWorkspace(root=tmp_path))


async def stream_events(tmp_path, client):
    ctx = RunContext(user_id="u", session_id="s")
    async with make_agent(tmp_path, client) as agent:
        events = [e async for e in agent.run_stream_events("¿resultado?", run_context=ctx,
                                                            stream_tokens=True)]
    return events, ctx


async def test_a_broken_stream_is_reset_before_the_retry(tmp_path):
    events, ctx = await stream_events(tmp_path, FlakyStream(sent=3))

    shown = []
    for event in events:
        if isinstance(event, ModelRetryEvent):
            assert event.attempt == 1 and event.reason == "stream_interrupted"
            assert event.discard_partial
            shown.clear()
        elif isinstance(event, ModelStreamChunkEvent):
            shown.append(event.chunk)
    assert "".join(shown) == "Hola, el resultado es 42."
    assert ctx.messages[-1].text() == "Hola, el resultado es 42."


async def test_nothing_to_discard_when_nothing_was_streamed(tmp_path):
    events, ctx = await stream_events(tmp_path, FlakyStream(sent=0))

    [retry] = [e for e in events if isinstance(e, ModelRetryEvent)]
    assert not retry.discard_partial
    assert ctx.messages[-1].text() == "Hola, el resultado es 42."


async def test_run_stream_marks_the_restart(tmp_path):
    async with make_agent(tmp_path, FlakyStream(sent=3)) as agent:
        text = "".join([c async for c in agent.run_stream(
            "¿resultado?", run_context=RunContext(user_id="u", session_id="s"))])

    assert text == f"Hola, el resultado {STREAM_RESTARTED}Hola, el resultado es 42."
