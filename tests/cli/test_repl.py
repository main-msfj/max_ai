"""Tests for the terminal REPL (max_ai.cli.repl).

The riskiest part is the concurrent human-input handshake: the agent blocks
on a future before emitting the input event, so the REPL drives the stream in
a background task and answers from the foreground. These tests use a fake
agent that mimics that exact ordering to prove the handshake doesn't deadlock.
"""

from __future__ import annotations

import asyncio
import typing as t

import pytest
from rich.console import Console

from max_ai.cli.repl import _run_one_turn, _sanitize_orphan_tool_calls
from max_ai.cli.renderer import CliRenderer
from max_ai.core.event_type import (
    ModelStreamChunkEvent,
    UserInputRequestEvent,
    ReasoningCompleteEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
    ErrorEvent,
)
from max_ai.types.tool_call import ToolResult
from max_ai.core.messages import AssistantMessage, ToolMessage, ToolCall, UserMessage
from max_ai.types.agent_response import AgentResponse
from max_ai.types.run_context import RunContext
from max_ai.types.completions import Usage
from max_ai.types.tool_call import ToolCallRecord


def _chunk(text: str) -> ModelStreamChunkEvent:
    return ModelStreamChunkEvent(source="fake", chunk=text, is_final=False)


def _response(finish_reason: str = "stop") -> AgentResponse:
    return AgentResponse(
        context=RunContext(),
        source="fake",
        usage=Usage(),
        finish_reason=finish_reason,
    )


class FakePlainAgent:
    """Streams two text chunks then a final response. No questions."""

    name = "fake"

    def run_stream_events(self, task=None, run_context=None, stream_tokens=False, **kw):
        async def _gen():
            yield _chunk("hello ")
            yield _chunk("world")
            yield ReasoningCompleteEvent(source="fake", finish_reason="stop", total_iterations=1)
            yield _response("stop")
        return _gen()

    # No pending question ever.
    pending_user_input = None
    pending_user_input_options = None


class FakeAskingAgent:
    """Mimics the real blocking order: the stream sets a pending question and
    waits on a future *before* emitting UserInputRequestEvent — exactly what
    the real loop does. provide_user_input resolves the future."""

    name = "fake"

    def __init__(self) -> None:
        self._future: asyncio.Future[str] | None = None
        self._question: str | None = None
        self._options: list[str] | None = None
        self.answered: str | None = None

    def run_stream_events(self, task=None, run_context=None, stream_tokens=False, **kw):
        async def _gen():
            yield _chunk("let me check… ")
            # Pause: set the pending question and block, like the real tool.
            self._future = asyncio.get_running_loop().create_future()
            self._question = "Which format?"
            self._options = ["JSON", "CSV"]
            answer = await self._future
            self.answered = answer
            # Only now does the event surface (post-answer), as in production.
            yield UserInputRequestEvent(source="fake", question="Which format?", options=["JSON", "CSV"])
            yield _chunk(f"using {answer}")
            yield ReasoningCompleteEvent(source="fake", finish_reason="stop", total_iterations=2)
            yield _response("stop")
        return _gen()

    @property
    def pending_user_input(self) -> str | None:
        if self._future is None or self._future.done():
            return None
        return self._question

    @property
    def pending_user_input_options(self) -> list[str] | None:
        if self._future is None or self._future.done():
            return None
        return self._options

    def provide_user_input(self, answer: str) -> None:
        assert self._future is not None and not self._future.done()
        self._future.set_result(answer)


def _renderer() -> CliRenderer:
    # record=True keeps output off the real terminal during tests.
    return CliRenderer(console=Console(record=True, force_terminal=False))


@pytest.mark.asyncio
async def test_plain_turn_streams_and_returns_context():
    agent = FakePlainAgent()
    renderer = _renderer()

    ctx = await _run_one_turn(agent, "hi", None, renderer)

    assert isinstance(ctx, RunContext)
    out = renderer.console.export_text()
    assert "hello world" in out


@pytest.mark.asyncio
async def test_human_input_handshake_does_not_deadlock(monkeypatch):
    """The REPL must answer the agent's question and let the turn finish.

    Patches the input read so no real stdin is needed; asserts the answer
    reached the agent and the post-answer text streamed."""
    agent = FakeAskingAgent()
    renderer = _renderer()

    async def fake_ainput(_renderer, _prompt):
        return "JSON"

    monkeypatch.setattr("max_ai.cli.repl._ainput", fake_ainput)

    ctx = await asyncio.wait_for(
        _run_one_turn(agent, "fetch data", None, renderer), timeout=5.0
    )

    assert agent.answered == "JSON"
    assert isinstance(ctx, RunContext)
    out = renderer.console.export_text()
    assert "using JSON" in out


# -------- TOOL SPINNER --------------------------------------------------------
def test_tool_spinner_starts_on_call_and_stops_on_response():
    """A ToolCallEvent arms a 'running…' spinner; the response stops it, then
    re-arms the thinking spinner (the model runs next)."""
    renderer = _renderer()
    renderer.handle(
        ToolCallEvent(source="a", tool_name="web_search", parameters={"q": "x"}, tool_call_id="c1")
    )
    assert renderer._spinner is not None  # spinner running during the tool

    renderer.handle(
        ToolCallResponseEvent(
            source="a", tool_call_id="c1",
            tool_result=ToolResult(tool_call_id="c1", success=True, result="ok"),
        )
    )
    # Response re-arms the thinking spinner; a terminal event clears it.
    assert renderer._spinner is not None
    renderer.handle(ReasoningCompleteEvent(source="a", finish_reason="stop", total_iterations=1))
    assert renderer._spinner is None


# -------- ORPHAN TOOL-CALL SANITIZER ------------------------------------------
def test_sanitize_drops_orphan_assistant_tool_call():
    """An assistant tool_call with no matching ToolMessage is removed; the rest
    of the conversation is preserved."""
    ctx = RunContext()
    ctx.messages.extend([
        UserMessage(source="user", content="search NVDA"),
        AssistantMessage(
            source="a", content="",
            tool_calls=[ToolCall(id="call_x", tool_name="web_search", parameters={})],
        ),
        # No ToolMessage for call_x -> orphan.
    ])

    _sanitize_orphan_tool_calls(ctx)

    assert len(ctx.messages) == 1
    assert isinstance(ctx.messages[0], UserMessage)


def test_sanitize_keeps_answered_tool_call():
    """An assistant tool_call WITH a matching ToolMessage is kept intact."""
    ctx = RunContext()
    ctx.messages.extend([
        UserMessage(source="user", content="search NVDA"),
        AssistantMessage(
            source="a", content="",
            tool_calls=[ToolCall(id="call_x", tool_name="web_search", parameters={})],
        ),
        ToolMessage(source="t", tool_call_id="call_x", tool_name="web_search",
                    success=True, content="result"),
    ])

    before = len(ctx.messages)
    _sanitize_orphan_tool_calls(ctx)

    assert len(ctx.messages) == before  # nothing dropped


class FakeErroringAgent:
    """Streams a chunk, an ErrorEvent, then raises from the stream — exactly
    how a provider 400 surfaces (ErrorEvent emitted, then the exception)."""

    name = "fake"
    pending_user_input = None
    pending_user_input_options = None

    def run_stream_events(self, task=None, run_context=None, stream_tokens=False, **kw):
        async def _gen():
            yield _chunk("working… ")
            yield ErrorEvent(
                source="fake", error_message="Invalid request: 400 bad messages",
                error_type="ClientError", is_recoverable=False,
            )
            raise RuntimeError("Invalid request: 400 bad messages")
            yield  # unreachable; makes this an async generator
        return _gen()


@pytest.mark.asyncio
async def test_stream_error_does_not_crash_repl():
    """A failing turn must not propagate a raw traceback out of _run_one_turn —
    the error is rendered and the turn returns."""
    agent = FakeErroringAgent()
    renderer = _renderer()

    # Should NOT raise.
    ctx = await asyncio.wait_for(
        _run_one_turn(agent, "hi", None, renderer), timeout=5.0
    )

    out = renderer.console.export_text()
    assert "error" in out.lower()
    # No AgentResponse was produced, so the incoming ctx (None) is returned.
    assert ctx is None


# -------- APPROVAL FLOW -------------------------------------------------------
class FakeApprovalAgent:
    """First turn pauses needing approval for a tool; on resume it runs and
    finishes. Mirrors how an ASK_APPROVED MCP tool (e.g. Tavily) behaves."""

    name = "fake"
    pending_user_input = None
    pending_user_input_options = None

    def __init__(self) -> None:
        self._ctx = RunContext()
        self._record = ToolCallRecord(
            id="call_t", tool_name="web_search", parameters={"q": "NVDA"},
            session_id="s",
        )
        self._ctx.tool_state.add(self._record)  # defaults to PENDING_APPROVAL

    @property
    def applied(self) -> list[tuple[str, bool]]:
        """Decisions actually applied to the pending record (read from state)."""
        rec = self._ctx.tool_state.records["call_t"]
        if rec.is_pending_approval:
            return []
        return [("call_t", rec.is_approved)]

    def run_stream_events(self, task=None, run_context=None, stream_tokens=False, **kw):
        async def _gen():
            yield _chunk("let me search… ")
            yield AgentResponse(
                context=self._ctx, source="fake", usage=Usage(),
                finish_reason="approval_needed",
            )
        return _gen()

    def resume_stream_events(self, run_context=None, stream_tokens=False, **kw):
        async def _gen():
            yield _chunk("found it")
            yield _response("stop")
        return _gen()


@pytest.mark.asyncio
async def test_approval_flow_approves_and_resumes(monkeypatch):
    """A turn that pauses for approval: the CLI prompts, approves, and resumes
    so the tool runs and the turn completes."""
    agent = FakeApprovalAgent()
    renderer = _renderer()

    async def fake_ainput(_renderer, _prompt):
        return "y"

    monkeypatch.setattr("max_ai.cli.repl._ainput", fake_ainput)

    await asyncio.wait_for(
        _run_one_turn(agent, "search NVDA", None, renderer), timeout=5.0
    )

    assert agent.applied == [("call_t", True)]
    out = renderer.console.export_text()
    assert "found it" in out  # resumed stream produced the result


@pytest.mark.asyncio
async def test_approval_flow_reject_stops(monkeypatch):
    """Rejecting the only pending tool stops the turn without resuming."""
    agent = FakeApprovalAgent()
    renderer = _renderer()

    async def fake_ainput(_renderer, _prompt):
        return "n"

    monkeypatch.setattr("max_ai.cli.repl._ainput", fake_ainput)

    await asyncio.wait_for(
        _run_one_turn(agent, "search NVDA", None, renderer), timeout=5.0
    )

    assert agent.applied == [("call_t", False)]
    out = renderer.console.export_text()
    assert "found it" not in out  # never resumed


@pytest.mark.asyncio
async def test_human_input_numeric_option_maps_to_choice(monkeypatch):
    """Typing '2' selects the 2nd option (CSV) rather than sending '2'."""
    agent = FakeAskingAgent()
    renderer = _renderer()

    async def fake_ainput(_renderer, _prompt):
        return "2"

    monkeypatch.setattr("max_ai.cli.repl._ainput", fake_ainput)

    await asyncio.wait_for(
        _run_one_turn(agent, "fetch data", None, renderer), timeout=5.0
    )

    assert agent.answered == "CSV"
