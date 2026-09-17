"""Tests for the terminal REPL (max_ai.cli.repl).

Elicitation is durable state: a question ends the stream segment with
``finish_reason='input_needed'`` and an ``INPUT_NEEDED`` record on the
context. The REPL prompts between segments, applies the answer via
``tool_state.apply_user_answer``, and resumes — the same shape as the
approval flow. No polling, no background task.
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


def _response(
    finish_reason: str = "stop", ctx: RunContext | None = None
) -> AgentResponse:
    return AgentResponse(
        context=ctx if ctx is not None else RunContext(),
        source="fake",
        usage=Usage(),
        finish_reason=finish_reason,
    )


def _question_record(
    record_id: str, question: str, options: list[str] | None = None
) -> ToolCallRecord:
    record = ToolCallRecord(
        id=record_id,
        tool_name="ask_user",
        parameters={"question": question, "options": options},
        session_id="s",
    )
    record.await_user_input(question, options)
    return record


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


class FakeAskingAgent:
    """Pauses with an INPUT_NEEDED record; the resumed segment reads the
    applied answer off the context — exactly how the durable flow works."""

    name = "fake"

    def __init__(self) -> None:
        self._ctx = RunContext()
        self._record = _question_record("call_q", "Which format?", ["JSON", "CSV"])
        self._ctx.tool_state.add(self._record)
        self.answered: str | None = None

    def run_stream_events(self, task=None, run_context=None, stream_tokens=False, **kw):
        async def _gen():
            yield _chunk("let me check… ")
            yield UserInputRequestEvent(
                source="fake",
                question="Which format?",
                options=["JSON", "CSV"],
                tool_call_id="call_q",
            )
            yield AgentResponse(
                context=self._ctx, source="fake", usage=Usage(),
                finish_reason="input_needed",
            )
        return _gen()

    def resume_stream_events(self, run_context=None, stream_tokens=False, **kw):
        async def _gen():
            record = self._ctx.tool_state.records["call_q"]
            assert record.user_answer is not None, "resumed without an answer"
            self.answered = record.user_answer
            yield _chunk(f"using {record.user_answer}")
            yield _response("stop")
        return _gen()


class FakeTwoQuestionAgent:
    """Asks TWO questions across two pauses in one turn: the first segment
    pauses on question 1; the resumed segment pauses on question 2; the
    second resume finishes. Each question must be prompted exactly once."""

    name = "fake"

    def __init__(self) -> None:
        self._ctx = RunContext()
        self._q1 = _question_record("call_q1", "Which email?", ["a@x.com", "b@x.com"])
        self._ctx.tool_state.add(self._q1)
        self.answers: list[str] = []

    def run_stream_events(self, task=None, run_context=None, stream_tokens=False, **kw):
        async def _gen():
            yield _chunk("ok… ")
            yield AgentResponse(
                context=self._ctx, source="fake", usage=Usage(),
                finish_reason="input_needed",
            )
        return _gen()

    def resume_stream_events(self, run_context=None, stream_tokens=False, **kw):
        async def _gen():
            if len(self.answers) == 0:
                self.answers.append(self._ctx.tool_state.records["call_q1"].user_answer)
                q2 = _question_record("call_q2", "Which format?", ["short", "long"])
                self._ctx.tool_state.add(q2)
                yield AgentResponse(
                    context=self._ctx, source="fake", usage=Usage(),
                    finish_reason="input_needed",
                )
                return
            self.answers.append(self._ctx.tool_state.records["call_q2"].user_answer)
            yield _chunk(f"sent to {self.answers[0]} as {self.answers[1]}")
            yield _response("stop")
        return _gen()


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
async def test_two_questions_each_answered_once(monkeypatch):
    """Two questions across two pauses must each be prompted exactly once.
    Answers are fed in order and mapped by option number."""
    agent = FakeTwoQuestionAgent()
    renderer = _renderer()

    answers = iter(["1", "2"])  # email -> a@x.com (opt 1), format -> long (opt 2)

    async def fake_ainput(_renderer, _prompt):
        return next(answers)

    monkeypatch.setattr("max_ai.cli.repl._ainput", fake_ainput)

    await asyncio.wait_for(
        _run_one_turn(agent, "send jokes", None, renderer), timeout=5.0
    )

    # Exactly two answers consumed (no duplicate prompting), mapped by option.
    assert agent.answers == ["a@x.com", "long"]
    out = renderer.console.export_text()
    assert "sent to a@x.com as long" in out


@pytest.mark.asyncio
async def test_question_pause_answer_resume(monkeypatch):
    """The REPL prompts on the input pause, applies the answer to the run's
    tool state, and resumes the turn to completion."""
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
