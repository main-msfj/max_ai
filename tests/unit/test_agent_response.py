"""Tests for AgentResponse."""

from __future__ import annotations

import pytest
import typing as t 

from max_ai.core.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from max_ai.types.agent_response import AgentResponse
from max_ai.types.completions import Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.chat_history import ChatHistory


# -------- HELPERS -----------------------------------------------------------
def make_usage(**overrides: t.Any) -> Usage:
    base = {"duration_ms": 1500, "llm_calls": 1, "tokens_input": 100, "tokens_output": 50}
    base.update(overrides)
    return Usage(**base)


def make_response(**overrides: t.Any) -> AgentResponse:
    base: dict[str, t.Any] = {
        "source": "test-agent",
        "usage": make_usage(),
        "finish_reason": "stop",
    }
    base.update(overrides)
    return AgentResponse(**base)


# -------- BASIC -----------------------------------------------------------
def test_minimal_construction():
    r = make_response()
    assert r.source == "test-agent"
    assert r.context is None
    assert r.finish_reason == "stop"
    assert r.created_at is not None


def test_is_frozen():
    r = make_response()
    with pytest.raises(Exception):  # pydantic ValidationError
        r.source = "other"  # type: ignore[misc]


# -------- MESSAGES (NO CONTEXT) -----------------------------------------------------------
def test_messages_empty_when_no_context():
    r = make_response()
    assert r.messages == []


def test_final_message_none_when_no_context():
    r = make_response()
    assert r.final_message is None
    assert r.final_text == ""


# -------- MESSAGES (WITH CONTEXT) -----------------------------------------------------------
def test_messages_combines_history_and_new():
    hist_msg = UserMessage(source="user", content="previous turn")
    new_msg = UserMessage(source="user", content="current turn")

    ctx = RunContext(
        messages=[new_msg],
        message_history=ChatHistory(message_history=[hist_msg]),
    )
    r = make_response(context=ctx)

    assert len(r.messages) == 2
    assert r.messages[0].text() == "previous turn"
    assert r.messages[1].text() == "current turn"


def test_messages_does_not_include_system():
    """System messages are synthetic and never part of persisted state."""
    user = UserMessage(source="user", content="hello")
    asst = AssistantMessage(source="llm", content="hi")
    ctx = RunContext(messages=[user, asst])
    r = make_response(context=ctx)

    # No SystemMessage was ever added; just user + assistant.
    assert len(r.messages) == 2
    assert all(not isinstance(m, SystemMessage) for m in r.messages)


# -------- FINAL_MESSAGE / FINAL_TEXT -----------------------------------------------------------
def test_final_message_returns_last_assistant():
    user = UserMessage(source="user", content="ask")
    asst1 = AssistantMessage(source="llm", content="first reply")
    asst2 = AssistantMessage(source="llm", content="final reply")
    ctx = RunContext(messages=[user, asst1, asst2])
    r = make_response(context=ctx)

    assert r.final_message is not None
    assert r.final_message.text() == "final reply"


def test_final_message_skips_non_assistant_at_end():
    """If the last message is a ToolMessage (e.g. run paused after tool),
    final_message should still find the last assistant message before it."""
    user = UserMessage(source="user", content="ask")
    asst = AssistantMessage(source="llm", content="calling tool")
    tool = ToolMessage.success_message(
        tool_call_id="t1",
        tool_name="search",
        content="result",
        source="agent",
    )
    ctx = RunContext(messages=[user, asst, tool])
    r = make_response(context=ctx)

    assert r.final_message is not None
    assert r.final_message.text() == "calling tool"


def test_final_message_none_when_no_assistant():
    user = UserMessage(source="user", content="hi")
    ctx = RunContext(messages=[user])
    r = make_response(context=ctx)

    assert r.final_message is None
    assert r.final_text == ""


def test_final_text_returns_assistant_content():
    asst = AssistantMessage(source="llm", content="the answer is 42")
    ctx = RunContext(messages=[asst])
    r = make_response(context=ctx)

    assert r.final_text == "the answer is 42"


# -------- APPROVAL FLOW -----------------------------------------------------------
def test_needs_approval_false_when_no_context():
    r = make_response()
    assert r.needs_approval is False
    assert r.pending_approvals == []


def test_needs_approval_false_when_no_pending_records():
    ctx = RunContext()
    r = make_response(context=ctx)
    assert r.needs_approval is False


def test_needs_approval_true_with_pending_record():
    ctx = RunContext()
    record = ToolCallRecord(tool_name="delete_user", parameters={"id": "u_1"})
    ctx.tool_state.add(record)

    r = make_response(context=ctx, finish_reason="approval_needed")
    assert r.needs_approval is True
    assert len(r.pending_approvals) == 1
    assert r.pending_approvals[0].tool_name == "delete_user"


def test_needs_approval_false_after_record_decided():
    ctx = RunContext()
    record = ToolCallRecord(tool_name="t", parameters={})
    ctx.tool_state.add(record)
    record.approve()  # no longer pending

    r = make_response(context=ctx)
    assert r.needs_approval is False
    assert r.pending_approvals == []


# -------- FINISH REASON -----------------------------------------------------------
def test_finish_reason_accepts_all_literal_values():
    """All Literal values are accepted."""
    for reason in [
        "stop", "max_iterations", "approval_needed",
        "tool_direct_return", "no_result", "error", "cancelled",
    ]:
        r = make_response(finish_reason=reason)
        assert r.finish_reason == reason


# -------- DUNDERS -----------------------------------------------------------
def test_str_contains_messages_and_usage_summary():
    user = UserMessage(source="user", content="hi")
    asst = AssistantMessage(source="llm", content="hello")
    ctx = RunContext(messages=[user, asst])
    r = make_response(context=ctx)

    s = str(r)
    assert "hi" in s
    assert "hello" in s
    assert "duration" in s
    assert "tokens" in s
    assert "stop" in s


def test_str_shows_approvals_when_pending():
    ctx = RunContext()
    record = ToolCallRecord(tool_name="t", parameters={})
    ctx.tool_state.add(record)

    r = make_response(context=ctx, finish_reason="approval_needed")
    s = str(r)
    assert "approval(s) needed" in s


def test_repr_is_concise():
    r = make_response()
    assert "AgentResponse" in repr(r)
    assert "test-agent" in repr(r)
    assert "stop" in repr(r)


# -------- SERIALIZATION -----------------------------------------------------------
def test_round_trip_with_context():
    user = UserMessage(source="user", content="hi")
    ctx = RunContext(messages=[user])
    r = make_response(context=ctx)

    dumped = r.model_dump()
    restored = AgentResponse.model_validate(dumped)

    assert restored.source == r.source
    assert restored.finish_reason == r.finish_reason
    assert len(restored.messages) == 1