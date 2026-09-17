"""Tests for the web UI's server-side display transcript.

The agent is stateless by design: compaction evicts messages from
``ctx.messages`` and the model works from a summary. Whatever the user
keeps seeing is the UI layer's responsibility — the server owns an
append-only display log per session/agent (``_display_messages``).
"""

from __future__ import annotations

from max_ai.core.messages import AssistantMessage, UserMessage
from max_ai.types.run_context import RunContext
from max_ai.ui.server import _display_messages


def make_session() -> dict:
    return {"contexts": {}, "active_tokens": {}, "display": {}}


def test_display_log_keeps_messages_evicted_by_compaction():
    """A message compaction removes from ctx.messages stays visible."""
    session = make_session()
    old = UserMessage(source="user", content="old question")
    reply = AssistantMessage(source="a", content="old answer")
    recent = UserMessage(source="user", content="new question")
    ctx = RunContext(messages=[old, reply, recent])

    first = _display_messages(session, "a", ctx)
    assert [m["content"] for m in first] == [
        "old question", "old answer", "new question",
    ]

    # Compaction evicts the two oldest messages from the live context.
    ctx.messages[:] = [recent]
    second = _display_messages(session, "a", ctx)

    assert [m["content"] for m in second] == [
        "old question", "old answer", "new question",
    ]


def test_display_log_upserts_in_place_on_interim_flip():
    """A promoted answer (interim flipped back to False) updates its
    existing entry instead of duplicating — same created_at, same key."""
    session = make_session()
    draft = AssistantMessage(source="a", content="final answer", interim=True)
    ctx = RunContext(messages=[draft])

    _display_messages(session, "a", ctx)
    ctx.messages[0] = draft.model_copy(update={"interim": False})
    result = _display_messages(session, "a", ctx)

    assert len(result) == 1
    assert result[0]["content"] == "final answer"
    assert result[0]["interim"] is False


def test_display_log_is_per_agent():
    session = make_session()
    ctx_a = RunContext(messages=[UserMessage(source="user", content="for A")])
    ctx_b = RunContext(messages=[UserMessage(source="user", content="for B")])

    _display_messages(session, "agent_a", ctx_a)
    result_b = _display_messages(session, "agent_b", ctx_b)

    assert [m["content"] for m in result_b] == ["for B"]


def test_display_without_session_falls_back_to_context():
    """The legacy no-session path serializes the live context as before."""
    ctx = RunContext(messages=[UserMessage(source="user", content="hi")])
    assert [m["content"] for m in _display_messages(None, "a", ctx)] == ["hi"]
