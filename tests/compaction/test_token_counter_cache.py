"""Each message is tokenized once per counter, and thinking is never counted."""

from __future__ import annotations

import gc

import pytest

from max_ai.core.compaction import token_counter as module
from max_ai.core.compaction.token_counter import TokenCounter
from max_ai.core.messages import AssistantMessage, ToolCall, UserMessage


class CountingEncoding:
    """Stands in for tiktoken: one token per word, and counts its calls."""

    def __init__(self):
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return text.split()


@pytest.fixture
def encoding(monkeypatch):
    fake = CountingEncoding()
    monkeypatch.setattr(module, "_encoding", lambda name: fake)
    return fake


def test_a_message_is_tokenized_once(encoding):
    counter = TokenCounter()
    message = UserMessage(source="u", content="uno dos tres")
    first = counter.count_message(message)
    assert first == 3 + counter.message_overhead_tokens
    for _ in range(5):
        assert counter.count_message(message) == first
        assert counter.count_messages([message, message]) == 2 * first
    assert encoding.calls == 1


def test_the_cache_forgets_collected_messages(encoding):
    counter = TokenCounter()
    counter.count_message(UserMessage(source="u", content="temporal"))
    gc.collect()
    assert counter._counts == {}


def test_thinking_is_not_counted(encoding):
    counter = TokenCounter()
    call = ToolCall(id="c1", tool_name="search", parameters={"q": "x"})
    plain = AssistantMessage(source="a", content="ok", tool_calls=[call])
    thinking = AssistantMessage(source="a", content="ok", tool_calls=[call],
                                thinking="a very long private chain of thought " * 50)
    assert counter.count_message(thinking) == counter.count_message(plain)
    assert (counter.count_message(thinking.model_copy(update={"tool_calls": []}))
            == counter.count_message(plain.model_copy(update={"tool_calls": []})))
