from __future__ import annotations

from max_ai.core.compaction import CompactionOutput
from max_ai.base.compaction import (
    TokenCounter,
    group_atomic_messages,
    split_recent_messages,
)
from max_ai.core.messages import (
    AssistantMessage,
    ImagePart,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class FixedCounter:
    def count_message(self, message):
        return message.token_count or 1

    def count_messages(self, messages):
        return sum(self.count_message(message) for message in messages)


def user(content: str, *, tokens: int = 1) -> UserMessage:
    return UserMessage(source="user", content=content, token_count=tokens)


def assistant(content: str, *, tokens: int = 1) -> AssistantMessage:
    return AssistantMessage(source="agent", content=content, token_count=tokens)



def test_compaction_output_defaults_to_empty_sections():
    output = CompactionOutput(summary="Working on compaction")

    assert output.summary == "Working on compaction"
    assert output.objective == []
    assert output.pending == []
    assert output.successfully_done == []
    assert output.decisions == []
    assert output.important_context == []

def test_count_message_prefers_existing_token_count():
    strategy = TokenCounter()
    message = user("this text should not be counted", tokens=123)

    assert strategy.count_message(message) == 123


def test_count_message_counts_tool_call_as_serialized_payload():
    strategy = TokenCounter()
    tool_call = ToolCall(
        id="call_1",
        tool_name="search",
        parameters={"query": "weather in Santo Domingo"},
    )
    message = AssistantMessage(
        source="agent",
        content="",
        tool_calls=[tool_call],
    )

    assert strategy.count_message(message) > 0
    assert strategy.count_message(message) == strategy.count_serialized(message)


def test_count_message_sanitizes_multimodal_bytes():
    strategy = TokenCounter()
    message = UserMessage(
        source="user",
        content=[
            TextPart(text="describe this image"),
            ImagePart(data=b"x" * 10_000, mime_type="image/png"),
        ],
    )

    token_count = strategy.count_message(message)

    assert token_count > 0
    assert token_count < strategy.count_text(message.model_dump_json(exclude_none=True))


def test_group_atomic_messages_keeps_assistant_tool_call_with_tool_result():
    tool_call = ToolCall(id="call_1", tool_name="ping", parameters={"x": 1})
    assistant_call = AssistantMessage(
        source="agent",
        content="",
        tool_calls=[tool_call],
        token_count=3,
    )
    tool_result = ToolMessage.success_message(
        tool_call_id="call_1",
        tool_name="ping",
        content="pong",
        source="tool",
    ).model_copy(update={"token_count": 2})
    messages = [
        user("hello", tokens=1),
        assistant_call,
        tool_result,
        assistant("done", tokens=1),
    ]

    groups = group_atomic_messages(messages, FixedCounter())

    assert len(groups) == 3
    assert groups[1].messages == [assistant_call, tool_result]
    assert groups[1].token_count == 5
    assert groups[1].is_tool_group is True


def test_split_recent_messages_uses_newest_groups_within_token_budget():
    groups = group_atomic_messages(
        [
            user("old 1", tokens=4),
            assistant("old 2", tokens=4),
            user("recent 1", tokens=3),
            assistant("recent 2", tokens=2),
        ],
        FixedCounter(),
    )

    old_messages, recent_messages = split_recent_messages(
        groups,
        max_tokens=5,
    )

    assert [message.text() for message in old_messages] == ["old 1", "old 2"]
    assert [message.text() for message in recent_messages] == [
        "recent 1",
        "recent 2",
    ]


def test_split_recent_messages_keeps_last_group_even_when_it_exceeds_budget():
    groups = group_atomic_messages(
        [
            user("old", tokens=1),
            assistant("huge latest message", tokens=100),
        ],
        FixedCounter(),
    )

    old_messages, recent_messages = split_recent_messages(
        groups,
        max_tokens=10,
    )

    assert [message.text() for message in old_messages] == ["old"]
    assert [message.text() for message in recent_messages] == ["huge latest message"]


def test_split_recent_messages_does_not_split_tool_group():
    tool_call = ToolCall(id="call_1", tool_name="lookup", parameters={})
    assistant_call = AssistantMessage(
        source="agent",
        content="",
        tool_calls=[tool_call],
        token_count=3,
    )
    tool_result = ToolMessage.success_message(
        tool_call_id="call_1",
        tool_name="lookup",
        content="result",
        source="tool",
    ).model_copy(update={"token_count": 3})
    groups = group_atomic_messages(
        [
            user("old", tokens=2),
            assistant_call,
            tool_result,
        ],
        FixedCounter(),
    )

    old_messages, recent_messages = split_recent_messages(
        groups,
        max_tokens=4,
    )

    assert old_messages == [groups[0].messages[0]]
    assert recent_messages == [assistant_call, tool_result]
