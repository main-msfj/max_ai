"""CoreCompaction._validate: the window a strategy returns must be sendable."""

from __future__ import annotations

import pytest

from max_ai.base.compaction import CoreCompaction
from max_ai.core.compaction import CompactionResult
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from max_ai.errors.compaction import CompactionError


class KeepAll(CoreCompaction):
    component_provider_override = "tests.KeepAll"

    async def _compact(self, blocks, *, state, budget_tokens, client):
        return CompactionResult(messages=[m for b in blocks for m in b.messages])


def user(text: str = "hi") -> UserMessage:
    return UserMessage(source="u", content=text)


def calls(*ids: str) -> AssistantMessage:
    return AssistantMessage(
        source="a", content="", tool_calls=[ToolCall(id=i, tool_name="bash") for i in ids],
    )


def result(tool_call_id: str) -> ToolMessage:
    return ToolMessage(
        source="tool", content="ok", tool_call_id=tool_call_id, tool_name="bash", success=True,
    )


def answer(text: str = "done") -> AssistantMessage:
    return AssistantMessage(source="a", content=text)


validate = KeepAll()._validate


@pytest.mark.parametrize("window", [
    [],
    [user(), answer()],
    [user(), calls("c1", "c2"), result("c1"), result("c2"), answer()],
    [calls("c1", "c2"), result("c2"), result("c1")],  # any order inside a block
    [user(), calls("c1"), result("c1"), calls("c2"), result("c2"), answer()],
    # A compacted window may start at any block boundary.
    [calls("c3"), result("c3"), answer()],
])
def test_valid_windows(window):
    validate(window)


def test_tool_result_without_its_call():
    with pytest.raises(CompactionError, match=r"messages\[1\] answers tool call 'c1'"):
        validate([user(), result("c1"), answer()])


def test_tool_result_from_another_block():
    with pytest.raises(CompactionError, match="'c1'"):
        validate([calls("c1"), result("c1"), user(), result("c1")])


def test_duplicated_tool_result():
    with pytest.raises(CompactionError, match=r"messages\[2\]"):
        validate([calls("c1"), result("c1"), result("c1")])


def test_call_left_without_result_before_next_message():
    with pytest.raises(CompactionError, match=r"messages\[1\] calls tools 'c2'"):
        validate([user(), calls("c1", "c2"), result("c1"), answer()])


def test_call_left_without_result_at_the_end():
    with pytest.raises(CompactionError, match=r"messages\[1\] calls tools 'c1', 'c2'"):
        validate([user(), calls("c1", "c2")])
