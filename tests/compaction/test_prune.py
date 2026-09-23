"""CoreCompaction._prune: the cheap, LLM-free cleanup pass."""

from __future__ import annotations

from max_ai.base.compaction import CoreCompaction
from max_ai.core.compaction import (
    CompactionResult,
    TokenCounter,
    current_turn_start,
    group_atomic_messages,
)
from max_ai.core.messages import (
    HARNESS_SOURCE,
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

BIG = "línea de log con bastante texto adentro\n" * 400  # ~3.6k tokens


class KeepAll(CoreCompaction):
    component_provider_override = "tests.KeepAll"

    async def _compact(self, blocks, *, state, budget_tokens, client):
        return CompactionResult(messages=[m for b in blocks for m in b.messages])


def user(text: str = "hola") -> UserMessage:
    return UserMessage(source="marvin", content=text)


def call(call_id: str, **params) -> AssistantMessage:
    return AssistantMessage(
        source="a", content="", tool_calls=[ToolCall(id=call_id, tool_name="bash", parameters=params)],
    )


def result(call_id: str, content: str = "ok", success: bool = True) -> ToolMessage:
    return ToolMessage(
        source="tool", content=content, tool_call_id=call_id, tool_name="bash",
        success=success, error=None if success else "exit 1",
    )


def gate(text: str = "Task not yet complete.") -> SystemMessage:
    return SystemMessage(source=HARNESS_SOURCE, content=text)


def answer(text: str = "listo") -> AssistantMessage:
    return AssistantMessage(source="a", content=text)


def prune(messages, strategy=None):
    strategy = strategy or KeepAll(min_keep_groups=2)
    blocks = group_atomic_messages(messages, TokenCounter())
    pruned = strategy._prune(blocks, current_turn_start=current_turn_start(blocks))
    flat = [m for b in pruned for m in b.messages]
    strategy._validate(flat)  # pruning must never break the window
    return pruned, flat


def test_current_turn_starts_at_last_real_user_message():
    msgs = [user("1"), answer(), user("2"), answer(),
            UserMessage(source=HARNESS_SOURCE, content="max iterations"), answer()]
    blocks = group_atomic_messages(msgs, TokenCounter())
    assert current_turn_start(blocks) == 2


def test_drops_harness_messages_only_from_previous_turns():
    msgs = [user("1"), answer(), gate("old rejection"), user("2"), gate("current rejection"),
            answer(), answer(), answer()]
    _, flat = prune(msgs)
    texts = [m.text() for m in flat]
    assert "old rejection" not in texts
    assert "current rejection" in texts  # the model still has to act on it


def test_truncates_old_tool_output_keeping_structure():
    msgs = [user(), call("c1", command="cat app.log"), result("c1", BIG), answer(), user(), answer()]
    pruned, flat = prune(msgs)
    tool = next(m for m in flat if isinstance(m, ToolMessage))
    assert (tool.tool_call_id, tool.tool_name, tool.success) == ("c1", "bash", True)
    assert tool.text().startswith("línea de log")
    assert "kept the first 500" in tool.text() and "Call the tool again" in tool.text()
    assert TokenCounter().count_text(tool.text()) < 600
    assert pruned[1].token_count < 700  # block total recounted


def test_failed_tool_hint_keeps_the_error():
    msgs = [user(), call("c1"), result("c1", BIG, success=False), answer(), user(), answer()]
    _, flat = prune(msgs)
    tool = next(m for m in flat if isinstance(m, ToolMessage))
    assert "failed: exit 1" in tool.text() and tool.success is False


def test_truncates_giant_arguments():
    msgs = [user(), call("c1", path="a.py", content=BIG), result("c1"), answer(), user(), answer()]
    _, flat = prune(msgs)
    params = next(m for m in flat if isinstance(m, AssistantMessage) and m.tool_calls).tool_calls[0].parameters
    assert params["path"] == "a.py"
    assert "argument truncated by compaction" in params["content"]


def test_newest_blocks_are_never_touched():
    msgs = [user(), call("c1"), result("c1", BIG)]  # the model is working with this output
    pruned, flat = prune(msgs)
    assert flat[-1].text() == BIG


def test_small_outputs_and_untouched_blocks_are_reused_as_is():
    msgs = [user(), call("c1"), result("c1", "short"), answer(), user(), answer()]
    blocks = group_atomic_messages(msgs, TokenCounter())
    pruned = KeepAll(min_keep_groups=2)._prune(blocks, current_turn_start=4)
    assert all(a is b for a, b in zip(pruned, blocks))


def test_never_mutates_original_messages():
    original = result("c1", BIG)
    msgs = [user(), call("c1"), original, answer(), user(), answer()]
    prune(msgs)
    assert original.text() == BIG


def test_switches_turn_both_passes_off():
    msgs = [user("1"), gate("old"), call("c1"), result("c1", BIG), user("2"), answer(), answer()]
    strategy = KeepAll(min_keep_groups=2, truncate_tool_outputs=False, drop_harness_messages=False)
    _, flat = prune(msgs, strategy)
    assert [m.text() for m in flat] == [m.text() for m in msgs]
