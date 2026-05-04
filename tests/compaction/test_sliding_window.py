from __future__ import annotations

import pytest

from max_ai.compaction import SlidingWindowCompaction
from max_ai.core.compaction import TokenBudgetStrategy
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx


def user(content: str, *, tokens: int = 1) -> UserMessage:
    return UserMessage(source="user", content=content, token_count=tokens)


def assistant(content: str, *, tokens: int = 1) -> AssistantMessage:
    return AssistantMessage(source="agent", content=content, token_count=tokens)


def make_compaction(*, max_history_tokens: int = 10) -> SlidingWindowCompaction:
    return SlidingWindowCompaction(
        token_strategy=TokenBudgetStrategy(
            input_context_ratio=0.9,
            reserved_output_ratio=0.0,
            safety_margin_ratio=0.0,
            max_summary_tokens=1,
            max_history_tokens=max_history_tokens,
        )
    )


def make_prompts() -> PromptCtx:
    return PromptCtx.model_construct(stack=None, variables={}, rendered_layers={})


@pytest.mark.asyncio
async def test_sliding_window_noops_when_context_fits():
    ctx = RunContext(
        messages=[
            user("hello", tokens=2),
            assistant("hi", tokens=2),
        ]
    )
    compaction = make_compaction(max_history_tokens=10)

    result = await compaction.compact(
        ctx=ctx,
        prompts=make_prompts(),
        max_context_tokens=100,
    )

    assert result.changed is False
    assert result.old_messages == []
    assert result.recent_messages == ctx.messages
    assert result.recent_token_count == 4


@pytest.mark.asyncio
async def test_sliding_window_mutates_context_to_recent_messages():
    ctx = RunContext(
        messages=[
            user("old 1", tokens=4),
            assistant("old 2", tokens=4),
            user("recent 1", tokens=3),
            assistant("recent 2", tokens=2),
        ]
    )
    compaction = make_compaction(max_history_tokens=5)

    result = await compaction.compact(
        ctx=ctx,
        prompts=make_prompts(),
        max_context_tokens=100,
    )

    assert result.changed is True
    assert [message.text() for message in result.old_messages] == ["old 1", "old 2"]
    assert [message.text() for message in ctx.messages] == [
        "recent 1",
        "recent 2",
    ]


@pytest.mark.asyncio
async def test_sliding_window_does_not_split_tool_group():
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
    ctx = RunContext(
        messages=[
            user("old", tokens=2),
            assistant_call,
            tool_result,
        ]
    )
    compaction = make_compaction(max_history_tokens=4)

    result = await compaction.compact(
        ctx=ctx,
        prompts=make_prompts(),
        max_context_tokens=100,
    )

    assert [message.text() for message in result.old_messages] == ["old"]
    assert ctx.messages == [assistant_call, tool_result]
    assert result.recent_token_count == 6
