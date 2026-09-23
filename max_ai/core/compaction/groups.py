"""Atomic message blocks: a tool call and its results never get split."""

from __future__ import annotations

from ..messages import (
    HARNESS_SOURCE,
    AssistantMessage,
    CoreMessage,
    ToolMessage,
    UserMessage,
)
from ..model.compaction import MessageGroup
from .token_counter import TokenCounter


def group_atomic_messages(
    messages: list[CoreMessage],
    counter: TokenCounter,
) -> list[MessageGroup]:
    """Group messages without splitting assistant tool calls from tool results."""

    groups: list[MessageGroup] = []
    i = 0

    while i < len(messages):
        message = messages[i]

        if isinstance(message, AssistantMessage) and message.tool_calls:
            expected_tool_ids = {tool_call.id for tool_call in message.tool_calls}
            group_messages: list[CoreMessage] = [message]
            i += 1

            while i < len(messages):
                next_message = messages[i]
                if (
                    isinstance(next_message, ToolMessage)
                    and next_message.tool_call_id in expected_tool_ids
                ):
                    group_messages.append(next_message)
                    i += 1
                    continue

                break

            groups.append(
                MessageGroup(
                    messages=group_messages,
                    token_count=counter.count_messages(group_messages),
                )
            )
            continue

        groups.append(
            MessageGroup(
                messages=[message],
                token_count=counter.count_message(message),
            )
        )
        i += 1

    return groups


def split_recent_messages(
    groups: list[MessageGroup],
    *,
    max_tokens: int,
    min_keep_groups: int = 1,
) -> tuple[list[CoreMessage], list[CoreMessage]]:
    """Split atomic groups into old messages and newest messages within budget.

    ``min_keep_groups`` recent groups always survive, budget or not: a
    group is an assistant message plus its tool results, and evicting the
    ones the model is actively working from makes it re-call the same
    tools (the data literally disappeared from its context).
    """

    if not groups:
        return [], []

    kept_groups: list[MessageGroup] = []
    used_tokens = 0

    for group in reversed(groups):
        next_total = used_tokens + group.token_count

        if len(kept_groups) >= min_keep_groups:
            if next_total > max_tokens:
                break

        kept_groups.append(group)
        used_tokens = next_total

        if len(kept_groups) >= min_keep_groups and used_tokens >= max_tokens:
            break

    kept_groups.reverse()
    kept_group_ids = {id(group) for group in kept_groups}

    old_messages = [
        message
        for group in groups
        if id(group) not in kept_group_ids
        for message in group.messages
    ]
    recent_messages = [message for group in kept_groups for message in group.messages]

    return old_messages, recent_messages


def turn_starts(blocks: list[MessageGroup]) -> list[int]:
    """Indexes of the blocks that open a turn: a real user message.

    Harness-written user messages (e.g. "max iterations reached") don't
    start a turn.
    """
    return [
        index for index, block in enumerate(blocks)
        if isinstance(block.messages[0], UserMessage)
        and block.messages[0].source != HARNESS_SOURCE
    ]


def current_turn_start(blocks: list[MessageGroup]) -> int:
    """Index of the block holding the last real user message (0 if none)."""
    starts = turn_starts(blocks)
    return starts[-1] if starts else 0


__all__ = ["current_turn_start", "turn_starts", "group_atomic_messages", "split_recent_messages"]
