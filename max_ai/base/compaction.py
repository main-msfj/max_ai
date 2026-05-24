"""Core contract and helpers for context compaction strategies."""

from __future__ import annotations

import json
import logging
import typing as t
from abc import ABC, abstractmethod

import tiktoken
from pydantic import BaseModel, Field, PrivateAttr

from ..core.compaction import CompactionResult, MessageGroup
from ..core.messages import AssistantMessage, CoreMessage, ToolMessage

if t.TYPE_CHECKING:
    from .clients import CoreChatCompletionClient
    from ..types.run_context import RunContext
    from ..types.stacks import PromptCtx


logger = logging.getLogger(__name__)


class TokenCounter(BaseModel):
    """Token counting helper used by compaction and telemetry."""

    tokenizer_base: str = "o200k_base"
    message_overhead_tokens: int = Field(default=5, ge=0)

    _encoder: t.Any = PrivateAttr(default=None)

    def model_post_init(self, __context: t.Any) -> None:
        try:
            self._encoder = tiktoken.get_encoding(self.tokenizer_base)
        except Exception:
            logger.warning(
                "Tokenizer %r not found; falling back to o200k_base",
                self.tokenizer_base,
            )
            self._encoder = tiktoken.get_encoding("o200k_base")

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def count_message(self, message: CoreMessage) -> int:
        if message.token_count > 0:
            return message.token_count

        if isinstance(message, ToolMessage):
            return self.count_serialized(message)

        if isinstance(message, AssistantMessage) and message.tool_calls:
            return self.count_serialized(message)

        if message.is_multimodal():
            return self.count_text(self._message_budget_payload(message))

        return self.count_text(message.text()) + self.message_overhead_tokens

    def count_messages(self, messages: list[CoreMessage]) -> int:
        return sum(self.count_message(message) for message in messages)

    def count_serialized(self, value: t.Any) -> int:
        if isinstance(value, BaseModel):
            return self.count_text(value.model_dump_json(exclude_none=True))

        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)

        return self.count_text(text)

    def _message_budget_payload(self, message: CoreMessage) -> str:
        data = message.model_dump(exclude_none=True)

        content = data.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("data") is not None:
                    part_type = part.get("type", "binary")
                    part["data"] = f"<{part_type}_bytes>"

        return json.dumps(data, ensure_ascii=False, default=str)


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
) -> tuple[list[CoreMessage], list[CoreMessage]]:
    """Split atomic groups into old messages and newest messages within budget."""

    if not groups:
        return [], []

    kept_groups: list[MessageGroup] = []
    used_tokens = 0

    for group in reversed(groups):
        next_total = used_tokens + group.token_count

        if next_total > max_tokens and kept_groups:
            break

        kept_groups.append(group)
        used_tokens = next_total

        if used_tokens >= max_tokens:
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


class CoreCompaction(BaseModel, ABC):
    """Abstract base for strategies that keep a run within context budget."""

    @abstractmethod
    async def compact(
        self,
        *,
        ctx: RunContext,
        prompts: PromptCtx,
        max_context_tokens: int,
        client: CoreChatCompletionClient,
    ) -> CompactionResult:
        """Apply compaction to the provided runtime context."""
        ...


__all__ = [
    "CompactionResult",
    "CoreCompaction",
    "TokenCounter",
    "group_atomic_messages",
    "split_recent_messages",
]
