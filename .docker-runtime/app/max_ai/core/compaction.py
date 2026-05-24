"""Core models and helpers for context compaction."""

from __future__ import annotations

import json
import logging
import typing as t

import tiktoken
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .messages import AssistantMessage, CoreMessage, ToolMessage


logger = logging.getLogger(__name__)


class TokenBudget(BaseModel):
    """Materialized token limits for one model context window."""

    max_context_tokens: int = Field(..., description="Maximum model context window.")
    max_input_tokens: int = Field(..., description="Maximum input token budget.")
    reserved_output_tokens: int = Field(
        ..., description="Tokens reserved for the model response."
    )
    safety_margin_tokens: int = Field(
        ..., description="Unused margin for provider overhead and counting drift."
    )
    max_summary_tokens: int = Field(..., description="Maximum summary token budget.")
    max_history_tokens: int = Field(
        ..., description="Maximum recent raw-history token budget."
    )
    prompt_budget_tokens: int = Field(
        ..., description="Remaining budget for prompt layers and tool schemas."
    )


class MessageGroup(BaseModel):
    """Atomic message group that should be kept or compacted as one unit."""

    messages: list[CoreMessage] = Field(default_factory=list)
    token_count: int = 0

    @property
    def is_tool_group(self) -> bool:
        return any(
            isinstance(message, AssistantMessage) and bool(message.tool_calls)
            for message in self.messages
        )


class TokenCounter(t.Protocol):
    """Minimal token-counting surface required by compaction helpers."""

    def count_message(self, message: CoreMessage) -> int: ...

    def count_messages(self, messages: list[CoreMessage]) -> int: ...


class TokenBudgetStrategy(BaseModel):
    """Token allocation and counting for one LLM context window."""

    input_context_ratio: float = Field(default=0.4, gt=0, lt=1)
    reserved_output_ratio: float = Field(default=0.3, ge=0, lt=1)
    safety_margin_ratio: float = Field(default=0.3, ge=0, lt=1)

    max_summary_tokens: int = Field(default=1000, gt=0)
    max_history_tokens: int = Field(default=1000, gt=0)

    tokenizer_base: str = "o200k_base"
    message_overhead_tokens: int = Field(default=5, ge=0)

    _encoder: t.Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_ratios(self) -> t.Self:
        total = (
            self.input_context_ratio
            + self.reserved_output_ratio
            + self.safety_margin_ratio
        )
        if total > 1:
            raise ValueError("Token budget ratios cannot sum to more than 1.0")
        return self

    def model_post_init(self, __context: t.Any) -> None:
        try:
            self._encoder = tiktoken.get_encoding(self.tokenizer_base)
        except Exception:
            logger.warning(
                "Tokenizer %r not found; falling back to o200k_base",
                self.tokenizer_base,
            )
            self._encoder = tiktoken.get_encoding("o200k_base")

    def build_budget(self, *, max_context_tokens: int) -> TokenBudget:
        max_input_tokens = int(max_context_tokens * self.input_context_ratio)
        summary_budget = min(self.max_summary_tokens, max_input_tokens)
        history_budget = min(
            self.max_history_tokens,
            max(0, max_input_tokens - summary_budget),
        )

        return TokenBudget(
            max_context_tokens=max_context_tokens,
            max_input_tokens=max_input_tokens,
            reserved_output_tokens=int(max_context_tokens * self.reserved_output_ratio),
            safety_margin_tokens=int(max_context_tokens * self.safety_margin_ratio),
            max_summary_tokens=summary_budget,
            max_history_tokens=history_budget,
            prompt_budget_tokens=max_input_tokens - summary_budget - history_budget,
        )

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
    max_history_tokens: int,
) -> tuple[list[CoreMessage], list[CoreMessage]]:
    """Split atomic groups into old messages and recent messages."""

    if not groups:
        return [], []

    kept_groups: list[MessageGroup] = []
    used_tokens = 0

    for group in reversed(groups):
        next_total = used_tokens + group.token_count

        if next_total > max_history_tokens and kept_groups:
            break

        kept_groups.append(group)
        used_tokens = next_total

        if used_tokens >= max_history_tokens:
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


__all__ = [
    "MessageGroup",
    "TokenBudget",
    "TokenBudgetStrategy",
    "TokenCounter",
    "group_atomic_messages",
    "split_recent_messages",
]
