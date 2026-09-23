"""Token budget math: how much of the window live messages may use.

    window = system prompt + live messages + reserved output + safety margin

``capacity`` is what is left for live messages; compaction triggers above
``threshold`` of it and keeps ``budget`` of it as raw recent messages.
"""

from __future__ import annotations

import typing as t

from ...config import setting


def client_max_output_tokens(client: t.Any) -> int:
    options = getattr(client, "generation_options", None)
    if isinstance(options, dict) and options.get("max_tokens") is not None:
        return int(options["max_tokens"])

    config = getattr(client, "config", None)
    max_output = getattr(config, "max_output_tokens", 0) or 0
    if max_output:
        return int(max_output)

    return setting.compaction_min_output_tokens


def live_message_capacity_tokens(
    max_context_tokens: int,
    *,
    max_output_tokens: int,
    prompt_tokens: int | None = None,
) -> int:
    """Tokens available for live messages.

    ``prompt_tokens`` is the *actual* rendered system-prompt size when the
    caller knows it (``PromptCtx.prompt_tokens``); the configured
    ``compaction_prompt_budget_tokens`` constant is only a fallback for
    callers without a prompt context. Using the real number matters: a fat
    memory/knowledge prompt can dwarf the fixed budget and silently blow
    the window.
    """
    if max_context_tokens <= 0:
        return 0

    prompt_budget = (
        prompt_tokens
        if prompt_tokens is not None and prompt_tokens > 0
        else setting.compaction_prompt_budget_tokens
    )
    safety_margin = int(max_context_tokens * setting.compaction_safety_margin_ratio)
    live_tokens = (
        max_context_tokens
        - prompt_budget
        - max_output_tokens
        - safety_margin
    )
    return max(0, live_tokens)


def live_message_threshold_tokens(
    max_context_tokens: int,
    *,
    max_output_tokens: int,
    prompt_tokens: int | None = None,
    ratio: float | None = None,
) -> int:
    """Live-message tokens above which compaction triggers.

    ``ratio`` is the strategy's threshold; ``None`` falls back to the global
    ``compaction_live_message_threshold`` setting.
    """
    ratio = setting.compaction_live_message_threshold if ratio is None else ratio
    capacity = live_message_capacity_tokens(
        max_context_tokens,
        max_output_tokens=max_output_tokens,
        prompt_tokens=prompt_tokens,
    )
    if capacity <= 0:
        return 0
    return max(1, int(capacity * ratio))


def live_message_budget_tokens(capacity_tokens: int, ratio: float | None = None) -> int:
    """Raw recent-message tokens kept after compacting (``ratio`` of capacity)."""
    if capacity_tokens <= 0:
        return 0
    ratio = setting.compaction_live_message_keep_ratio if ratio is None else ratio
    return max(1, int(capacity_tokens * ratio))


__all__ = [
    "client_max_output_tokens",
    "live_message_budget_tokens",
    "live_message_capacity_tokens",
    "live_message_threshold_tokens",
]
