"""The compaction contract. Shared machinery lives in ``core.compaction``."""

from __future__ import annotations

import copy
import logging
import typing as t
from abc import ABC, abstractmethod

from pydantic import JsonValue

from ..core.compaction import (
    CompactionConfig,
    CompactionResult,
    MessageGroup,
    TokenCounter,
    client_max_output_tokens,
    current_turn_start,
    group_atomic_messages,
    live_message_budget_tokens,
    live_message_capacity_tokens,
    live_message_threshold_tokens,
)
from ..core.messages import HARNESS_SOURCE, AssistantMessage, CoreMessage, ToolMessage
from ..errors.compaction import CompactionError
from .component import ComponentBase

if t.TYPE_CHECKING:
    from ..types.run_context import RunContext
    from ..types.stacks import PromptCtx
    from .clients import CoreChatCompletionClient
    from .memory import CoreMemoryRegistry

logger = logging.getLogger(__name__)


class CoreCompaction(ComponentBase[CompactionConfig], ABC):
    """Base for strategies that keep a run within its context window.

    ``compact`` is concrete and owned by the harness: it groups messages into
    atomic blocks, runs the cheap prune pass, calls the strategy only when
    still over budget, validates the output and updates memory. Strategies
    implement ``_compact`` and never see ``RunContext``.

    Contract: ``compact`` never mutates ``ctx``. The caller (the reasoning
    loop) applies ``result.messages`` and ``result.state``.
    """

    component_type = "compaction"
    component_schema = CompactionConfig

    def __init__(
        self,
        *,
        threshold: float = 0.8,
        keep_ratio: float = 0.4,
        min_keep_groups: int = 2,
        truncate_tool_outputs: bool = True,
        tool_output_max_tokens: int = 500,
        drop_harness_messages: bool = True,
        client: CoreChatCompletionClient | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        # Validated through the config so bad values fail at construction.
        self.config = CompactionConfig(
            threshold=threshold,
            keep_ratio=keep_ratio,
            min_keep_groups=min_keep_groups,
            truncate_tool_outputs=truncate_tool_outputs,
            tool_output_max_tokens=tool_output_max_tokens,
            drop_harness_messages=drop_harness_messages,
        )
        self.client = client
        self.token_counter = token_counter or TokenCounter()

    # -------- HARNESS (not overridden) ------------------------------------------------
    def should_compact(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
        max_context_tokens: int,
        max_output_tokens: int,
    ) -> bool:
        """Cheap check the loop runs before every model call.

        Delegates to ``_over_limit``, which strategies override (e.g. more
        than N turns). Pending tool calls are not checked here but in
        ``compact``, which is never overridden, so no strategy can skip it.
        """
        threshold = live_message_threshold_tokens(
            max_context_tokens,
            max_output_tokens=max_output_tokens,
            prompt_tokens=prompts.prompt_tokens or None,
            ratio=self.config.threshold,
        )
        blocks = group_atomic_messages(ctx.messages, self.token_counter)
        # The history is sent to the model too, so it counts against the window.
        live = self.token_counter.count_messages(ctx.message_history.iter_messages())
        live += sum(block.token_count for block in blocks)
        return self._over_limit(blocks, live_tokens=live, threshold=threshold)

    async def compact(
        self,
        *,
        ctx: RunContext,
        prompts: PromptCtx,
        max_context_tokens: int,
        client: CoreChatCompletionClient,
        memory: CoreMemoryRegistry | None = None,
    ) -> CompactionResult:
        """Run the full pipeline. Never mutates ``ctx``.

        ``client`` is the agent's: it sizes the reserved output. Summaries and
        memory use ``self.client`` when set. With tool calls still pending
        (approval, ask_user) nothing is touched: their blocks aren't closed.
        """
        count = self.token_counter.count_messages
        history = count(ctx.message_history.iter_messages())
        before = history + count(ctx.messages)
        unchanged = CompactionResult(
            messages=list(ctx.messages),
            state=copy.deepcopy(ctx.compaction.state),
            tokens_before=before,
            tokens_after=before,
        )
        if any(not record.is_consumed for record in ctx.tool_state.records.values()):
            return unchanged

        max_output = client_max_output_tokens(client)
        prompt_tokens = prompts.prompt_tokens or None
        capacity = live_message_capacity_tokens(
            max_context_tokens, max_output_tokens=max_output, prompt_tokens=prompt_tokens,
        )
        threshold = live_message_threshold_tokens(
            max_context_tokens, max_output_tokens=max_output,
            prompt_tokens=prompt_tokens, ratio=self.config.threshold,
        )
        # The history also takes window space; strategies only see messages.
        budget = max(0, live_message_budget_tokens(capacity, ratio=self.config.keep_ratio) - history)

        blocks = group_atomic_messages(ctx.messages, self.token_counter)
        blocks = self._prune(blocks, current_turn_start=current_turn_start(blocks))
        pruned = [message for block in blocks for message in block.messages]
        pruned_tokens = history + count(pruned)
        if not self._over_limit(blocks, live_tokens=pruned_tokens, threshold=threshold):
            return unchanged.model_copy(update={
                "changed": pruned != ctx.messages,
                "messages": pruned,
                "tokens_after": pruned_tokens,
                "pruned_only": True,
            })

        strategy_client = self.client or client
        result = await self._compact(
            blocks,
            state=copy.deepcopy(ctx.compaction.state),
            budget_tokens=budget,
            client=strategy_client,
        )
        self._validate(result.messages)

        if memory is not None and result.old_messages:
            try:
                await self._update_memory(
                    result.old_messages, memory=memory, client=strategy_client,
                )
            except Exception:
                logger.exception("Memory update during compaction failed; continuing")

        return result.model_copy(update={
            "changed": result.changed or result.messages != ctx.messages,
            "tokens_before": before,
            "tokens_after": history + count(result.messages),
        })

    # -------- STRATEGY ---------------------------------------------------------------
    @abstractmethod
    async def _compact(
        self,
        blocks: list[MessageGroup],
        *,
        state: dict[str, JsonValue],
        budget_tokens: int,
        client: CoreChatCompletionClient,
    ) -> CompactionResult:
        """Strategy hook. Receives whole blocks only, so it cannot split a
        tool call from its results. Returns the new window in ``messages``,
        what left it in ``old_messages`` and its new ``state``."""

    def _over_limit(
        self, blocks: list[MessageGroup], *, live_tokens: int, threshold: int,
    ) -> bool:
        """Whether the window needs compacting. Asked before compacting and
        again after the prune pass (if pruning was enough, ``_compact`` is
        skipped). Default: live tokens (history included) over the
        threshold; ``threshold`` is 0 when the model window is unknown.
        """
        return threshold > 0 and live_tokens > threshold

    def render(self, state: dict[str, JsonValue]) -> str | None:
        """What the model sees of the compacted past (a system prompt block).
        Default: nothing. Summary strategies return their summary."""
        return None

    # -------- OVERRIDABLE STEPS ---------------------------------------------------------
    def _prune(
        self,
        blocks: list[MessageGroup],
        *,
        current_turn_start: int,
    ) -> list[MessageGroup]:
        """Cheap, LLM-free cleanup of blocks older than the ``min_keep_groups``
        newest. Returns NEW blocks; never edits the original messages.

        - ``drop_harness_messages``: harness messages from previous turns.
        - ``truncate_tool_outputs``: shrink old tool results and arguments,
          keeping ids and structure, leaving a hint (size, status, first lines).

        ``current_turn_start`` is a block index (see
        ``core.compaction.current_turn_start``).
        """
        protected = len(blocks) - self.config.min_keep_groups
        pruned: list[MessageGroup] = []
        for index, block in enumerate(blocks):
            if index >= protected:
                pruned.append(block)
                continue
            messages = block.messages
            if self.config.drop_harness_messages and index < current_turn_start:
                messages = [m for m in messages if m.source != HARNESS_SOURCE]
            if self.config.truncate_tool_outputs:
                messages = [self._shrink(m) for m in messages]
            if not messages:
                continue
            if messages == block.messages:
                pruned.append(block)
            else:
                pruned.append(MessageGroup(
                    messages=messages,
                    token_count=self.token_counter.count_messages(messages),
                ))
        return pruned

    def _shrink(self, message: CoreMessage) -> CoreMessage:
        """Copy of ``message`` with oversized tool output/arguments cut down."""
        cap = self.config.tool_output_max_tokens
        if isinstance(message, ToolMessage):
            text = message.text()
            cut = self._truncate_text(text, cap)
            if cut is None and not message.is_multimodal():
                return message
            status = "ok" if message.success else f"failed: {message.error or 'error'}"
            return message.model_copy(update={
                "content": (cut if cut is not None else text) + (
                    f"\n[{message.tool_name} result {status}; truncated by "
                    "compaction. Call the tool again if you need the rest.]"
                ),
                "token_count": 0,
            })
        if isinstance(message, AssistantMessage) and message.tool_calls:
            calls, changed = [], False
            for call in message.tool_calls:
                params = {}
                for key, value in call.parameters.items():
                    cut = self._truncate_text(value, cap) if isinstance(value, str) else None
                    params[key] = value if cut is None else cut + "\n[argument truncated by compaction]"
                    changed |= cut is not None
                calls.append(call.model_copy(update={"parameters": params}))
            if changed:
                return message.model_copy(update={"tool_calls": calls, "token_count": 0})
        return message

    def _truncate_text(self, text: str, cap: int) -> str | None:
        """First ``cap`` tokens of ``text`` plus its original size, or ``None``
        when it already fits."""
        tokens = self.token_counter.encode(text)
        if len(tokens) <= cap:
            return None
        head = self.token_counter.decode(tokens[:cap])
        return f"{head}\n[… {len(tokens):,} tokens, kept the first {cap:,}]"

    async def _update_memory(
        self,
        old_messages: list[CoreMessage],
        *,
        memory: CoreMemoryRegistry,
        client: CoreChatCompletionClient,
    ) -> None:
        """Extract durable facts from messages leaving the window and save
        them (``MemoryMaintenanceOutput`` → ``memory.create_or_update``).
        ``compact`` logs failures; they never reach the turn."""
        # TODO
        ...

    def _validate(self, messages: list[CoreMessage]) -> None:
        """Check the window a provider will accept, or raise ``CompactionError``.

        Every ToolMessage answers a call of the assistant message that opens
        its block, and every tool call of that message has its result.
        """
        open_index, waiting = -1, set[str]()
        for index, message in enumerate(messages):
            if isinstance(message, ToolMessage):
                if message.tool_call_id not in waiting:
                    raise CompactionError.orphan_tool_result(index, message.tool_call_id)
                waiting.discard(message.tool_call_id)
                continue
            # Any other message closes the previous block.
            if waiting:
                raise CompactionError.unanswered_tool_calls(open_index, waiting)
            if isinstance(message, AssistantMessage) and message.tool_calls:
                open_index = index
                waiting = {call.id for call in message.tool_calls}
        if waiting:
            raise CompactionError.unanswered_tool_calls(open_index, waiting)

    # -------- SERIALIZATION ------------------------------------------------------------
    def _to_config(self) -> CompactionConfig:
        client = self.client.dump_component().model_dump() if self.client else None
        return self.config.model_copy(update={"client": client})

    @classmethod
    def _from_config(cls, config: CompactionConfig) -> t.Self:
        from .clients import CoreChatCompletionClient

        client = (
            CoreChatCompletionClient.load_component(config.client)
            if config.client else None
        )
        return cls(**config.model_dump(exclude={"client"}), client=client)


__all__ = [
    "CompactionConfig",
    "CompactionResult",
    "CoreCompaction",
]
