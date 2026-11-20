"""The compaction contract: make a conversation fit in the model's window."""

from __future__ import annotations

import asyncio
import logging
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from ..core.compaction.groups import group_atomic_messages, split_recent_messages
from ..core.messages import CoreMessage, Message, ToolMessage
from ..core.model.compaction import MessageGroup
from ..core.tokenizer import TokenCounter, default_counter
from ..errors.middleware import StopRun
from ..types.completions import Usage
from ..types.context import Context
from ..types.summary import Summary
from .clients import CoreLLMClient
from .component import ComponentBase, ConfigT

if t.TYPE_CHECKING:
    from ..core.middleware.chain import MiddlewareChain
    from ..types.run_context import RunContext

logger = logging.getLogger(__name__)
AnyClient = CoreLLMClient[t.Any]
# A caller that wants the messages that leave, while they are summarized.
Leaving = t.Callable[[list[Message]], t.Awaitable[None]]


class CompactionResult(BaseModel):
    """What ``compact`` gives back."""

    messages: list[Message]  # the conversation to go on with
    summary: Summary | None = None  # what is kept of the messages that left
    changed: bool = False  # something was shortened or removed
    removed: int = 0  # messages that left
    usage: Usage = Field(default_factory=Usage)  # what keeping the summary cost
    context: Context = Field(default_factory=Context)  # the window, part by part, after it
    tokens_before: int = 0
    tokens_after: int = 0


class WindowCompactionConfig(BaseModel):
    threshold: float = Field(default=0.8, gt=0, le=1)  # part of the room the messages may fill
    keep: float = Field(default=0.1, gt=0, lt=1)  # part of it the newest messages keep after
    max_result_chars: int = Field(default=2000, gt=0)  # an old tool result longer than this is cut
    keep_groups: int = Field(default=2, gt=0)  # last exchanges never touched


class CoreCompaction(ComponentBase[ConfigT], ABC):
    """Implement once per way of keeping what leaves: nothing, a summary.

    ``compact`` does only what is needed, in order: cut the old tool results
    that are long, then take the old messages out and ask ``reduce`` what
    is kept of them. Messages in, messages and a summary out: it writes no
    message of its own, and where the summary is shown is the caller's choice.
    """

    component_type = "compaction"

    CUT: t.ClassVar[str] = "[result cut to make room: {left} more characters removed]"

    def __init__(
        self,
        threshold: float = 0.8,
        keep: float = 0.1,
        max_result_chars: int = 2000,
        keep_groups: int = 2,
        counter: TokenCounter | None = None,
    ) -> None:
        # The room is the window without the answer, the system prompt and
        # the tools. The messages may fill ``threshold`` of it (the rest is
        # margin); after a compaction the newest that fit in ``keep`` stay.
        self.threshold = threshold
        self.keep = keep
        self.max_result_chars = max_result_chars  # an old tool result longer than this is cut
        self.keep_groups = keep_groups  # last exchanges never touched
        self.counter = counter or default_counter()

    # -------- EACH WAY -----------------------------------------------------------
    @abstractmethod
    async def reduce(
        self,
        old: list[Message],
        client: AnyClient,
        summary: Summary | None,
        ctx: RunContext | None = None,
        middleware: MiddlewareChain | None = None,
    ) -> tuple[Summary | None, Usage]:
        """What is kept of ``old``, the messages that leave, and what it
        cost; ``summary`` is what was kept before and must not be lost.
        A model call made here takes ``ctx`` and ``middleware``, so it is seen."""

    # -------- FOR EVERY WAY ------------------------------------------------------
    def measure(
        self, messages: list[Message], client: AnyClient, prompt_tokens: int = 0,
        tool_tokens: int = 0,
    ) -> Context:
        """The model's window, part by part, for a call with ``messages``."""
        config = client.config
        window, output = config.max_context_window, config.max_output_tokens
        room = max(0, window - output - prompt_tokens - tool_tokens)
        return Context(
            window=window, prompt=prompt_tokens, tools=tool_tokens, output=output,
            messages=self.counter.count_messages(messages),
            margin=room - int(self.threshold * room),
        )

    def needed(
        self, messages: list[Message], client: AnyClient, prompt_tokens: int = 0,
        tool_tokens: int = 0,
    ) -> bool:
        """The messages passed what they may take."""
        context = self.measure(messages, client, prompt_tokens, tool_tokens)
        return context.messages > context.capacity

    async def compact(
        self,
        messages: list[Message],
        client: AnyClient,
        *,
        summary: Summary | None = None,
        prompt_tokens: int = 0,
        tool_tokens: int = 0,
        force: bool = False,
        ctx: RunContext | None = None,
        middleware: MiddlewareChain | None = None,
        leaving: Leaving | None = None,
    ) -> CompactionResult:
        """Make ``messages`` fit beside what else goes in the call (the
        system prompt and the tools, in tokens). ``summary``: what earlier
        compactions kept. ``force`` compacts even under the threshold.
        ``leaving`` gets the messages that leave, while they are summarized."""
        context = self.measure(messages, client, prompt_tokens, tool_tokens)
        limit, before = context.capacity, context.messages
        result = CompactionResult(
            messages=messages, summary=summary, tokens_before=before, tokens_after=before,
            context=context,
        )
        if not force and before <= limit:
            return result

        # 1. Old tool results that are long are cut; that may be enough.
        groups = self.groups(self.cut(messages))
        cut = self.flat(groups)
        result.changed = cut != messages
        if not force and self.counter.count_messages(cut) <= limit:
            return self.done(result, cut)

        # 2. The old messages leave, whole exchanges only; the recent ones stay.
        #    By force, at most half of what there is stays: the window was wrong.
        budget = int(self.keep * limit)
        if force:
            budget = min(budget, before // 2)
        left, stay = split_recent_messages(
            groups, max_tokens=budget, min_keep_groups=self.keep_groups
        )
        old, recent = t.cast("list[Message]", left), t.cast("list[Message]", stay)
        if not old:
            return self.done(result, recent)

        # 3. What is kept of them, while whoever asked looks at them too;
        #    a failure in either never stops the agent.
        reduced, looked = await asyncio.gather(
            self.reduce(old, client, summary, ctx, middleware),
            leaving(old) if leaving is not None else asyncio.sleep(0),
            return_exceptions=True,
        )
        for outcome in (reduced, looked):
            # A middleware ended the turn: that is not a failure to swallow.
            if isinstance(outcome, (StopRun, asyncio.CancelledError)):
                raise outcome
        if isinstance(reduced, BaseException):
            logger.error(
                "Compaction could not reduce the old messages; they are dropped", exc_info=reduced
            )
        else:
            result.summary, result.usage = reduced
        if isinstance(looked, BaseException):
            logger.error("Who looked at the messages that leave failed", exc_info=looked)
        result.changed, result.removed = True, len(old)
        return self.done(result, recent)

    def done(self, result: CompactionResult, messages: list[Message]) -> CompactionResult:
        """``result`` with the messages that stay, counted."""
        result.messages = messages
        result.tokens_after = self.counter.count_messages(messages)
        result.context = result.context.model_copy(update={"messages": result.tokens_after})
        return result

    def groups(self, messages: list[Message]) -> list[MessageGroup]:
        """``messages`` in whole exchanges: a tool call is never apart from its answers."""
        return group_atomic_messages(t.cast("list[CoreMessage]", messages), self.counter)

    @staticmethod
    def flat(groups: list[MessageGroup]) -> list[Message]:
        return t.cast("list[Message]", [m for group in groups for m in group.messages])

    def cut(self, messages: list[Message]) -> list[Message]:
        """``messages`` with the long tool results cut, except in the last exchanges."""
        groups = self.groups(messages)
        fresh = {id(m) for group in groups[-self.keep_groups :] for m in group.messages}
        limit = self.max_result_chars
        done: list[Message] = []
        for message in messages:
            content = message.content
            long = isinstance(message, ToolMessage) and isinstance(content, str) and len(content) > limit
            if long and id(message) not in fresh:
                assert isinstance(content, str)
                note = self.CUT.format(left=len(content) - limit)
                message = message.model_copy(update={"content": f"{content[:limit]}\n{note}"})
            done.append(message)
        return done


class WindowCompaction(CoreCompaction[WindowCompactionConfig]):
    """Keeps the recent messages and drops the old ones. No model call: it
    is what a loop uses when no other compaction is given."""

    component_schema = WindowCompactionConfig
    component_provider_override = "max_ai.base.compaction.WindowCompaction"

    async def reduce(
        self,
        old: list[Message],
        client: AnyClient,
        summary: Summary | None,
        ctx: RunContext | None = None,
        middleware: MiddlewareChain | None = None,
    ) -> tuple[Summary | None, Usage]:
        return summary, Usage()

    def _to_config(self) -> WindowCompactionConfig:
        return WindowCompactionConfig(
            threshold=self.threshold, keep=self.keep, max_result_chars=self.max_result_chars,
            keep_groups=self.keep_groups,
        )

    @classmethod
    def _from_config(cls, config: WindowCompactionConfig) -> t.Self:
        return cls(config.threshold, config.keep, config.max_result_chars, config.keep_groups)
