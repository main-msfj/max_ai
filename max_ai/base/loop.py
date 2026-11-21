"""The loop contract: one turn of the agent, from the conversation to the final answer."""

from __future__ import annotations

import asyncio
import json
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from ..core.event_type import (
    CompactionEvent,
    CoreEvent,
    LoopNoticeEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelRetryEvent,
    ModelStreamChunkEvent,
)
from ..core.ids import short_id
from ..core.messages import (
    HARNESS_SOURCE,
    HARNESS_TAG,
    AssistantMessage,
    Message,
    ToolMessage,
    UserMessage,
)
from ..core.retry import RetryPolicy
from ..core.tool import ToolDispatcher
from ..types.completions import Usage
from ..types.context import Context
from ..types.run_context import RunContext
from ..types.tool_call import ToolCallRecord, ToolResult
from .clients import CoreLLMClient
from .compaction import CoreCompaction, WindowCompaction
from .component import ComponentBase, ConfigT

if t.TYPE_CHECKING:
    from ..core.middleware.chain import MiddlewareChain

AnyClient = CoreLLMClient[t.Any]


class LoopState(BaseModel):
    """What one run of a loop counts. Pass your own to ``run`` to read it after."""

    iteration: int = 0  # model calls made in this run
    usage: Usage = Field(default_factory=Usage)  # sum of every model call
    finish_reason: str = "unknown"  # the model's last reason, or why the loop stopped
    message: AssistantMessage | None = None  # the model's last message
    limit: int = 0  # model calls allowed in this run
    notices: dict[str, int] = Field(default_factory=dict[str, int])  # times each rule spoke
    compactions: int = 0  # times the conversation was compacted in this run
    last_calls: str = ""  # the tool calls of the last round, as text
    repeats: int = 0  # rounds in a row with those same calls

    @property
    def text(self) -> str:
        """The answer as text; empty when the turn ended without one."""
        return self.message.text() if self.message else ""


class Stop(BaseModel):
    """A hook's order to end the turn. ``answer`` is given to the user as
    the agent's answer."""

    reason: str
    answer: str | None = None


class Continue(BaseModel):
    """A hook's order to go on with the turn without telling the model anything."""


class Note(BaseModel):
    """A hook's order to tell the model something. The turn goes on; tools
    the model asked for still run, and the note is told after them."""

    text: str
    rule: str = "hook"  # the rule that speaks: counted in ``state.notices``

    def __init__(self, text: str, rule: str = "hook") -> None:
        super().__init__(text=text, rule=rule)


class Refuse(BaseModel):
    """A hook's order not to run the tools the model asked for: each one
    gets ``text`` as its answer."""

    text: str
    rule: str = "hook"  # the rule that speaks: counted in ``state.notices``

    def __init__(self, text: str, rule: str = "hook") -> None:
        super().__init__(text=text, rule=rule)


class Wrap(BaseModel):
    """A hook's order to close the turn: the model is told ``text`` and gets
    one call more, without tools, to answer the user."""

    text: str
    reason: str  # the turn's finish reason


Decision = Note | Refuse | Wrap | Stop | Continue | None


class LoopHook:
    """What a loop asks at each moment of a turn. Inherit and write only the
    moments you need; as it is, the usual rules.

    Each moment answers None (go on as usual), ``Note`` (told to the
    model), ``Refuse`` (the tools it asked for do not run and get that text
    as their answer), ``Continue`` (the turn goes on, nothing is told),
    ``Wrap`` (the model gets one call more, without tools, to answer) or
    ``Stop`` (the turn ends)."""

    # What the model is told. Replace a text to say it another way.
    EMPTY_ANSWER: t.ClassVar[str] = (
        "Your last answer arrived empty. Answer the user, or call a tool."
    )
    NEAR_LIMIT: t.ClassVar[str] = (
        "You have {left} model calls left in this turn. Finish what is essential and "
        "then answer saying what you did and what is missing."
    )
    LAST_ROUND: t.ClassVar[str] = (
        "One tool round is left in this turn. After it, please give the final answer."
    )
    NO_ROUNDS: t.ClassVar[str] = (
        "The turn has no tool rounds left. Please give the final answer now, saying what "
        "was done and what is missing."
    )
    OPEN_TODOS: t.ClassVar[str] = (
        "{open} items of your list are still open. Finish them, or update the list if "
        "they no longer apply, and then answer."
    )
    REPEATED: t.ClassVar[str] = (
        "Not run: this is the same call as before and it gave the same result. "
        "Read that result and try something different."
    )
    ARGUMENTS_CUT: t.ClassVar[str] = (
        "Not run: your answer was cut at the output limit, so the arguments of this "
        "call arrived incomplete. Send less content in each call."
    )
    # What the user reads when the provider refused the request, and what is
    # left of a tool result that may have caused it. Neither names what was
    # found: said again in the conversation, it could be refused again.
    BLOCKED: t.ClassVar[str] = (
        "The provider blocked this request because of its content, so I could not answer. "
        "What was just sent was taken out of the conversation; please say it another way."
    )
    REMOVED: t.ClassVar[str] = "[removed: the provider blocked a request that held this content]"
    # What the user reads when a rule ends the turn.
    ANSWER_BLOCKED: t.ClassVar[str] = (
        "The provider blocked the answer because of its content, so there is nothing to "
        "show. Please ask in another way."
    )
    STUCK: t.ClassVar[str] = (
        "I kept making the same tool call with the same result, so I stopped. Tell me how "
        "to go on, or ask in another way."
    )
    NO_ANSWER: t.ClassVar[str] = (
        "The model answered with nothing twice, so the turn ended. Please try again."
    )
    GIVE_UP_AFTER: t.ClassVar[int] = 2  # rounds refused as repeated before the turn ends

    def __init__(self, warn_before: int = 3, max_repeats: int = 3) -> None:
        self.warn_before = warn_before  # calls left when the model is told; 0 = never
        self.max_repeats = max_repeats  # same calls in a row before they stop running

    async def leaving(
        self,
        ctx: RunContext,
        old: list[Message],
        client: AnyClient,
        middleware: MiddlewareChain | None,
    ) -> None:
        """Messages are about to leave the conversation: the last chance to
        keep something of them. It runs while they are summarized. As it
        is, nothing."""

    async def after_compact(self, ctx: RunContext, state: LoopState) -> None:
        """The loop compacted the conversation: renew what the model no
        longer sees (the memory index), save ``ctx``."""

    async def notes(self, ctx: RunContext, state: LoopState) -> list[str]:
        """Called before every model call: what happened outside since the
        last one (a task that ended, a result that arrived). Each text is
        told to the model. As it is, nothing."""
        return []

    async def before_model(self, ctx: RunContext, state: LoopState) -> Decision:
        """Before each model call."""
        # 1. What the user wrote while the turn ran enters the conversation.
        self.take_inbox(ctx)

        # 2. The turn reached its limit: one call more, without tools, to answer.
        if state.iteration >= state.limit:
            return Wrap(text=self.NO_ROUNDS, reason="max_iterations")

        # 3. Few calls left: the model is told once, and once more at the last one.
        left = state.limit - state.iteration
        if left == self.warn_before and state.limit > left and not state.notices.get("near_limit"):
            return Note(self.NEAR_LIMIT.format(left=left), rule="near_limit")
        if left == 1 and state.limit > 1 and not state.notices.get("last_round"):
            return Note(self.LAST_ROUND, rule="last_round")
        return None

    async def after_model(self, ctx: RunContext, state: LoopState) -> Decision:
        """The model answered (``state.message``, ``state.finish_reason``)."""
        message = state.message
        if message is None:
            return None
        # 1. The provider blocked the answer: asking again gives the same.
        if state.finish_reason == "content_filter":
            self.drop_blocked_answer(ctx, message)
            return Stop(reason="content_filter", answer=self.ANSWER_BLOCKED)
        # 2. It asked for tools: cut arguments and repeated calls are not run.
        if message.tool_calls:
            if state.finish_reason == "length":
                return Refuse(self.ARGUMENTS_CUT, rule="arguments_cut")
            if state.repeats >= self.max_repeats + self.GIVE_UP_AFTER:
                return Stop(reason="repetition", answer=self.STUCK)
            if state.repeats >= self.max_repeats:
                return Refuse(self.REPEATED, rule="repetition")
            return None
        # 3. The user wrote while the model answered: the turn goes on with that.
        if self.take_inbox(ctx):
            return Continue()
        # 4. An empty answer is asked again once; then the turn ends.
        if not message.text().strip():
            if state.notices.get("empty_answer"):
                return Stop(reason="empty_answer", answer=self.NO_ANSWER)
            return Note(self.EMPTY_ANSWER, rule="empty_answer")
        # 5. It wants to end with items of its list still open: reminded once.
        still_open = sum(1 for todo in ctx.todos if todo.status != "done")
        if still_open and not state.notices.get("open_todos"):
            return Note(self.OPEN_TODOS.format(open=still_open), rule="open_todos")
        return None

    async def after_tools(self, ctx: RunContext, state: LoopState) -> None:
        """A round of tools has its answers: the moment to save ``ctx``."""

    async def on_error(self, ctx: RunContext, state: LoopState, error: Exception) -> Stop | None:
        """The turn failed. Return ``Stop`` to end it with an answer for the
        user; None lets the error go up."""
        # The provider refused the request for its content: the turn ends
        # clean, and what caused it leaves so the next turn is not refused too.
        if getattr(error, "kind", None) != "content_filter":
            return None
        self.remove_blocked(ctx)
        return Stop(reason="content_filter", answer=self.BLOCKED)

    @staticmethod
    def drop_blocked_answer(ctx: RunContext, message: AssistantMessage) -> None:
        """Take the answer the provider blocked out of the conversation: it
        is empty or cut. One that asked for tools stays, for their answers."""
        if not message.tool_calls and ctx.messages and ctx.messages[-1] is message:
            ctx.messages.pop()

    @classmethod
    def remove_blocked(cls, ctx: RunContext) -> None:
        """Take out what entered the conversation since the model last
        answered: only that can be new to the provider. What the user wrote
        leaves; a tool result keeps its place, without its content."""
        # 1. Where the model last answered.
        last = max(
            (n for n, message in enumerate(ctx.messages) if isinstance(message, AssistantMessage)),
            default=-1,
        )

        # 2. After it: tool results are emptied, the rest leaves.
        kept = ctx.messages[: last + 1]
        for message in ctx.messages[last + 1 :]:
            if isinstance(message, ToolMessage):
                kept.append(message.model_copy(update={"content": cls.REMOVED}))
        ctx.messages[:] = kept

    @staticmethod
    def take_inbox(ctx: RunContext) -> int:
        """Move what the user wrote during the turn (``ctx.inbox``) into the
        conversation, as their own messages. How many there were."""
        taken = len(ctx.inbox)
        while ctx.inbox:
            ctx.messages.append(UserMessage(source=ctx.user_id, content=ctx.inbox.pop(0)))
        return taken


class CoreLoop(ComponentBase[ConfigT], ABC):
    """Implement once per way of reasoning: ReAct, Ralph, plan and execute.

    A loop holds only its settings; what one run needs (conversation,
    client, tools) arrives in ``run``, so one loop serves many runs at once.
    """

    component_type = "loop"
    SOURCE: t.ClassVar[str] = "loop"  # who the events say they come from
    # The hook ends the turn at its limit; if a hook forgets, the loop does
    # it this many calls later.
    EXTRA_CALLS: t.ClassVar[int] = 5

    # How the conversation is kept inside the model's window. A loop always
    # has one: without a choice, the one that drops the old messages.
    compaction: CoreCompaction[t.Any] = WindowCompaction()

    # What the system prompt says after messages left the conversation.
    REMOVED: t.ClassVar[str] = "Earlier messages of this conversation were removed to make room."
    SUMMARY: t.ClassVar[str] = "This is the summary of what they held:"
    POSITION: t.ClassVar[str] = "Where things were at that moment:"

    # What a call gets when it is closed without running.
    NOT_ANSWERED: t.ClassVar[str] = (
        "The user wrote something else instead of answering. Do not take this as permission."
    )
    INTERRUPTED: t.ClassVar[str] = "The call was interrupted before it ran; its result is unknown."
    TURN_ENDED: t.ClassVar[str] = "Not run: the turn ended before this call."
    # What the model is told to get its final answer in the format asked.
    FORMAT_ANSWER: t.ClassVar[str] = (
        "Give your final answer again, now in the required format. Only the format, no tools."
    )
    INVALID_FORMAT: t.ClassVar[str] = (
        "Your answer does not follow the required format: {error}. Send it again, corrected."
    )

    # -------- EACH LOOP ----------------------------------------------------------
    @abstractmethod
    def run(
        self,
        ctx: RunContext,
        prompt: str,
        client: AnyClient,
        dispatcher: ToolDispatcher,
        *,
        stream: bool = False,
        hook: LoopHook | None = None,
        state: LoopState | None = None,
        output_format: type[BaseModel] | None = None,
        retry: RetryPolicy | None = None,
    ) -> t.AsyncGenerator[CoreEvent]:
        """Run one turn over ``ctx``: every message produced is added to
        ``ctx.messages``, and what happens goes out as events."""

    # -------- FOR EVERY LOOP -----------------------------------------------------
    async def call_model(
        self,
        ctx: RunContext,
        prompt: str,
        client: AnyClient,
        dispatcher: ToolDispatcher,
        state: LoopState,
        *,
        output_format: type[BaseModel] | None = None,
        stream: bool = False,
        retry: RetryPolicy | None = None,
        tools: bool = True,  # False: the model can only answer
    ) -> t.AsyncGenerator[CoreEvent]:
        """One model call. Its answer is added to ``ctx.messages`` and left
        in ``state.message``; a call that is cut adds nothing."""
        # 1. What the model sees: the conversation and the loaded tools.
        messages = client.format_messages(ctx, self.system(ctx, prompt))
        given = dispatcher.registry.definitions() if tools else []
        yield ModelCallEvent(source=self.SOURCE, model=client.model, input_messages=messages)

        # 2. Each try that fails is told when the call gives the control back.
        retries: list[ModelRetryEvent] = []
        attempts = (retry or client.retry).attempts

        def on_retry(attempt: int, error: BaseException, wait: float) -> None:
            reason = str(getattr(error, "kind", type(error).__name__))
            retries.append(ModelRetryEvent(
                source=self.SOURCE, attempt=attempt, max_attempts=attempts, delay=wait,
                reason=reason,
            ))

        # 3. The call: piece by piece, or whole.
        message, usage, finish = None, Usage(), None
        if stream:
            pieces = client.stream(
                messages, given, output_format, retry=retry, on_retry=on_retry,
                ctx=ctx, middleware=dispatcher.middleware,
            )
            async for chunk in pieces:
                for event in self.drain(retries):
                    yield event
                if chunk.content or chunk.thinking:
                    yield ModelStreamChunkEvent(
                        source=self.SOURCE, chunk=chunk.content, thinking=chunk.thinking,
                    )
                if chunk.is_complete:
                    message, usage, finish = chunk.message, chunk.usage or usage, chunk.finish_reason
        else:
            result = await client.run(
                messages, given, output_format, retry=retry, on_retry=on_retry,
                ctx=ctx, middleware=dispatcher.middleware,
            )
            message, usage, finish = result.message, result.usage, result.finish_reason
        for event in self.drain(retries):
            yield event
        if message is None:
            raise RuntimeError(f"{type(client).__name__} ended its stream without a message")

        # 4. Each tool call needs an id of its own in this conversation.
        used = set(ctx.tool_state.records) | {
            call.id for m in ctx.messages if isinstance(m, AssistantMessage) for call in m.tool_calls
        }
        for call in message.tool_calls:
            if call.id in used:
                call.id = short_id()
            used.add(call.id)

        # 5. The answer enters the conversation.
        ctx.messages.append(message)
        state.message, state.usage = message, state.usage + usage
        state.finish_reason = finish or "stop"
        yield ModelResponseEvent(
            source=self.SOURCE, response=message.text(),
            has_tool_calls=bool(message.tool_calls), usage=usage,
        )

    async def run_tools(
        self, ctx: RunContext, dispatcher: ToolDispatcher, records: list[ToolCallRecord]
    ) -> t.AsyncGenerator[CoreEvent]:
        """Run ``records``; each answer is added to ``ctx.messages``. A call
        that waits for a person is left open: ``waiting`` tells."""
        try:
            async for item in dispatcher.dispatch_many(records, ctx):
                if isinstance(item, ToolMessage):
                    self.place(ctx, item)
                else:
                    yield item
        except (asyncio.CancelledError, GeneratorExit):
            # The turn was stopped: every call gets its answer, so the
            # conversation can go on.
            dispatcher.close_interrupted(ctx)
            dispatcher.answer_unanswered(ctx, records)
            raise

    @staticmethod
    def new_records(ctx: RunContext, message: AssistantMessage) -> list[ToolCallRecord]:
        """A record in ``ctx.tool_state`` for each tool call of ``message``."""
        records: list[ToolCallRecord] = []
        for call in message.tool_calls:
            record = ToolCallRecord(
                id=call.id, tool_name=call.tool_name, parameters=call.parameters,
                session_id=ctx.session_id,
            )
            records.append(ctx.tool_state.add(record))
        return records

    @staticmethod
    def open_records(ctx: RunContext) -> list[ToolCallRecord]:
        """The calls without an answer yet: they wait for a person, or an
        earlier run left them."""
        return [record for record in ctx.tool_state.records.values() if not record.is_consumed]

    @staticmethod
    def waiting(records: list[ToolCallRecord]) -> bool:
        """Some of ``records`` still waits for a person."""
        return any(not record.is_consumed for record in records)

    # -------- THE CONVERSATION STAYS VALID ---------------------------------------
    @staticmethod
    def place(ctx: RunContext, answer: ToolMessage) -> None:
        """Put a tool's answer right after the message that asked for it: a
        provider refuses an answer that comes later."""
        messages = ctx.messages
        for index in range(len(messages) - 1, -1, -1):
            asked = messages[index]
            if isinstance(asked, AssistantMessage) and any(
                call.id == answer.tool_call_id for call in asked.tool_calls
            ):
                # After the answers that message already has.
                index += 1
                while index < len(messages) and isinstance(messages[index], ToolMessage):
                    index += 1
                messages.insert(index, answer)
                return
        messages.append(answer)

    def close(self, ctx: RunContext, record: ToolCallRecord, reason: str) -> None:
        """Answer ``record`` with ``reason`` without running it."""
        result = ToolResult.tool_failure(record.id, reason)
        record.force_consume(result)
        self.place(ctx, ToolDispatcher.tool_message(record, result))

    def settle(self, ctx: RunContext) -> None:
        """Before a turn: every tool call of the conversation has its answer
        or a record that will give it."""
        # 1. A call with no record was lost when a run died: its result is unknown.
        answered = {m.tool_call_id for m in ctx.messages if isinstance(m, ToolMessage)}
        for message in list(ctx.messages):
            if not isinstance(message, AssistantMessage):
                continue
            for call in message.tool_calls:
                if call.id not in answered and ctx.tool_state.get(call.id) is None:
                    record = ToolCallRecord(
                        id=call.id, tool_name=call.tool_name, parameters=call.parameters,
                        session_id=ctx.session_id,
                    )
                    self.close(ctx, ctx.tool_state.add(record), self.INTERRUPTED)
        # 2. The user wrote instead of approving or answering: those calls are closed.
        if ctx.messages and isinstance(ctx.messages[-1], UserMessage):
            for record in self.open_records(ctx):
                if record.is_pending_approval or record.is_awaiting_input:
                    self.close(ctx, record, self.NOT_ANSWERED)

    # -------- WHAT THE HOOK DECIDES ----------------------------------------------
    def note(self, ctx: RunContext, state: LoopState, said: Note | Refuse) -> LoopNoticeEvent:
        """Tell the model what a rule said, as a harness message; the rule is counted."""
        ctx.messages.append(self.harness(said.text))
        state.notices[said.rule] = state.notices.get(said.rule, 0) + 1
        return self.notice(said.rule, said.text)

    def refuse(
        self, ctx: RunContext, state: LoopState, records: list[ToolCallRecord], said: Refuse
    ) -> LoopNoticeEvent:
        """Answer every call of ``records`` with what a rule said, without
        running it; the rule is counted."""
        for record in records:
            self.close(ctx, record, said.text)
        state.notices[said.rule] = state.notices.get(said.rule, 0) + 1
        return self.notice(said.rule, said.text)

    def notice(self, rule: str, text: str = "") -> LoopNoticeEvent:
        """The event that says a rule spoke: which one, and what it said."""
        return LoopNoticeEvent(source=self.SOURCE, kind=rule, message=text)

    async def wrap(
        self,
        ctx: RunContext,
        prompt: str,
        client: AnyClient,
        dispatcher: ToolDispatcher,
        state: LoopState,
        hook: LoopHook,
        said: Wrap,
        **options: t.Any,
    ) -> t.AsyncGenerator[CoreEvent]:
        """Close the turn as the hook ordered: the model is told why, and
        answers the user in one call without tools."""
        yield self.note(ctx, state, Note(said.text, rule=said.reason))
        state.iteration += 1
        async for event in self.ask_model(
            ctx, prompt, client, dispatcher, state, hook, tools=False, **options
        ):
            yield event
        # A model that asks for tools anyway gets them closed: none was offered.
        if state.message is not None:
            for record in self.new_records(ctx, state.message):
                self.close(ctx, record, self.TURN_ENDED)
        state.finish_reason = said.reason

    def end(self, ctx: RunContext, state: LoopState, stop: Stop) -> LoopNoticeEvent:
        """End the turn as the hook ordered; its answer becomes the agent's."""
        state.finish_reason = stop.reason
        if stop.answer:
            state.message = AssistantMessage(source=HARNESS_SOURCE, content=stop.answer)
            ctx.messages.append(state.message)
        return self.notice(stop.reason, stop.answer or "")

    async def ask_model(
        self,
        ctx: RunContext,
        prompt: str,
        client: AnyClient,
        dispatcher: ToolDispatcher,
        state: LoopState,
        hook: LoopHook,
        **options: t.Any,
    ) -> t.AsyncGenerator[CoreEvent]:
        """One model call, after the notes from outside and with the
        conversation compacted if needed; if the model still refuses it as
        too long, it is compacted by force and the call is made once more."""
        for text in await hook.notes(ctx, state):
            yield self.note(ctx, state, Note(text, rule="note"))
        if event := await self.compact(ctx, prompt, client, dispatcher, state, hook):
            yield event
        try:
            async for event in self.call_model(ctx, prompt, client, dispatcher, state, **options):
                yield event
        except Exception as error:
            if getattr(error, "kind", None) != "token_limit":
                raise
            forced = await self.compact(ctx, prompt, client, dispatcher, state, hook, force=True)
            if forced is None:
                raise
            yield forced
            async for event in self.call_model(ctx, prompt, client, dispatcher, state, **options):
                yield event

    async def compact(
        self,
        ctx: RunContext,
        prompt: str,
        client: AnyClient,
        dispatcher: ToolDispatcher,
        state: LoopState,
        hook: LoopHook,
        force: bool = False,
    ) -> CompactionEvent | None:
        """Make the conversation fit and leave ``ctx`` in order, with the
        window's parts in ``ctx.context``. The event that tells a
        compaction, or None when nothing changed."""
        # 1. What else goes in the call: the system prompt and the tools.
        count = self.compaction.counter
        tools = dispatcher.registry.tokens(count)
        # 2. Compact; what it kept and what it cost stay in ctx and in the run.
        result = await self.compaction.compact(
            ctx.messages, client, summary=ctx.summary,
            prompt_tokens=count.count_text(self.system(ctx, prompt)), tool_tokens=tools,
            force=force, ctx=ctx, middleware=dispatcher.middleware,
            leaving=lambda old: hook.leaving(ctx, old, client, dispatcher.middleware),
        )
        ctx.context = result.context
        if not result.changed:
            return None
        ctx.messages[:] = result.messages
        ctx.summary = result.summary
        state.usage = state.usage + result.usage
        state.compactions += 1
        ctx.compaction.compactions += 1  # the session's own count, kept with it
        # 3. Messages left: where the agent is goes to the system prompt, the
        #    model forgot the files it read, and the closed calls are not needed.
        if result.removed:
            ctx.position = self.position(ctx)
            self.measure(ctx, prompt, client, dispatcher)
            ctx.files_seen.clear()
            records = ctx.tool_state.records
            for call_id in [key for key, record in records.items() if record.is_consumed]:
                del records[call_id]
        await hook.after_compact(ctx, state)
        return CompactionEvent(
            source=self.SOURCE, strategy=type(self.compaction).__name__,
            changed=True, pruned_only=not result.removed, tokens_before=result.tokens_before,
            tokens_after=result.tokens_after, kept_message_count=len(result.messages),
            summary=result.summary.text() if result.summary else None, usage=result.usage,
            removed=result.removed, summary_failed=result.summary_failed,
        )

    def measure(
        self, ctx: RunContext, prompt: str, client: AnyClient, dispatcher: ToolDispatcher
    ) -> Context:
        """Share out the window with what is loaded now and keep it in
        ``ctx.context``. Nothing is compacted and no model is called, so it
        can run when a session opens, before the first message."""
        # 1. The tools are counted by their registry; the prompt here, because
        #    the summary and the position are part of it.
        count = self.compaction.counter
        words = count.count_text(self.system(ctx, prompt))
        # 2. Share out the window.
        ctx.context = self.compaction.measure(
            ctx.messages, client, words, dispatcher.registry.tokens(count)
        )
        return ctx.context

    def system(self, ctx: RunContext, prompt: str) -> str:
        """The system prompt of a call: ``prompt``, then what compaction
        kept of the messages that left, and where the agent was then."""
        if ctx.summary is None and ctx.position is None:
            return prompt
        parts = [prompt, self.REMOVED]
        if ctx.summary is not None and ctx.summary.text():
            parts.append(f"{self.SUMMARY}\n\n{ctx.summary.text()}")
        if ctx.position:
            parts.append(f"{self.POSITION}\n{ctx.position}")
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def position(ctx: RunContext) -> str:
        """Where the agent is and what it has left, for after a compaction."""
        lines: list[str] = []
        if ctx.cwd:
            lines.append(f"Current folder: {ctx.cwd}")
        if ctx.todos:
            lines.append("Your list:")
            lines += [f"- [{todo.status}] {todo.content}" for todo in ctx.todos]
        return "\n".join(lines)

    async def format_answer(
        self,
        ctx: RunContext,
        prompt: str,
        client: AnyClient,
        state: LoopState,
        output_format: type[BaseModel],
        retry: RetryPolicy | None = None,
        middleware: MiddlewareChain | None = None,
    ) -> t.AsyncGenerator[CoreEvent]:
        """The model's final answer in ``output_format``, left in
        ``state.message.structured_output``. One call more, without tools;
        nothing is added to the conversation."""
        answer = state.message
        if answer is None:
            return
        system = self.system(ctx, prompt)
        asked: list[Message] = [*client.format_messages(ctx, system), self.harness(self.FORMAT_ANSWER)]
        for last in (False, True):
            # 1. Ask for the answer in the format.
            yield ModelCallEvent(source=self.SOURCE, model=client.model, input_messages=asked)
            result = await client.run(
                asked, None, output_format, retry=retry, ctx=ctx, middleware=middleware
            )
            state.usage = state.usage + result.usage
            text = result.message.text()
            yield ModelResponseEvent(source=self.SOURCE, response=text, usage=result.usage)
            # 2. It fits: the answer is kept with its structured form (a message
            #    is not changed: a copy takes its place).
            try:
                parsed = output_format.model_validate_json(text)
            except ValueError as error:
                problem = str(error)
            else:
                state.message = answer.model_copy(update={"structured_output": parsed})
                ctx.messages[:] = [state.message if m is answer else m for m in ctx.messages]
                return
            # 3. It does not: the model is told why and tries once more.
            yield self.notice("invalid_format", problem)
            if not last:
                asked += [result.message, self.harness(self.INVALID_FORMAT.format(error=problem))]
        state.finish_reason = "invalid_format"

    @staticmethod
    def harness(text: str) -> UserMessage:
        """``text`` as a message from the harness."""
        return UserMessage(source=HARNESS_SOURCE, content=f"{HARNESS_TAG} {text}")

    @staticmethod
    def repeated(state: LoopState, message: AssistantMessage) -> int:
        """Rounds in a row with these same tool calls, this one included."""
        calls = sorted(
            f"{call.tool_name} {json.dumps(call.parameters, sort_keys=True, default=str)}"
            for call in message.tool_calls
        )
        text = "\n".join(calls)
        state.repeats = state.repeats + 1 if text and text == state.last_calls else 1
        state.last_calls = text
        return state.repeats

    @staticmethod
    def drain(events: list[ModelRetryEvent]) -> list[ModelRetryEvent]:
        """The events kept so far; the list is left empty."""
        kept = list(events)
        events.clear()
        return kept
