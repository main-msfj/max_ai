"""
Abstract base class for reasoning loops.

A reasoning loop drives a single agent turn: call the LLM, dispatch
any tool calls, and decide whether to call the LLM again.

Lifecycle from the user's perspective:

  1. User constructs the loop with config-only kwargs (e.g.
     ``ReactLoop(max_loop_iterations=5)``). No client, no
     dispatcher, no middleware — those don't exist yet at config time.
  2. The agent calls ``loop.bind(name, client, dispatcher=...,
     tool_context=...)`` inside ``run()`` to wire the runtime context.
  3. The agent calls ``loop.execute_reasoning_loop(...)`` to drive the
     turn.

Loops are provider-agnostic — they consume a
``CoreChatCompletionClient`` via its ``run()`` entry point, so prompt
assembly and tool-schema conversion stay inside the client. Loops are
also transcript-agnostic: they read and write ``ctx.messages`` directly.
The agent is responsible for handing them a clean ``RunContext`` (no
orphaned tool calls / unanswered tool responses) before calling
``execute_reasoning_loop``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from ..core.compaction import TokenCounter
from ..core.event_type import (
    CompactionEvent,
    CoreEvent,
    ErrorEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelStreamChunkEvent,
)
from ..core.events_bus import EventBus
from ..core.messages import AssistantMessage, CoreMessage
from ..core.middleware.chain import MiddlewareChain
from ..core.termination import CancellationToken
from ..errors.client import ClientError
from ..loggers import ScopedLogger
from ..types.chat_history import ChatHistory
from ..types.completions import ChatCompletionChunk, ChatCompletionResult, Usage
from ..types.run_context import RunContext
from ..types.stacks import PromptCtx
from .clients import CoreChatCompletionClient
from .completion_gate import CompletionDecision
from .component import ComponentBase
from .middleware import MiddlewareContext, ModelRequest
from .tools import CoreTool, ToolContext

if t.TYPE_CHECKING:
    from ..core.messages import ToolCall
    from ..core.tool.dispatcher import ToolDispatcher
    from .compaction import CoreCompaction
    from .memory import CoreMemoryRegistry


logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["ReasoningLoop"])


# Kinds of ClientError that warrant a retry with exponential backoff.
# Everything else (auth, validation, token_limit, invalid_response, ...)
# fails immediately.
_TRANSIENT_KINDS: frozenset[str] = frozenset(
    {"rate_limit", "timeout", "stream_interrupted", "api_error", "provider"}
)


# -------- LOOP STATE -----------------------------------------------------------
class BaseLoopState(BaseModel):
    """Ephemeral metrics + last-result accumulator for a single run.

    Subclasses (one per reasoning strategy) extend this with whatever
    extra bookkeeping they need (e.g. ``ReActLoopState``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    iteration: int = Field(default=0)
    finish_reason: str = Field(default="unknown")
    start_time: float = Field(default_factory=time.monotonic)

    llm_calls: int = Field(default=0)
    tool_calls: int = Field(default=0)
    attempts_to_call_api: int = Field(default=0)
    tokens_input: int = Field(default=0)
    tokens_output: int = Field(default=0)
    tokens_cached: int = Field(default=0)

    # Last LLM result — the loop body needs this to read tool_calls
    # and finish_reason after an LLM call returns.
    last_result: ChatCompletionResult | None = Field(default=None)

    # Last CompletionDecision computed, whatever its status — Agent reads
    # this to populate AgentResponse.completion after the turn ends.
    last_completion_decision: CompletionDecision | None = Field(default=None)

    # Turn-scoped scratch space for loop guards (repetition counters,
    # fired-once flags). Fresh per turn because the loop state is.
    guard_state: dict[str, t.Any] = Field(default_factory=dict)

    @property
    def retries(self) -> int:
        return max(0, self.attempts_to_call_api - self.llm_calls)

    def record_usage(self, usage: Usage) -> None:
        """Accumulate usage from a ``ChatCompletionResult``."""
        self.llm_calls += usage.llm_calls
        self.attempts_to_call_api += usage.attempts_to_call_api
        self.tokens_input += usage.tokens_input
        self.tokens_output += usage.tokens_output
        self.tokens_cached += usage.tokens_cached

    def record_completion(self, result: ChatCompletionResult) -> None:
        """Capture a result and accumulate its usage."""
        self.last_result = result
        self.record_usage(result.usage)

    # Fields carried across a pause/resume boundary so a resumed run
    # keeps its iteration budget and usage accounting instead of
    # starting from zero every segment.
    _METRIC_FIELDS: t.ClassVar[tuple[str, ...]] = (
        "iteration",
        "llm_calls",
        "tool_calls",
        "attempts_to_call_api",
        "tokens_input",
        "tokens_output",
        "tokens_cached",
    )

    def metrics_snapshot(self) -> dict[str, int]:
        """Serializable counters to stash on the RunContext at a pause."""
        return {name: getattr(self, name) for name in self._METRIC_FIELDS}

    def apply_metrics(self, snapshot: dict[str, t.Any]) -> None:
        """Restore counters captured by ``metrics_snapshot``."""
        for name in self._METRIC_FIELDS:
            value = snapshot.get(name)
            if isinstance(value, int) and value >= 0:
                setattr(self, name, value)


# -------- BASE REASONING -----------------------------------------------------------
class ReasoningConfig(BaseModel):
    """A reasoning loop's settings. Concrete loops extend it."""

    max_connection_retries: int = Field(default=3, ge=0)


class BaseReasoning(ComponentBase[ReasoningConfig], ABC):
    """Abstract reasoning cycle.

    Concrete subclasses implement ``execute_reasoning_loop`` — the
    iterative body of one agent turn. They also declare
    ``LOOP_STATE_CLS`` to tell the agent which state object to
    instantiate at the start of each run.

    The base class provides:

    - ``bind(...)``: wire runtime dependencies that don't exist at
      construction time (client, executor, middleware chain, agent
      name). Called by the agent inside ``run()``.
    - ``_call_llm`` / ``_call_llm_stream``: shared LLM-call helpers
      with retry on transient ``ClientError``, cancellation handling,
      and middleware integration. Subclasses can reuse these or
      override them entirely.

    Users subclass this to define custom reasoning strategies. They
    only declare config-time arguments in ``__init__`` (and call
    ``super().__init__(max_connection_retries=...)``); runtime
    arguments are injected later via ``bind()``.

    It is a serializable component: a custom loop declares its
    ``component_schema`` (extending ``ReasoningConfig``) and implements
    ``_to_config``/``_from_config``, so an agent's JSON can reference it.
    """

    component_type = "reasoning"
    component_schema: t.ClassVar[type[BaseModel]] = ReasoningConfig

    # Each subclass declares its own loop-state class. The agent reads
    # this to instantiate the right state at the start of each run.
    LOOP_STATE_CLS: t.ClassVar[type[BaseLoopState]] = BaseLoopState

    def __init__(
        self, max_connection_retries: int = 3, enable_human_input: bool = True
    ) -> None:
        """Initialize config-only state.

        Runtime dependencies (``name``, ``client``, ``dispatcher``,
        ``middleware_chain``) are not arguments — the agent injects
        them later via ``bind()``. This keeps user-constructed
        reasoning instances free of agent internals.

        Args:
            max_connection_retries: Per-call retry budget for transient
                ``ClientError`` (rate_limit, timeout, etc.).
            enable_human_input: Enable LLm to user Elicitation question to user
        """
        self.max_connection_retries = max_connection_retries
        self.enable_human_input = enable_human_input

        # Runtime — populated by bind(). Accessing any of these before
        # bind() raises a clear error rather than silently passing None.
        self._name: str | None = None
        self._client: CoreChatCompletionClient | None = None
        self._dispatcher: ToolDispatcher | None = None
        self._tool_context: ToolContext | None = None
        self._middleware_chain: MiddlewareChain | None = None
        self._compaction: CoreCompaction | None = None
        self._memory: CoreMemoryRegistry | None = None
        self._max_context_tokens: int = 0
        self._completion_bus: EventBus | None = None

        self._current_loop_state: BaseLoopState | None = None

    def _set_loop_state(self, state: BaseLoopState) -> None:
        self._current_loop_state = state

    # -------- BIND -----------------------------------------------------------
    def bind(
        self,
        name: str,
        client: CoreChatCompletionClient,
        middleware_chain: MiddlewareChain | None = None,
        compaction: CoreCompaction | None = None,
        max_context_tokens: int = 0,
        *,
        dispatcher: ToolDispatcher | None = None,
        tool_context: ToolContext | None = None,
        completion_bus: EventBus | None = None,
        memory: CoreMemoryRegistry | None = None,
    ) -> t.Self:
        """Wire runtime dependencies. Called by the agent inside ``run()``.

        Returns ``self`` so the agent can chain
        ``loop.bind(...).execute_reasoning_loop(...)``.

        Calling ``bind`` twice on the same instance overwrites the
        previous wiring. The agent currently does this every run, so
        a single user-provided instance shared across multiple
        concurrent ``agent.run()`` calls is unsafe — to be revisited
        when concurrent runs become a concern.
        """
        if dispatcher is None:
            raise ValueError("A tool dispatcher is required")
        if tool_context is None:
            raise ValueError("Dispatcher binding requires a ToolContext")
        self._dispatcher = dispatcher
        self._tool_context = tool_context
        self._name = name
        self._client = client
        self._middleware_chain = (
            middleware_chain if middleware_chain is not None else MiddlewareChain()
        )
        self._compaction = compaction
        self._memory = memory
        self._max_context_tokens = max_context_tokens
        self._completion_bus = (
            completion_bus if completion_bus is not None else EventBus()
        )
        return self

    # -------- SESSION STATE & COMPACTION -------------------------------------------------
    def _refresh_session_state(self, ctx: RunContext, prompts: PromptCtx) -> None:
        """Keep the prompt's session state (compaction summary, plan) in sync
        with ``ctx``. Re-renders the stack only when one of them changed."""
        state = {
            "compaction_summary": (
                self._compaction.render(ctx.compaction.state)
                if self._compaction
                else None
            ),
            "current_plan": ctx.plan.as_text() if ctx.plan and ctx.plan.steps else None,
        }
        if all(prompts.variables.get(key) == value for key, value in state.items()):
            return
        prompts.variables.update(state)
        if prompts.stack is not None:
            for layer in prompts.stack:
                prompts.rendered_layers[type(layer)] = layer.render(prompts.variables)
        prompts.measure(self._prompt_counter)

    @property
    def _prompt_counter(self) -> TokenCounter:
        """Counter with the client's tokenizer, built once per loop."""
        tokenizer = getattr(
            getattr(self.client, "config", None), "tokenizer_base", None
        )
        counter = getattr(self, "_cached_prompt_counter", None)
        if counter is None or counter.tokenizer_base != tokenizer:
            counter = (
                TokenCounter(tokenizer_base=tokenizer) if tokenizer else TokenCounter()
            )
            self._cached_prompt_counter = counter
        return counter

    async def _compact_if_needed(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Run the strategy before a model call and apply its result to ``ctx``.

        A failing compaction never ends the turn: it is reported as a
        recoverable error and the call goes ahead with the window as it is.
        """
        from ..core.compaction import client_max_output_tokens

        compaction = self._compaction
        if compaction is None or not compaction.should_compact(
            ctx,
            prompts,
            self._max_context_tokens,
            client_max_output_tokens(self.client),
        ):
            return
        strategy = type(compaction).__name__
        yield CompactionEvent(source=self.name, phase="start", strategy=strategy)
        try:
            result = await compaction.compact(
                ctx=ctx,
                prompts=prompts,
                max_context_tokens=self._max_context_tokens,
                client=self.client,
                memory=self._memory,
            )
        except Exception as error:
            log.error("Compaction failed", exc=error, strategy=strategy)
            yield ErrorEvent(
                source=self.name,
                error_type="compaction_failed",
                error_message=f"{strategy}: {error}",
                is_recoverable=True,
            )
            return
        if result.changed:
            ctx.messages[:] = result.messages
            ctx.compaction.state = result.state
            ctx.compaction.compactions += 1
            ctx.compaction.archived_messages += len(result.old_messages)
            self._refresh_session_state(ctx, prompts)
        yield CompactionEvent(
            source=self.name,
            phase="end",
            strategy=strategy,
            changed=result.changed,
            pruned_only=result.pruned_only,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            kept_message_count=len(result.messages),
            old_messages=result.old_messages,
            summary=compaction.render(ctx.compaction.state),
        )

    @property
    def name(self) -> str:
        if self._name is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call .bind() first.")
        return self._name

    @property
    def client(self) -> CoreChatCompletionClient:
        if self._client is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call .bind() first.")
        return self._client

    @property
    def dispatcher(self) -> ToolDispatcher:
        if self._dispatcher is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call .bind() first.")
        return self._dispatcher

    @property
    def middleware_chain(self) -> MiddlewareChain:
        if self._middleware_chain is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call .bind() first.")
        return self._middleware_chain

    def _middleware_context(self, ctx: RunContext) -> MiddlewareContext:
        emit = self._tool_context.emit_event if self._tool_context else None
        return MiddlewareContext(
            ctx=ctx, agent=self.name, emit=emit or (lambda event: None)
        )

    @property
    def completion_bus(self) -> EventBus:
        if self._completion_bus is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call .bind() first.")
        return self._completion_bus

    @property
    def _tools(self) -> list[CoreTool]:
        """Tools currently available to the agent.

        Sourced from the executor each turn rather than cached, so a
        skill registry that mutates the catalog after ``prepare()`` is
        reflected in subsequent calls.
        """
        return list(self.tool_catalog.values())

    @property
    def tool_catalog(self) -> dict[str, CoreTool]:
        return {tool.name: tool for tool in self.dispatcher.registry.all_tools()}

    async def _execute_tools(
        self,
        ctx: RunContext,
        records: t.Any,
        cancellation_token: CancellationToken | None = None,
    ):
        from contextlib import aclosing

        async with aclosing(
            self.dispatcher.dispatch_many(
                records, self._tool_context, cancellation_token
            )
        ) as stream:
            async for item in stream:
                yield item

    @staticmethod
    def _conversation_messages(ctx: RunContext) -> list[CoreMessage]:
        """Return persisted history followed by the current live transcript."""
        return [
            *ctx.message_history.iter_messages(),
            *ctx.messages,
        ]

    @staticmethod
    def _without_assistant_thinking(
        messages: t.Sequence[CoreMessage],
    ) -> list[CoreMessage]:
        """Strip private assistant reasoning from model-facing messages."""
        cleaned: list[CoreMessage] = []
        for message in messages:
            if isinstance(message, AssistantMessage) and message.thinking:
                cleaned.append(message.model_copy(update={"thinking": None}))
                continue
            cleaned.append(message)
        return cleaned

    def _model_context(
        self,
        ctx: RunContext,
        transient_messages: t.Sequence[CoreMessage] | None = None,
    ) -> RunContext:
        """Return a copy safe to hand to chat-completion clients.

        ``transient_messages`` are appended for THIS call only — they are
        never written to ``ctx.messages``, so per-iteration steering
        (plan nudges, guard hints) doesn't pollute the durable transcript.
        """
        messages = self._without_assistant_thinking(ctx.messages)
        if transient_messages:
            messages = [*messages, *transient_messages]
        return ctx.model_copy(
            update={
                "message_history": ChatHistory(
                    message_history=self._without_assistant_thinking(
                        list(ctx.message_history.iter_messages())
                    )
                ),
                "messages": messages,
            }
        )

    def _model_input_messages(self, ctx: RunContext) -> list[CoreMessage]:
        return self._without_assistant_thinking(self._conversation_messages(ctx))

    def _input_messages_with_token_counts(
        self, messages: t.Sequence[CoreMessage]
    ) -> list[CoreMessage]:
        """Return model input messages with per-message token counts attached."""
        counter = TokenCounter(
            tokenizer_base=self.client.config.tokenizer_base,
        )
        counted: list["CoreMessage"] = []
        for message in messages:
            if message.token_count > 0:
                counted.append(message)
                continue
            counted.append(
                message.model_copy(
                    update={"token_count": counter.count_message(message)}
                )
            )
        return counted

    # -------- NON-STREAMING LLM CALL -----------------------------------------------------------
    async def _call_llm(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
        loop_state: BaseLoopState,
        cancellation_token: CancellationToken | None = None,
        output_format: type[BaseModel] | None = None,
        transient_messages: t.Sequence[CoreMessage] | None = None,
        tools_override: t.Sequence[CoreTool] | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Make one non-streaming LLM call through middleware.

        Yields any events produced by the middleware chain plus a
        ``ModelResponseEvent`` summarizing the assistant message.
        Captures the result into ``loop_state`` so the loop body can
        inspect it after the helper returns.

        ``tools_override``, when given, replaces the full tool catalog
        for THIS call only (e.g. restricting the model to ``ask_user``/
        ``update_plan`` right after a tool denial, so it can't reopen
        the approval chain it was just denied).

        Raises:
            asyncio.CancelledError: cancellation was requested.
            ClientError: non-transient, or transient after exhausting
                ``max_connection_retries``. An ``ErrorEvent`` is yielded
                immediately before the raise.
        """
        _log = log.child(
            run_id=ctx.run_id,
            session_id=ctx.session_id,
        )

        mw = self._middleware_context(ctx)
        request = await self.middleware_chain.model_request(
            mw,
            ModelRequest(
                messages=self._model_input_messages(ctx),
                tools=list(tools_override)
                if tools_override is not None
                else self._tools,
                options=dict(kwargs),
                output_format=output_format,
                model=str(getattr(self.client, "model", None) or "unknown"),
                model_config=getattr(self.client, "config", None),
            ),
        )
        model_name = request.model

        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()

        # clear any previous result
        loop_state.last_result = None

        async def _single_call() -> ChatCompletionResult:
            task = asyncio.create_task(
                self.client.run(
                    ctx=self._model_context(ctx, transient_messages),
                    prompts=prompts,
                    tools=request.tools,
                    output_format=request.output_format,
                    stream=False,
                    **request.options,
                )
            )
            if cancellation_token:
                cancellation_token.link_future(task)
            return t.cast(ChatCompletionResult, await task)

        yield ModelCallEvent(
            source=self.name,
            model=model_name,
            input_messages=self._input_messages_with_token_counts(
                self._model_input_messages(ctx)
            ),
            prompt_tokens=prompts.prompt_tokens,
        )

        backoff = 1.0
        for attempt in range(self.max_connection_retries + 1):
            try:
                result = await _single_call()
                break

            except asyncio.CancelledError:
                _log.info("LLM call cancelled by user request")
                raise

            except Exception as e:
                is_transient = isinstance(e, ClientError) and e.kind in _TRANSIENT_KINDS
                if not is_transient or attempt >= self.max_connection_retries:
                    recovered = await self.middleware_chain.model_error(mw, request, e)
                    if recovered is not None:
                        result = recovered
                        break
                    yield ErrorEvent(
                        source=self.name,
                        error_message=(
                            f"LLM call failed after {attempt + 1} attempt(s): {e}"
                        ),
                        error_type=type(e).__name__,
                        is_recoverable=False,
                    )
                    raise

                _log.warning(
                    "Transient LLM error",
                    attempt=attempt + 1,
                    max_attempts=self.max_connection_retries + 1,
                    error=str(e),
                    action="retry",
                )
                await asyncio.sleep(backoff)
                backoff *= 2

        result = await self.middleware_chain.model_response(mw, request, result)
        loop_state.record_completion(result)
        msg = result.message
        yield ModelResponseEvent(
            source=self.name,
            response=(
                msg.structured_output.model_dump_json()
                if msg.structured_output is not None
                else msg.text()
            ),
            has_tool_calls=bool(msg.tool_calls),
            usage=result.usage,
        )

    # -------- STREAMING LLM CALL -----------------------------------------------------------
    async def _call_llm_stream(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
        loop_state: BaseLoopState,
        cancellation_token: CancellationToken | None = None,
        output_format: type[BaseModel] | None = None,
        transient_messages: t.Sequence[CoreMessage] | None = None,
        tools_override: t.Sequence[CoreTool] | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Make one streaming LLM call through middleware.

        Forwards every content chunk as a ``ModelStreamChunkEvent``,
        accumulates content + tool-call fragments, and on the final
        chunk synthesizes a ``ChatCompletionResult`` deposited into
        ``loop_state``. Mirrors ``_call_llm``'s contract so the loop
        body can branch on ``stream_tokens`` without caring about
        the wire format.

        ``tool_call_chunk`` is treated as the OpenAI-style nested
        shape (``{"id": ..., "function": {"name": ..., "arguments": ...}}``)
        accumulated by call id. Providers that emit complete tool
        calls per chunk (e.g. Ollama) still match this shape — they
        just send all the fragments in a single chunk.

        ``tools_override`` — see ``_call_llm``.
        """
        _log = log.child(
            run_id=ctx.run_id,
            session_id=ctx.session_id,
        )
        mw = self._middleware_context(ctx)
        request = await self.middleware_chain.model_request(
            mw,
            ModelRequest(
                messages=self._model_input_messages(ctx),
                tools=list(tools_override)
                if tools_override is not None
                else self._tools,
                options=dict(kwargs),
                output_format=output_format,
                stream=True,
                model=str(getattr(self.client, "model", None) or "unknown"),
                model_config=getattr(self.client, "config", None),
            ),
        )
        model_name = request.model

        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()

        # clear any previous result
        loop_state.last_result = None

        async def _streaming_call() -> t.AsyncGenerator[ChatCompletionChunk, None]:
            stream = await self.client.run(
                ctx=self._model_context(ctx, transient_messages),
                prompts=prompts,
                tools=request.tools,
                output_format=request.output_format,
                stream=True,
                **request.options,
            )
            stream = t.cast(t.AsyncGenerator[ChatCompletionChunk, None], stream)
            async for chunk in stream:
                if cancellation_token and cancellation_token.is_cancelled():
                    raise asyncio.CancelledError()
                if (
                    chunk := await self.middleware_chain.model_chunk(mw, request, chunk)
                ) is not None:
                    yield chunk

        backoff = 1.0
        for attempt in range(self.max_connection_retries + 1):
            content_chunks: list[str] = []
            thinking_chunks: list[str] = []
            accumulated_tool_calls: dict[str, dict[str, t.Any]] = {}
            final_chunk: ChatCompletionChunk | None = None

            try:
                yield ModelCallEvent(
                    source=self.name,
                    model=model_name,
                    input_messages=self._input_messages_with_token_counts(
                        self._model_input_messages(ctx)
                    ),
                    prompt_tokens=prompts.prompt_tokens,
                )
                async for chunk in _streaming_call():
                    if not chunk.is_complete:
                        if chunk.content:
                            content_chunks.append(chunk.content)
                            yield ModelStreamChunkEvent(
                                source=self.name,
                                chunk=chunk.content,
                                is_final=False,
                            )

                        if chunk.thinking:
                            thinking_chunks.append(chunk.thinking)
                            yield ModelStreamChunkEvent(
                                source=self.name,
                                chunk="",
                                thinking=chunk.thinking,
                                is_final=False,
                            )

                        if chunk.tool_call_chunk:
                            self._merge_tool_call_chunk(
                                accumulated_tool_calls, chunk.tool_call_chunk
                            )
                        continue

                    final_chunk = chunk

                    tool_calls = self._build_tool_calls_from_chunks(
                        accumulated_tool_calls, _log
                    )
                    full_content = "".join(content_chunks)
                    full_thinking = (
                        "".join(thinking_chunks) if thinking_chunks else None
                    )

                    assistant_msg = AssistantMessage(
                        source=self.name,
                        content=full_content,
                        tool_calls=tool_calls,
                        thinking=full_thinking,
                        structured_output=final_chunk.structured_output,
                    )

                    usage = final_chunk.usage or Usage()
                    if tool_calls:
                        usage = usage.model_copy(update={"tool_calls": len(tool_calls)})

                    result = ChatCompletionResult(
                        message=assistant_msg,
                        usage=usage,
                        model=model_name,
                        finish_reason=(
                            final_chunk.finish_reason
                            or ("tool_calls" if tool_calls else "stop")
                        ),
                    )
                    result = await self.middleware_chain.model_response(
                        mw, request, result
                    )
                    loop_state.record_completion(result)
                    yield ModelResponseEvent(
                        source=self.name,
                        response=result.message.text(),
                        has_tool_calls=bool(result.message.tool_calls),
                        usage=result.usage,
                    )

                    yield ModelStreamChunkEvent(
                        source=self.name,
                        chunk="",
                        is_final=True,
                    )

                return

            except asyncio.CancelledError:
                _log.info(
                    "Streaming LLM call cancelled by user request",
                    attempt=attempt + 1,
                )
                raise

            except Exception as e:
                is_transient = isinstance(e, ClientError) and e.kind in _TRANSIENT_KINDS
                if not is_transient or attempt >= self.max_connection_retries:
                    recovered = await self.middleware_chain.model_error(mw, request, e)
                    if recovered is not None:
                        recovered = await self.middleware_chain.model_response(
                            mw, request, recovered
                        )
                        loop_state.record_completion(recovered)
                        yield ModelResponseEvent(
                            source=self.name,
                            response=recovered.message.text(),
                            has_tool_calls=bool(recovered.message.tool_calls),
                            usage=recovered.usage,
                        )
                        return
                    yield ErrorEvent(
                        source=self.name,
                        error_message=(
                            f"Streaming LLM call failed after "
                            f"{attempt + 1} attempt(s): {e}"
                        ),
                        error_type=type(e).__name__,
                        is_recoverable=False,
                    )
                    raise

                _log.warning(
                    "Transient streaming LLM error",
                    attempt=attempt + 1,
                    max_attempts=self.max_connection_retries + 1,
                    error=str(e),
                    action="retry",
                )
                await asyncio.sleep(backoff)
                backoff *= 2

    # -------- STREAM HELPERS -----------------------------------------------------------
    @staticmethod
    def _merge_tool_call_chunk(
        accumulated: dict[str, dict[str, t.Any]],
        fragment: dict[str, t.Any],
    ) -> None:
        """Merge an incoming tool-call fragment into the accumulator.

        Expected fragment shape (OpenAI-style)::

            {"id": "call_abc", "function": {"name": "x", "arguments": "{...}"}}

        Providers that emit complete tool calls in a single chunk
        (Ollama) still match this shape — the merge below is a no-op
        for the second+ fragment because there is none. Providers
        that stream argument deltas (OpenAI) accumulate the
        ``arguments`` string across fragments.
        """
        call_id = fragment.get("id")
        if not call_id:
            return

        existing = accumulated.get(call_id)
        if existing is None:
            fn: dict[str, t.Any] = fragment.get("function") or {}
            accumulated[call_id] = {
                "id": call_id,
                "function": {
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments") or "",
                },
            }
            return

        new_fn: dict[str, t.Any] = fragment.get("function") or {}
        existing_fn = existing["function"]
        if new_fn.get("name") and not existing_fn.get("name"):
            existing_fn["name"] = new_fn["name"]
        if new_fn.get("arguments"):
            existing_fn["arguments"] = (existing_fn.get("arguments") or "") + new_fn[
                "arguments"
            ]

    @staticmethod
    def _build_tool_calls_from_chunks(
        accumulated: dict[str, dict[str, t.Any]],
        _log: ScopedLogger,
    ) -> list["ToolCall"]:
        """Convert accumulated chunk fragments into ``ToolCall`` instances.

        Returns the loop's per-message tool calls. Fragments that fail
        to parse (missing name, malformed JSON arguments) are dropped
        with a warning rather than aborting the whole turn — the LLM
        gets the chance to retry on the next iteration.
        """
        from ..core.messages import ToolCall

        tool_calls: list[ToolCall] = []
        for call_id, data in accumulated.items():
            fn: dict[str, t.Any] = data.get("function") or {}
            name = fn.get("name")
            args_str = fn.get("arguments") or "{}"
            if not name:
                _log.warning(
                    "Dropping tool call chunk with no function name",
                    call_id=call_id,
                )
                continue
            try:
                params: dict[str, t.Any] = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError as e:
                _log.warning(
                    "Dropping tool call chunk with malformed arguments",
                    call_id=call_id,
                    arguments_preview=args_str[:120],
                    error=str(e),
                )
                continue
            tool_calls.append(ToolCall(id=call_id, tool_name=name, parameters=params))
        return tool_calls

    # -------- ABSTRACT -----------------------------------------------------------
    @abstractmethod
    async def execute_reasoning_loop(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
        loop_state: BaseLoopState,
        stream_tokens: bool = False,
        cancellation_token: CancellationToken | None = None,
        output_format: t.Type[BaseModel] | None = None,
        eval_criteria: list[str] | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Execute the reasoning loop for one agent turn.

        Implementations drive the iterative body: call the LLM,
        dispatch tool calls, decide when to stop. They must:

        - Append every produced message (assistant + tool) to
          ``ctx.messages`` so the next turn sees them.
        - Register every tool call as a ``ToolCallRecord`` on
          ``ctx.tool_state`` before passing to the executor.
        - Emit a final ``ReasoningCompleteEvent`` with the chosen
          ``finish_reason``.

        Yields:
            ``CoreEvent`` instances forwarded to the agent.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement execute_reasoning_loop"
        )
        yield
