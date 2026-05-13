"""
Abstract base class for reasoning loops.

A reasoning loop drives a single agent turn: call the LLM, dispatch
any tool calls, and decide whether to call the LLM again.

Lifecycle from the user's perspective:

  1. User constructs the loop with config-only kwargs (e.g.
     ``ReActLoop(max_loop_iterations=5)``). No client, no executor,
     no middleware — those don't exist yet at config time.
  2. The agent calls ``loop.bind(name, client, tool_executor,
     middleware_chain)`` inside ``run()`` to wire the runtime context.
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

import json
import time
import asyncio
import logging
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, ConfigDict

from ..loggers import ScopedLogger
from ..middleware.chain import MiddlewareChain
from ..termination import CancellationToken

from ..base.clients import CoreChatCompletionClient
from ..base.tools import CoreTool

from ..core.messages import AssistantMessage
from ..core.event_type import (
    CoreEvent,
    ErrorEvent,
    ModelResponseEvent,
    ModelStreamChunkEvent,
    ToolApprovalEvent,
    ModelCallEvent,
)

from ..types.run_context import RunContext
from ..types.stacks import PromptCtx
from ..types.completions import (
    ChatCompletionChunk,
    ChatCompletionResult,
    Usage,
)

from ..errors.client import ClientError

if t.TYPE_CHECKING:
    from .tool_executor import ToolExecutor
    from ..core.messages import ToolCall


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


# -------- BASE REASONING -----------------------------------------------------------
class BaseReasoning(ABC):
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
    """

    # Each subclass declares its own loop-state class. The agent reads
    # this to instantiate the right state at the start of each run.
    LOOP_STATE_CLS: t.ClassVar[type[BaseLoopState]] = BaseLoopState

    def __init__(
        self,
        max_connection_retries: int = 3,
    ) -> None:
        """Initialize config-only state.

        Runtime dependencies (``name``, ``client``, ``tool_executor``,
        ``middleware_chain``) are not arguments — the agent injects
        them later via ``bind()``. This keeps user-constructed
        reasoning instances free of agent internals.

        Args:
            max_connection_retries: Per-call retry budget for transient
                ``ClientError`` (rate_limit, timeout, etc.).
        """
        self.max_connection_retries = max_connection_retries

        # Runtime — populated by bind(). Accessing any of these before
        # bind() raises a clear error rather than silently passing None.
        self._name: str | None = None
        self._client: CoreChatCompletionClient | None = None
        self._tool_executor: "ToolExecutor | None" = None
        self._middleware_chain: MiddlewareChain | None = None

    # -------- BIND -----------------------------------------------------------
    def bind(
        self,
        name: str,
        client: CoreChatCompletionClient,
        tool_executor: "ToolExecutor",
        middleware_chain: MiddlewareChain,
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
        self._name = name
        self._client = client
        self._tool_executor = tool_executor
        self._middleware_chain = middleware_chain
        return self

    # -------- RUNTIME ACCESSORS -----------------------------------------------------------
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
    def tool_executor(self) -> "ToolExecutor":
        if self._tool_executor is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call .bind() first.")
        return self._tool_executor

    @property
    def middleware_chain(self) -> MiddlewareChain:
        if self._middleware_chain is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call .bind() first.")
        return self._middleware_chain

    @property
    def _tools(self) -> list[CoreTool]:
        """Tools currently available to the agent.

        Sourced from the executor each turn rather than cached, so a
        skill registry that mutates the catalog after ``prepare()`` is
        reflected in subsequent calls.
        """
        return list(self.tool_executor.tools.values())

    # -------- NON-STREAMING LLM CALL -----------------------------------------------------------
    async def _call_llm(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
        loop_state: BaseLoopState,
        cancellation_token: CancellationToken | None = None,
        output_format: t.Type[BaseModel] | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Make one non-streaming LLM call through middleware.

        Yields any events produced by the middleware chain plus a
        ``ModelResponseEvent`` summarizing the assistant message.
        Captures the result into ``loop_state`` so the loop body can
        inspect it after the helper returns.

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

        model_metadata: dict[str, t.Any] = {
            "model": getattr(self.client, "model", None),
            "tools": self._tools,
        }

        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()

        async def _single_call(_ctx: RunContext) -> ChatCompletionResult:
            task = asyncio.create_task(
                self.client.run(
                    ctx=_ctx,
                    prompts=prompts,
                    tools=self._tools,
                    output_format=output_format,
                    stream=False,
                    **kwargs,
                )
            )
            if cancellation_token:
                cancellation_token.link_future(task)
            return t.cast(ChatCompletionResult, await task)

        yield ModelCallEvent(
            source=self.name,
            model=str(model_metadata.get("model") or "unknown"),
            input_messages=ctx.messages,
        )

        backoff = 1.0
        for attempt in range(self.max_connection_retries + 1):
            try:
                async for item in self.middleware_chain.execute(
                    action="model_call",
                    ctx=ctx,
                    data=ctx,
                    func=_single_call,
                    metadata=model_metadata,
                ):
                    if isinstance(item, ChatCompletionResult):
                        loop_state.record_completion(item)
                        msg = item.message
                        response_text = (
                            msg.structured_output.model_dump_json()
                            if msg.structured_output is not None
                            else msg.text()
                        )
                        yield ModelResponseEvent(
                            source=self.name,
                            response=response_text,
                            has_tool_calls=bool(msg.tool_calls),
                            usage=item.usage,
                        )
                    else:
                        yield item

                return

            except asyncio.CancelledError:
                _log.info("LLM call cancelled by user request")
                raise

            except Exception as e:
                is_transient = isinstance(e, ClientError) and e.kind in _TRANSIENT_KINDS
                if not is_transient or attempt >= self.max_connection_retries:
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

    # -------- STREAMING LLM CALL -----------------------------------------------------------
    async def _call_llm_stream(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
        loop_state: BaseLoopState,
        cancellation_token: CancellationToken | None = None,
        output_format: t.Type[BaseModel] | None = None,
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
        """
        _log = log.child(
            run_id=ctx.run_id,
            session_id=ctx.session_id,
        )
        model_metadata = {
            "model": getattr(self.client, "model", None),
            "tools": self._tools,
        }

        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()

        async def _streaming_call(
            _ctx: RunContext,
        ) -> t.AsyncGenerator[ChatCompletionChunk, None]:
            stream = await self.client.run(
                ctx=_ctx,
                prompts=prompts,
                tools=self._tools,
                output_format=output_format,
                stream=True,
                **kwargs,
            )
            stream = t.cast(t.AsyncGenerator[ChatCompletionChunk, None], stream)
            async for chunk in stream:
                if cancellation_token and cancellation_token.is_cancelled():
                    raise asyncio.CancelledError()
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
                    model=str(model_metadata.get("model") or "unknown"),
                    input_messages=ctx.messages,
                )
                async for item in self.middleware_chain.execute_stream(
                    action="model_call_stream",
                    ctx=ctx,
                    data=ctx,
                    stream_func=_streaming_call,
                    metadata=model_metadata,
                ):
                    if isinstance(item, ToolApprovalEvent):
                        yield item
                        return

                    if not isinstance(item, ChatCompletionChunk):
                        yield item
                        continue

                    chunk = item

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
                        model=str(model_metadata.get("model") or "unknown"),
                        finish_reason="tool_calls" if tool_calls else "stop",
                    )
                    loop_state.record_completion(result)
                    yield ModelResponseEvent(
                        source=self.name,
                        response=assistant_msg.text(),
                        has_tool_calls=bool(tool_calls),
                        usage=usage,
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
