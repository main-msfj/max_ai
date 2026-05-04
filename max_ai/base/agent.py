"""Main Class and Base for Agent Implementation"""

from __future__ import annotations

import asyncio
import logging
import time
import typing as t
from abc import ABC

from pydantic import BaseModel

from .component import ComponentBase
from .clients import CoreChatCompletionClient
from .tool_executor import ToolExecutor

from ..loggers import ScopedLogger
from ..core.models import AgentConfig
from ..base.middleware import CoreMiddleware
from ..core.messages import CoreMessage, UserMessage
from ..core.event_type import (
    CompactionEvent,
    CoreEvent,
    ErrorEvent,
    ModelStreamChunkEvent,
)
from ..types.agent_response import AgentResponse
from ..types.completions import Usage
from ..types.run_context import RunContext
from ..types.stacks import PromptCtx
from ..termination import CancellationToken
from ..manager import AgentCapabilities
from ..manager.stacks import PromptVariablesBuilder, build_default_stack
from ..errors.agent import AgentError
from ..executor.local import LocalExecutor

if t.TYPE_CHECKING:
    from .executor import CoreExecutor
    from ..stacks import CoreLayer
    from .tools import CoreTool
    from .skill import CoreSkillRegistry
    from .memory import CoreMemoryRegistry
    from .context import CoreLogBookRegistry
    from .routines import CoreRoutineRegistry
    from ..manager.stacks import LayerContainer
    from .knowledge import CoreKnowledgeRegistry
    from .reasoning import BaseReasoning
    from .compaction import CoreCompaction


# Single yield type for the streaming engine. The terminal item of every
# run_stream_events call is an AgentResponse; everything before is a CoreEvent.
RunYield = t.Union[CoreEvent, AgentResponse]


# -------- LOGGER -----------------------------------------------------------
logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["Agent"])


class Agent(ComponentBase[BaseModel], ABC):
    """
    Core interface for all MaxAI agents.

    Lifecycle:
      1. ``__init__`` — sync. Stores configuration, builds the
         ``AgentCapabilities`` (validating tool name uniqueness,
         priority_tools coherence, etc.) and the ``LayerContainer``
         (validating every layer's template against its declared
         contract). Anything broken at this stage raises immediately.
      2. ``await agent.prepare()`` — async. Loads skills via their
         registry, fetches the current memory snapshot and session
         summary, renders every layer of the prompt stack with the
         right variables, and stores the result.
      3. ``await agent.run(...)`` (or one of its streaming variants) —
         async. By the time the run starts, ``self.rendered_layers``
         is fully populated and every tool is registered.

    Three public APIs share a single engine:
      - ``run_stream_events`` — yields every CoreEvent produced during
        the run, terminated by a single AgentResponse. The engine.
      - ``run_stream`` — yields only assistant text chunks. Convenience
        for CLI / chat use.
      - ``run`` — consumes everything, returns the AgentResponse.
        Convenience for scripts / tests.
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        client: CoreChatCompletionClient,
        memory: CoreMemoryRegistry | None = None,
        skills: CoreSkillRegistry | None = None,
        logbook: CoreLogBookRegistry | None = None,
        routines: CoreRoutineRegistry | None = None,
        toolset: t.Sequence[CoreTool] | None = None,
        knowledge: t.Sequence[CoreKnowledgeRegistry] | None = None,
        middlewares: t.Sequence[CoreMiddleware] | None = None,
        framework_layers: t.Sequence[CoreLayer] | None = None,
        reasoning: BaseReasoning | None = None,
        compaction: CoreCompaction | None = None,
        executor: CoreExecutor | None = None,
        output_format: t.Type[BaseModel] | None = None,
        priority_tools: list[str] | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        """
        Args:
            name: Unique identifier for the agent.
            description: External metadata for orchestrators or
                multi-agent discovery.
            instructions: Agent core instructions as a string of
                directives.
            client: LLM provider abstraction for API interactions.
            memory: Persistent user-fact backend.
            skills: Skill registry — resolved during ``prepare()``.
            logbook: Conversation-logbook registry (per session
                summaries + cross-session search).
            toolset: Executable functions available to the agent.
            routines: Repetitive task procedures discovered via tools.
            knowledge: Persistent sources of truth (RAG backends).
            middlewares: Logic hooks to intercept and process
                operations.
            framework_layers: Custom prompt layers to override or extend
                the default stack. Each entry replaces the default
                layer of the same concrete type. Unknown layer types
                are appended. Validation runs during construction —
                a broken layer raises before the agent is ever used.
            reasoning: Optional user-provided reasoning loop. Defaults
                to ReActLoop. Config-only at construction; runtime
                wiring is injected later via ``bind()``.
            compaction: Optional user-provided context compaction strategy.
                Defaults to SlidingWindowCompaction.
            executor: Optional user-provided execution strategy. Defaults to
                ``LocalExecutor``.
            output_format: Pydantic model for structured response.
                Forwarded to the client on every LLM call.
            priority_tools: Tool names the agent should strongly prefer
                when relevant. Not mandatory — relevance to the user's
                query remains the deciding factor.
            config: Optional ``AgentConfig`` for execution tuning. If
                None, system defaults are used for timeouts, retries,
                and loops.
        """
        self.name = self.require_type(name, str, "name")
        self.description = self.require_type(description, str, "description")
        self.instructions = self.require_type(instructions, str, "instructions")
        self.client = self.require_type(client, CoreChatCompletionClient, "client")
        self.config = self.require_type(config or AgentConfig(), AgentConfig, "config")

        self.reasoning = reasoning
        if compaction is None:
            from ..compaction import SlidingWindowCompaction

            compaction = SlidingWindowCompaction()
        self.compaction = compaction
        self.output_format = output_format
        self.middlewares = list(middlewares or [])
        self.executor = executor or LocalExecutor(self.config.tool_timeout)
        self.prompt_stack: LayerContainer = build_default_stack(framework_layers)

        self.capabilities = AgentCapabilities(
            memory=memory,
            routines=routines,
            skills=skills,
            logbook=logbook,
            priority_tools=priority_tools,
            toolset=toolset,
            knowledge=knowledge,
        )

        self._variables_builder: PromptVariablesBuilder = PromptVariablesBuilder(self)
        self._rendered_layers: dict[type[CoreLayer], str] = {}
        self._prepared: bool = False

    # -------- LIFECYCLE -----------------------------------------------------------
    async def prepare(self) -> None:
        """Hydrate async capabilities and render the prompt stack.

        Must be awaited once before any ``run*`` call (the engine does
        this automatically). Idempotent — calling twice is a no-op.

        Steps:
          1. Resolve async capabilities (loads skills, builds the
             ``read_skill_resource`` tool, revalidates tool names with
             skill tools merged in).
          2. For each layer in the prompt stack, ask the variables
             builder for the right inputs and render the layer.
          3. Store the rendered string keyed by layer type, so the
             client can later assemble the final system prompt in its
             provider-native order.
        """
        if self._prepared:
            return

        await self.capabilities.prepare()

        for layer in self.prompt_stack:
            variables = await self._variables_builder.collect(type(layer))
            try:
                rendered = layer.render(variables)
            except Exception as e:
                raise AgentError.layer_render_failed(
                    layer_name=type(layer).__name__,
                    error=e,
                ) from e
            self._rendered_layers[type(layer)] = rendered

        self._prepared = True

    def _ensure_prepared(self) -> None:
        if not self._prepared:
            raise AgentError.not_prepared(agent_name=self.name)

    # -------- INTERNAL HELPERS -----------------------------------------------------------
    def _normalize_run_context(
        self,
        task: str | CoreMessage | list[CoreMessage] | None,
        run_context: RunContext | None,
    ) -> RunContext:
        ctx = run_context if run_context is not None else RunContext()
        if task is None:
            return ctx

        if isinstance(task, str):
            ctx.messages.append(UserMessage(source="user", content=task))
            return ctx

        if isinstance(task, CoreMessage):
            ctx.messages.append(task)
            return ctx

        ctx.messages.extend(task)
        return ctx

    def _build_prompt_ctx(self) -> PromptCtx:
        return PromptCtx(
            stack=self.prompt_stack,
            variables={},
            rendered_layers=self.rendered_layers,
        )

    async def _apply_compaction(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
    ) -> CompactionEvent | None:
        max_context_tokens = getattr(self.client.config, "max_context_window", 0) or 0
        if max_context_tokens <= 0:
            return None

        result = await self.compaction.compact(
            ctx=ctx,
            prompts=prompts,
            max_context_tokens=max_context_tokens,
        )
        if not result.changed:
            return None

        return CompactionEvent(
            source=self.name,
            strategy=type(self.compaction).__name__,
            old_message_count=len(result.old_messages),
            recent_message_count=len(result.recent_messages),
            old_token_count=result.old_token_count,
            recent_token_count=result.recent_token_count,
            total_token_count=result.total_token_count,
            max_history_tokens=result.max_history_tokens,
        )

    def _build_reasoning(
        self,
        tool_executor: ToolExecutor,
    ) -> "BaseReasoning":
        """Resolve config-time reasoning into a runtime-bound instance.

        Accepts None (default ReActLoop) or a user-provided BaseReasoning
        instance. The agent injects runtime dependencies via .bind().
        """
        reasoning = self.reasoning
        if reasoning is None:
            from ..reasoning.react import ReActLoop

            reasoning = ReActLoop(
                max_loop_iterations=self.config.max_loop_iterations,
                max_connection_retries=self.config.max_connection_retries,
            )

        return reasoning.bind(
            name=self.name,
            client=self.client,
            tool_executor=tool_executor,
            middleware_chain=tool_executor.mw_chain,
        )

    def _build_response(
        self,
        ctx: RunContext,
        loop_state: t.Any,
        start_time: float,
    ) -> AgentResponse:
        finish_reason = loop_state.finish_reason or "unknown"
        if finish_reason == "max_iterations_exceeded":
            finish_reason = "max_iterations"
        elif finish_reason == "tool_calls":
            finish_reason = "stop"
        elif finish_reason not in {
            "stop",
            "max_iterations",
            "approval_needed",
            "tool_direct_return",
            "no_result",
            "error",
            "cancelled",
        }:
            finish_reason = "error"

        usage = Usage(
            duration_ms=int((time.monotonic() - start_time) * 1000),
            llm_calls=loop_state.llm_calls,
            tool_calls=loop_state.tool_calls,
            attempts_to_call_api=loop_state.attempts_to_call_api,
            retries=loop_state.retries,
            tokens_input=loop_state.tokens_input,
            tokens_output=loop_state.tokens_output,
            tokens_cached=loop_state.tokens_cached,
        )
        return AgentResponse(
            context=ctx,
            source=self.name,
            usage=usage,
            finish_reason=t.cast(t.Any, finish_reason),
        )

    # -------- PUBLIC API — STREAMING ENGINE -----------------------------------------------------------
    async def run_stream_events(
        self,
        task: str | CoreMessage | list[CoreMessage] | None = None,
        run_context: RunContext | None = None,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[RunYield, None]:
        """Drive one full run, yielding every CoreEvent then a terminal AgentResponse.

        This is the engine all other public methods consume. Yields
        each ``CoreEvent`` produced by the reasoning loop in real time
        (model events, reasoning events, tool events, errors), and
        ends with a single ``AgentResponse`` summarizing the run.

        Args:
            task: User input — string, a single ``CoreMessage``, a list
                of messages, or None to use only ``run_context``.
            run_context: Optional pre-existing context. If None, a fresh
                ``RunContext`` is created. The task (when provided) is
                appended to its ``messages``.
            cancellation_token: External cancellation signal.
            stream_tokens: If True, the underlying loop streams LLM
                output token-by-token via ``ModelStreamChunkEvent``.
                If False, only ``ModelResponseEvent`` is emitted per
                LLM call. Pass True when the consumer wants live text.
            **kwargs: Provider-specific overrides forwarded to the
                client.

        Yields:
            Zero or more ``CoreEvent`` instances, then exactly one
            ``AgentResponse`` as the final item.

        Raises:
            asyncio.CancelledError: External cancellation. The response
                is NOT yielded — the consumer cleans up.
        """
        await self.prepare()

        ctx = self._normalize_run_context(task, run_context)
        tool_executor = ToolExecutor(
            tools=self.capabilities.all_tools,
            middlewares=self.middlewares,
            agent_name=self.name,
            executor=self.executor,
            max_concurrent_tools=self.config.tool_call_concurrency,
        )
        reasoning = self._build_reasoning(tool_executor)
        prompts = self._build_prompt_ctx()
        compaction_event = await self._apply_compaction(ctx, prompts)
        if compaction_event is not None:
            yield compaction_event

        loop_state = reasoning.LOOP_STATE_CLS()
        start_time = time.monotonic()

        try:
            async for event in reasoning.execute_reasoning_loop(
                ctx=ctx,
                prompts=prompts,
                loop_state=loop_state,
                stream_tokens=stream_tokens,
                cancellation_token=cancellation_token,
                output_format=self.output_format,
                **kwargs,
            ):
                yield event

        except asyncio.CancelledError:
            # Caller-initiated cancellation — propagate without
            # yielding a response. The caller is unwinding.
            raise

        except Exception as e:
            # Loop-internal failure. Surface as ErrorEvent so the UI
            # can render it, then mark the response and continue.
            log.error(
                "Agent run failed",
                agent_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
            )
            yield ErrorEvent(
                source=self.name,
                error_message=str(e),
                error_type=type(e).__name__,
                is_recoverable=False,
            )
            loop_state.finish_reason = "error"

        # Terminal yield — always last, always exactly one.
        yield self._build_response(ctx, loop_state, start_time)

    # -------- PUBLIC API — TEXT STREAM -----------------------------------------------------------
    async def run_stream(
        self,
        task: str | CoreMessage | list[CoreMessage] | None = None,
        run_context: RunContext | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[str, None]:
        """Yield assistant text chunks as they stream.

        Convenience over ``run_stream_events`` for CLI / chat use:
        consumes events, yields only the text chunks emitted by the
        LLM (no tool messages, no events, no final response).

        ``stream_tokens=True`` is set automatically — calling this
        method without streaming would defeat the point.

        Args:
            task: User input.
            run_context: Optional pre-existing context.
            cancellation_token: External cancellation signal.
            **kwargs: Forwarded to the client.

        Yields:
            ``str`` chunks of assistant content. Empty strings are
            filtered. Tool call arguments and final marker chunks
            are not yielded.
        """
        async for item in self.run_stream_events(
            task=task,
            run_context=run_context,
            cancellation_token=cancellation_token,
            stream_tokens=True,
            **kwargs,
        ):
            if (
                isinstance(item, ModelStreamChunkEvent)
                and not item.is_final
                and item.chunk
            ):
                yield item.chunk

    # -------- PUBLIC API — BLOCKING -----------------------------------------------------------
    async def run(
        self,
        task: str | CoreMessage | list[CoreMessage] | None = None,
        run_context: RunContext | None = None,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False,
        **kwargs: t.Any,
    ) -> AgentResponse:
        """Run the agent to completion and return the final response.

        Convenience over ``run_stream_events`` for scripts and tests.
        Discards every intermediate event and returns the terminal
        ``AgentResponse``.

        Args:
            task: User input.
            run_context: Optional pre-existing context.
            cancellation_token: External cancellation signal.
            stream_tokens: Forwarded to the engine. Has no effect on
                the return value (events are discarded either way) but
                affects whether middleware sees streaming chunks.
            **kwargs: Forwarded to the client.

        Returns:
            ``AgentResponse`` with the run's final state.
        """
        response: AgentResponse | None = None
        async for item in self.run_stream_events(
            task=task,
            run_context=run_context,
            cancellation_token=cancellation_token,
            stream_tokens=stream_tokens,
            **kwargs,
        ):
            if isinstance(item, AgentResponse):
                response = item

        # The engine guarantees exactly one AgentResponse as the final
        # item; reaching here without one means the engine itself broke.
        assert response is not None, "run_stream_events did not yield an AgentResponse"
        return response

    # -------- PUBLIC API — RESUME -----------------------------------------------------------
    async def resume(
        self,
        run_context: RunContext,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False,
        **kwargs: t.Any,
    ) -> AgentResponse:
        """Resume a paused run after the user has decided pending approvals.

        Convenience over ``resume_stream_events``. Discards intermediate
        events and returns the terminal ``AgentResponse``.
        """
        response: AgentResponse | None = None
        async for item in self.resume_stream_events(
            run_context=run_context,
            cancellation_token=cancellation_token,
            stream_tokens=stream_tokens,
            **kwargs,
        ):
            if isinstance(item, AgentResponse):
                response = item

        assert response is not None, (
            "resume_stream_events did not yield an AgentResponse"
        )
        return response

    async def resume_stream(
        self,
        run_context: RunContext,
        cancellation_token: CancellationToken | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[str, None]:
        """Resume and yield only assistant text chunks.

        Convenience for CLI / chat use. ``stream_tokens=True`` is set
        automatically.
        """
        async for item in self.resume_stream_events(
            run_context=run_context,
            cancellation_token=cancellation_token,
            stream_tokens=True,
            **kwargs,
        ):
            if (
                isinstance(item, ModelStreamChunkEvent)
                and not item.is_final
                and item.chunk
            ):
                yield item.chunk

    async def resume_stream_events(
        self,
        run_context: RunContext,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[RunYield, None]:
        """Resume a paused run, yielding events then a terminal response.

        Validates that the context is in a state that warrants resuming
        (pending approvals were resolved, no orphan executing records),
        applies the default stale-execution policy (fail), and then
        delegates to the same engine as ``run_stream_events`` with
        ``task=None``.

        Args:
            run_context: Context loaded from a store, with the user's
                approval decisions already applied via
                ``ctx.tool_state.apply_approval(...)``.
            cancellation_token: External cancellation signal.
            stream_tokens: Forwarded to the loop.
            **kwargs: Forwarded to the client.

        Yields:
            ``CoreEvent`` instances followed by exactly one
            ``AgentResponse``.

        Raises:
            AgentError: If the context has unresolved pending
                approvals, or has no work left to do.
        """
        self._validate_resumable(run_context)
        self._handle_stale_executions(run_context)

        async for item in self.run_stream_events(
            task=None,
            run_context=run_context,
            cancellation_token=cancellation_token,
            stream_tokens=stream_tokens,
            **kwargs,
        ):
            yield item

    # -------- RESUME HELPERS -----------------------------------------------------------
    def _validate_resumable(self, ctx: RunContext) -> None:
        """Raise if the context can't or shouldn't be resumed.

        Two failure modes:

        - Records still in ``PENDING_APPROVAL`` — the caller forgot to
          apply user decisions. Resuming would just pause again, which
          is technically idempotent but almost certainly a bug.
        - Nothing actionable left — every record consumed and no
          assistant message awaiting follow-up. There is no work to do;
          calling resume here is a logic error in the caller.
        """
        pending = ctx.tool_state.pending_approvals
        if pending:
            ids = [r.id for r in pending]
            raise AgentError.unresolved_approvals(
                agent_name=self.name, tool_call_ids=ids
            )

        # If there are no actionable records and no stale executions,
        # the loop has nothing to do beyond what already happened. The
        # caller probably wants ``run(task=...)`` instead.
        actionable = ctx.tool_state.actionable_calls
        rejected = ctx.tool_state.rejected_calls
        stale = ctx.tool_state.stale_executions
        if not actionable and not rejected and not stale:
            raise AgentError.nothing_to_resume(agent_name=self.name)

    def _handle_stale_executions(self, ctx: RunContext) -> None:
        """Default stale-execution policy: mark them all as failed.

        Stale records are tool calls that were ``EXECUTING`` when the
        previous run died. We have no way to know whether they actually
        completed, so the safe default is to surface them as failures —
        the LLM sees the failure on the next turn and decides what to
        do (retry, ask, give up).

        Subclasses can override this method to implement a different
        policy (e.g. retry stale, ask the user, leave alone).
        """
        for record in list(ctx.tool_state.stale_executions):
            ctx.tool_state.fail_stale(
                record.id,
                reason="Tool execution did not complete in a previous run.",
            )

    @property
    def rendered_layers(self) -> dict[type["CoreLayer"], str]:
        """Rendered prompt layers keyed by concrete type.

        Available only after ``prepare()`` has run. The dict iteration
        order matches the underlying stack's insertion order, which the
        client can use as a default ordering hint when assembling the
        system prompt for its provider.
        """
        self._ensure_prepared()
        return dict(self._rendered_layers)

    @property
    def is_prepared(self) -> bool:
        return self._prepared

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"prepared={self._prepared}, "
            f"capabilities={self.capabilities!r})"
        )
