"""ReAct reasoning loop — the framework's reasoning engine.

Drives the canonical Reason-Act cycle: the LLM proposes an action (tool
call or final answer), the executor runs the action, the result is folded
back into the transcript, and control returns to the LLM until it produces
a tool-free answer the CompletionGate accepts, or iterations run out.

The loop is transcript-aware (reads/writes ``ctx.messages``) and
tool-state-aware (registers ``ToolCallRecord`` instances on
``ctx.tool_state``). It is *not* prompt-aware — prompt assembly happens
inside the client, behind ``client.run()``.

User-facing constructor takes only config. The agent injects runtime
dependencies (client, dispatcher, completion_bus, ...) via ``bind()``.
"""

from __future__ import annotations

import asyncio
import logging
import typing as t

from pydantic import BaseModel

from ...loggers import ScopedLogger
from ...termination import CancellationToken
from ...base.reasoning import BaseReasoning, BaseLoopState

from ...core.messages import ToolMessage, UserMessage
from ...core.event_type import (
    CoreEvent,
    ReasoningCompleteEvent,
    ReasoningIterationEvent,
    ToolApprovalEvent,
    UserInputRequestEvent,
)
from ...types.stacks import PromptCtx
from ...types.run_context import RunContext

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["ReactLoop"])

NEED_APPROVAL = "need_approval"
ASK_USER = "input_needed"
MAX_ITERATIONS = "max_iterations"
MAX_ITERATIONS_REACHED = "max_iterations_reached"
HARNESS = "harness"


# -------- LOOP STATE -----------------------------------------------------------
class ReActLoopState(BaseLoopState):
    """Loop state for the ReAct loop.

    Currently identical to ``BaseLoopState``. Reserved as the extension
    point for loop-specific fields without bleeding into custom loops.
    """


# -------- LOOP -----------------------------------------------------------
class ReactLoop(BaseReasoning):
    """ReAct loop: call the model, run any tool calls, repeat.

    Terminates on: no tool calls and the CompletionGate accepts, max
    iterations, approval pause, input pause, missing LLM result, or
    external cancellation.
    """

    LOOP_STATE_CLS: t.ClassVar[type[BaseLoopState]] = ReActLoopState

    def __init__(
        self,
        max_loop_iterations: int = 10,
        max_connection_retries: int = 3,
    ) -> None:
        """Initialize the loop.

        Only config goes here. The agent calls ``bind()`` later to inject
        runtime dependencies (name, client, dispatcher, completion_bus).
        Tool availability (including ask_user) is decided by what's in the
        toolset, not by the loop — see Agent.

        Args:
            max_loop_iterations: Cap on iterations within one turn.
            max_connection_retries: Per-call transient-error retry budget.
        """
        super().__init__(max_connection_retries=max_connection_retries)
        if max_loop_iterations < 1:
            raise ValueError("max_loop_iterations must be positive")
        self.max_loop_iterations = max_loop_iterations

    # -------- ENTRY POINT -----------------------------------------------------------
    async def execute_reasoning_loop(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
        loop_state: ReActLoopState,
        stream_tokens: bool = False,
        cancellation_token: CancellationToken | None = None,
        output_format: type[BaseModel] | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Drive the ReAct cycle for one agent turn.

        Args:
            ctx: Run state. Mutated: ``ctx.messages`` grows,
                ``ctx.tool_state`` gains records for every tool call the
                LLM proposes, and ``ctx.plan`` tracks the model's plan.
            prompts: Rendered prompt layers; reused for every iteration
                of this turn (compaction may inject a summary block).
            loop_state: Iteration counter + metrics + last result.
                Mutated in place.
            stream_tokens: If True, use ``_call_llm_stream``; otherwise
                ``_call_llm``.
            cancellation_token: External cancellation signal.
            output_format: Optional structured-output schema; forwarded
                to the client untouched.
            **kwargs: Provider-specific overrides forwarded to
                ``client.run()``.

        Yields:
            ``CoreEvent`` instances (model events, reasoning events,
            planning events, tool events, errors).
        """
        if not isinstance(loop_state, ReActLoopState):  # type:ignore : defensive
            loop_state = ReActLoopState(**loop_state.model_dump())

        self._set_loop_state(loop_state)
        _log = log.child(run_id=ctx.run_id, session_id=ctx.session_id)

        # Resume preamble: finish whatever was left pending from a previous
        # pause, before the model sees anything new. Only ever has content
        # right after a resume — runs once, not per iteration.
        pending = [r for r in ctx.tool_state.records.values() if not r.is_consumed]
        if pending:
            needs_approval = False
            pause_events: list[CoreEvent] = []

            # Execute Pending tools
            async for item in self._execute_tools(
                ctx=ctx, records=pending, cancellation_token=cancellation_token
            ):
                if isinstance(item, ToolMessage):
                    ctx.messages.append(item)
                    loop_state.tool_calls += 1
                    continue
                if isinstance(item, ToolApprovalEvent):
                    pause_events.append(item)
                    needs_approval = True
                    continue
                if isinstance(item, UserInputRequestEvent):
                    pause_events.append(item)
                    continue
                yield item

            # Notify the agent that we need to pause for approval or input. The
            # agent will resume the loop when the user approves or provides input.
            if pause_events:
                loop_state.finish_reason = NEED_APPROVAL if needs_approval else ASK_USER
                for ev in pause_events:
                    yield ev
                yield self._reasoning_complete(loop_state)
                return

        while True:
            # External cancellation check. This is a "soft" check, not a hard
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            # This is a soft notification to inform the agent in the next turn
            if loop_state.iteration >= self.max_loop_iterations:
                loop_state.finish_reason = MAX_ITERATIONS
                msg = UserMessage(source=HARNESS,content=MAX_ITERATIONS_REACHED)
                ctx.messages.append(msg)

            # PLACE O HOLD AS THE COMPACTINO STAREGY IS NOT WELL IMPLEMETNED
            # IN HERe WE SHOUDL COMPACT THE MESSAGES IF NEEDED, BUT THE STRATEGY IS NOT WELL IMPLEMENTED YET
            # AND EMIT THE EVENT 
            # if self._is_compaction_needed(ctx.messages, prompts):
            #     self._compact_messages(ctx.messages)

            loop_state.iteration += 1
            yield ReasoningIterationEvent(
                source=self.name,
                iteration=loop_state.iteration,
                max_iterations=self.max_loop_iterations,
            )

            # Call LLM (streming or non-streaming)
            call = self._call_llm_stream if stream_tokens else self._call_llm
            async for event in call(
                ctx=ctx,
                prompts=prompts,
                loop_state=loop_state,
                cancellation_token=cancellation_token,
                output_format=output_format,
                **kwargs,
            ):  
                # Tool is called and needs approval, we yield the event and continue the loop
                if isinstance(event, ToolApprovalEvent):
                   needs_approval = True
                   yield event
                   continue 

                # yield any other event
                yield event
               

        raise NotImplementedError("execute_reasoning_loop is being rebuilt")
        yield  # pragma: no cover - makes this a generator

    def _reasoning_complete(self, loop_state: ReActLoopState) -> ReasoningCompleteEvent:
        """Build the terminal event for the current turn."""
        return ReasoningCompleteEvent(
            source=self.name,
            finish_reason=loop_state.finish_reason or "unknown",
            total_iterations=loop_state.iteration,
        )
