"""
Pure ReAct reasoning loop.

Drives the canonical Reason-Act cycle: LLM proposes an action (tool
call or final answer), the executor runs the action, the result is
folded back into the transcript, and control returns to the LLM until
it produces a tool-free answer or iterations run out.

The loop is transcript-aware (reads/writes ``ctx.messages``) and
tool-state-aware (registers ``ToolCallRecord`` instances on
``ctx.tool_state``). It is *not* prompt-aware — prompt assembly
happens inside the client, behind ``client.run()``.

User-facing constructor takes only config (``max_loop_iterations``,
``max_connection_retries``). The agent injects runtime dependencies
(client, executor, middleware chain, agent name) via ``bind()``
inside its ``run()`` method.
"""

from __future__ import annotations

import asyncio
import logging
import typing as t

from pydantic import BaseModel

from ..base.reasoning import BaseReasoning, BaseLoopState

from ..loggers import ScopedLogger
from ..termination import CancellationToken

from ..core.messages import ToolMessage
from ..core.event_type import (
    CoreEvent,
    ReasoningIterationEvent,
    ReasoningCompleteEvent,
    ToolApprovalEvent,
)

from ..types.run_context import RunContext
from ..types.stacks import PromptCtx
from ..types.tool_call import ToolCallRecord

if t.TYPE_CHECKING:
    from ..core.messages import ToolCall


logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["ReActLoop"])


# -------- LOOP STATE -----------------------------------------------------------
class ReActLoopState(BaseLoopState):
    """ReAct-specific loop state.

    Currently identical to ``BaseLoopState``. Reserved as the
    extension point for ReAct-only fields (e.g. reflection markers,
    plan-step counters) without bleeding into other reasoning loops.
    """

    pass


# -------- LOOP -----------------------------------------------------------
class ReActLoop(BaseReasoning):
    """Pure ReAct reasoning loop.

    One iteration:
      1. Emit ``ReasoningIterationEvent``.
      2. Call the LLM (streaming or not). The base class deposits the
         resulting ``ChatCompletionResult`` into ``loop_state``.
      3. If the call paused for approval mid-stream, emit the pause
         events and a terminal ``ReasoningCompleteEvent``; return.
      4. Append the assistant message to ``ctx.messages``.
      5. If the assistant message has no tool calls, finish.
      6. Otherwise: build ``ToolCallRecord`` instances, register them
         on ``ctx.tool_state``, hand them to the executor.
      7. Collect ``ToolMessage`` outputs into ``ctx.messages`` and
         relay other events. If the executor pauses for approval,
         emit the pause events and a terminal
         ``ReasoningCompleteEvent``; return.
      8. Loop.

    The loop terminates on: no tool calls, max iterations, approval
    pause, missing LLM result, or external cancellation.
    """

    LOOP_STATE_CLS: t.ClassVar[type[BaseLoopState]] = ReActLoopState

    def __init__(
        self,
        max_loop_iterations: int = 3,
        max_connection_retries: int = 3,
    ) -> None:
        """Initialize the ReAct loop.

        Only config goes here. The agent calls ``bind()`` later to
        inject runtime dependencies (name, client, tool_executor,
        middleware_chain).

        Args:
            max_loop_iterations: Cap on iterations within one turn.
            max_connection_retries: Per-call transient-error retry budget.
        """
        super().__init__(max_connection_retries=max_connection_retries)
        self.max_loop_iterations = max_loop_iterations

    # -------- ENTRY POINT -----------------------------------------------------------
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
        """Drive the ReAct cycle for one agent turn.

        Args:
            ctx: Run state. Mutated: ``ctx.messages`` grows, and
                ``ctx.tool_state`` gains records for every tool call
                the LLM proposes.
            prompts: Rendered prompt layers; reused unchanged for
                every iteration of this turn.
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
            tool events, errors).
        """
        if not isinstance(loop_state, ReActLoopState):
            loop_state = ReActLoopState(**loop_state.model_dump())

        _log = log.child(
            run_id=ctx.run_id,
            session_id=ctx.session_id,
        )

        while loop_state.iteration < self.max_loop_iterations:
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            # Drain any records left actionable/rejected from a previous
            # turn (typically a resume after the user resolved approvals).
            # Must run BEFORE the LLM call so the LLM sees the tool results
            # in this same turn.
            pending = [
                r
                for r in ctx.tool_state.records.values()
                if r.is_actionable or r.is_rejected
            ]
            if pending:
                approval_events: list[ToolApprovalEvent] = []
                async for item in self.tool_executor.execute_tool_call(
                    ctx=ctx,
                    records=pending,
                    cancellation_token=cancellation_token,
                ):
                    if isinstance(item, ToolMessage):
                        ctx.messages.append(item)
                        loop_state.tool_calls += 1
                        continue
                    if isinstance(item, ToolApprovalEvent):
                        approval_events.append(item)
                        continue
                    yield item

                if approval_events:
                    loop_state.finish_reason = "approval_needed"
                    for ev in approval_events:
                        yield ev
                    yield self._reasoning_complete(loop_state)
                    return

            loop_state.iteration += 1

            yield ReasoningIterationEvent(
                source=self.name,
                iteration=loop_state.iteration,
                max_iterations=self.max_loop_iterations,
            )

            # 1. LLM call (stream or non-stream — same contract).
            call = self._call_llm_stream if stream_tokens else self._call_llm
            paused_in_call = False
            async for event in call(
                ctx=ctx,
                prompts=prompts,
                loop_state=loop_state,
                cancellation_token=cancellation_token,
                output_format=output_format,
                **kwargs,
            ):
                if isinstance(event, ToolApprovalEvent):
                    paused_in_call = True
                    yield event
                    continue
                yield event

            if paused_in_call:
                loop_state.finish_reason = "approval_needed"
                yield self._reasoning_complete(loop_state)
                return

            # 2. Did the LLM actually produce a result?
            result = loop_state.last_result
            if result is None:
                loop_state.finish_reason = "no_result"
                _log.warning("LLM call returned no result; ending turn")
                yield self._reasoning_complete(loop_state)
                return

            # 3. Record the assistant message in the transcript.
            assistant_msg = result.message
            ctx.messages.append(assistant_msg)

            # 4. Final answer — no tool calls, we're done.
            if not assistant_msg.tool_calls:
                loop_state.finish_reason = result.finish_reason or "stop"
                break

            # 5. Promote tool calls into records on ctx.tool_state.
            records = self._register_tool_calls(ctx, assistant_msg.tool_calls)

            # 6. Execute. Approval events are batched so the entire
            #    pause set arrives together at the end of the turn.
            approval_events: list[ToolApprovalEvent] = []
            async for item in self.tool_executor.execute_tool_call(
                ctx=ctx,
                records=records,
                cancellation_token=cancellation_token,
            ):
                if isinstance(item, ToolMessage):
                    ctx.messages.append(item)
                    loop_state.tool_calls += 1
                    continue

                if isinstance(item, ToolApprovalEvent):
                    approval_events.append(item)
                    continue

                yield item

            if approval_events:
                loop_state.finish_reason = "approval_needed"
                for ev in approval_events:
                    yield ev
                yield self._reasoning_complete(loop_state)
                return

            # 7. Tools ran. Loop back to let the LLM rephrase results.

        else:
            loop_state.finish_reason = "max_iterations_exceeded"
            _log.warning(
                "Max reasoning loop iterations exceeded",
                max_loop_iterations=self.max_loop_iterations,
                final_iteration=loop_state.iteration,
            )

        yield self._reasoning_complete(loop_state)

    # -------- HELPERS -----------------------------------------------------------
    def _register_tool_calls(
        self,
        ctx: RunContext,
        tool_calls: list["ToolCall"],
    ) -> list[ToolCallRecord]:
        """Create and register a ``ToolCallRecord`` for each tool call.

        Records start in ``PENDING_APPROVAL``; the executor decides
        whether to auto-approve based on each tool's ``approval_mode``.
        Adding to ``ctx.tool_state`` raises if a record with the same
        id is already tracked — the LLM should never reuse ids in a
        single turn, so a collision is a real bug worth surfacing.
        """
        records: list[ToolCallRecord] = []
        for tc in tool_calls:
            record = ToolCallRecord(
                id=tc.id,
                tool_name=tc.tool_name,
                parameters=tc.parameters,
                session_id=ctx.session_id,
            )
            ctx.tool_state.add(record)
            records.append(record)
        return records

    def _reasoning_complete(self, loop_state: BaseLoopState) -> ReasoningCompleteEvent:
        """Build the terminal event for the current turn."""
        return ReasoningCompleteEvent(
            source=self.name,
            finish_reason=loop_state.finish_reason or "unknown",
            total_iterations=loop_state.iteration,
        )
