"""
Self-directed ReAct reasoning loop.

The planning-as-tool counterpart to ``ReActLoopPlanning``. Here the loop
does *not* own the plan: it runs the plain ReAct cycle and exposes an
``update_plan`` tool the model calls whenever it wants to lay out or revise
its approach. The tool writes an ``AgentPlan`` onto the loop state; this
loop syncs it to ``ctx.plan`` and emits a ``PlanningEvent`` so the UI sees
the same plan object both loops produce.

Use this when you want the model to organise itself (like Claude Code's
todo tool) rather than be steered by an external controller. Planning is
therefore optional — the model plans only if and when it chooses to.
"""

from __future__ import annotations

import asyncio
import logging
import typing as t

from pydantic import BaseModel

from ..loggers import ScopedLogger
from ..termination import CancellationToken
from ..base.reasoning import BaseLoopState

from ..core.messages import ToolMessage
from ..core.event_type import (
    CoreEvent,
    PlanningEvent,
    ToolApprovalEvent,
    ScratchpadUpdateEvent,
    UserInputRequestEvent,
    ReasoningIterationEvent,
)

from ..types.stacks import PromptCtx
from ..types.run_context import RunContext

from .react_simple import ReActLoop, ReActLoopState


logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["ReActLoopSelfDirected"])


# -------- LOOP -----------------------------------------------------------
class ReActLoopSelfDirected(ReActLoop):
    """ReAct loop where the model manages its own plan via a tool.

    Identical to ``ReActLoop`` except that, after each round of tool calls,
    it checks whether the model used the ``update_plan`` tool (which sets
    ``loop_state.plan_updated``). If so it syncs the drafted plan onto
    ``ctx.plan`` and emits a ``PlanningEvent`` for the UI — the same event
    and plan model the controller loop (``ReActLoopPlanning``) emits.

    Reuses ``ReActLoop``'s helpers (``_register_tool_calls``,
    ``_reasoning_complete``); only ``execute_reasoning_loop`` is overridden
    to add the plan-sync step.
    """

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
        """Drive the ReAct cycle, syncing a model-managed plan to ctx.plan.

        See ``ReActLoop.execute_reasoning_loop`` for the base contract.
        ``eval_criteria`` is accepted to satisfy the reasoning contract and
        ignored — this loop has no self-eval.
        """
        if not isinstance(loop_state, ReActLoopState):
            loop_state = ReActLoopState(**loop_state.model_dump())
        self._set_loop_state(loop_state)

        # Register the update_plan tool bound to *this* run's loop_state so the
        # model can manage its own plan. Done here (not in __init__) because the
        # tool needs the per-run loop_state, which only exists now. Idempotent:
        # skip if already present (e.g. on resume, or a user-registered one).
        # Imported locally to avoid a loop<->tool import cycle.
        from ..tools.update_plan import UpdatePlanTool

        if UpdatePlanTool.TOOL_NAME not in self.tool_executor.tools:
            self.tool_executor.tools[UpdatePlanTool.TOOL_NAME] = UpdatePlanTool(
                loop_state=loop_state
            )

        _log = log.child(run_id=ctx.run_id, session_id=ctx.session_id)

        while loop_state.iteration < self.max_loop_iterations:
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            # Drain any records left actionable/rejected from a previous turn
            # (typically a resume after the user resolved approvals).
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

                # The drained tools may have included an update_plan call.
                async for ev in self._sync_plan(ctx, loop_state):
                    yield ev

            if self._should_compact(ctx):
                async for event in self._run_mid_loop_compaction(ctx, prompts):
                    yield event

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

            # 6. Execute. Approval events are batched.
            approval_events = []
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

            # 7. Tools ran. Sync any model-managed plan update, then relay
            #    the scratchpad event, then loop back.
            async for ev in self._sync_plan(ctx, loop_state):
                yield ev

            if loop_state.scratchpad_updated:
                loop_state.scratchpad_updated = False
                yield ScratchpadUpdateEvent(
                    source=self.name,
                    scratchpad=loop_state.scratchpad.model_copy(deep=True),
                )

            if loop_state.finish_reason == "input_needed":
                yield UserInputRequestEvent(
                    source=self.name,
                    question=loop_state.pending_user_input_question or "",
                    options=loop_state.pending_user_input_options,
                )
                yield self._reasoning_complete(loop_state)
                return

        else:
            loop_state.finish_reason = "max_iterations_exceeded"
            _log.warning(
                "Max reasoning loop iterations exceeded",
                max_loop_iterations=self.max_loop_iterations,
                final_iteration=loop_state.iteration,
            )

        yield self._reasoning_complete(loop_state)

    # -------- HELPERS -----------------------------------------------------------
    async def _sync_plan(
        self,
        ctx: RunContext,
        loop_state: BaseLoopState,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Promote a model-drafted plan onto ctx.plan and announce it.

        The ``update_plan`` tool sets ``loop_state.plan_draft`` and
        ``plan_updated``. When it has, copy the draft to ``ctx.plan`` (the
        same field the controller loop uses) and emit a ``PlanningEvent`` so
        the UI renders one plan format regardless of which loop produced it.
        """
        if not loop_state.plan_updated:
            return
        loop_state.plan_updated = False
        if loop_state.plan_draft is not None:
            ctx.plan = loop_state.plan_draft
            yield PlanningEvent(
                source=self.name, phase="progress", plan=ctx.plan
            )
