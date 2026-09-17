"""
Self-directed ReAct reasoning loop — the framework's reasoning engine.

Drives the canonical Reason-Act cycle: the LLM proposes an action (tool
call or final answer), the executor runs the action, the result is folded
back into the transcript, and control returns to the LLM until it produces
a tool-free answer or iterations run out.

Planning is native but optional: the loop exposes an ``update_plan`` tool
the model calls whenever it wants to lay out or revise its approach (like
Claude Code's todo tool). The tool writes an ``AgentPlan`` onto the loop
state; the loop syncs it to ``ctx.plan`` and emits a ``PlanningEvent`` so
UIs can render live progress. When the model runs tools without keeping
the plan current, the loop passes a *transient* reminder into the next LLM
call — steering without polluting the durable transcript.

The loop is transcript-aware (reads/writes ``ctx.messages``) and
tool-state-aware (registers ``ToolCallRecord`` instances on
``ctx.tool_state``). It is *not* prompt-aware — prompt assembly happens
inside the client, behind ``client.run()``.

User-facing constructor takes only config (``max_loop_iterations``,
``max_connection_retries``, ``enable_human_input``). The agent injects
runtime dependencies (client, executor, middleware chain, agent name)
via ``bind()`` inside its ``run()`` method.
"""

from __future__ import annotations

import asyncio
import json
import logging
import typing as t

from pydantic import BaseModel

from ..loggers import ScopedLogger
from ..termination import CancellationToken
from ..base.reasoning import BaseReasoning, BaseLoopState
from ..base.tool_executor import USER_INPUT_TOOL_NAMES
from ..base.tools import CoreRuntimeTool
from .guards import GuardContext, LoopGuard, default_guards

from ..core.messages import CoreMessage, SystemMessage, ToolMessage
from ..core.event_type import (
    CoreEvent,
    PlanningEvent,
    ToolApprovalEvent,
    ScratchpadUpdateEvent,
    UserInputRequestEvent,
    ReasoningCompleteEvent,
    ReasoningIterationEvent,
)

from ..types.stacks import PromptCtx
from ..types.run_context import RunContext
from ..types.tool_call import ToolCallRecord

if t.TYPE_CHECKING:
    from ..core.messages import ToolCall


logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["ReactLoop"])


_PLAN_NUDGE = (
    "You have an active plan with unfinished steps. If you just completed "
    "a step, call update_plan to mark it done (and set the next one "
    "active) before continuing."
)


# -------- LOOP STATE -----------------------------------------------------------
class ReActLoopState(BaseLoopState):
    """Loop state for the self-directed ReAct loop.

    Currently identical to ``BaseLoopState``. Reserved as the extension
    point for loop-specific fields without bleeding into custom loops.
    """

    pass


# -------- LOOP -----------------------------------------------------------
class ReactLoop(BaseReasoning):
    """ReAct loop where the model manages its own plan via a tool.

    One iteration:
      1. Drain records left actionable/rejected by a previous turn
         (resume after approvals), then sync any plan those tools drafted.
      2. Compact mid-loop if the live transcript crossed the threshold.
      3. Emit ``ReasoningIterationEvent`` and call the LLM (streaming or
         not); a transient plan nudge computed last round rides along
         without entering ``ctx.messages``.
      4. Append the assistant message. No tool calls → final answer, done.
      5. Register tool calls on ``ctx.tool_state`` and execute them.
         Pauses end the turn: approvals with
         ``finish_reason='approval_needed'``, user-input questions with
         ``finish_reason='input_needed'`` — both are pure record state on
         ``ctx.tool_state``, applied by the consumer before ``resume()``.
      6. Sync a model-drafted plan to ``ctx.plan`` (emitting
         ``PlanningEvent``), decide whether next call needs a plan nudge,
         relay scratchpad updates.
      7. Loop.

    The loop terminates on: no tool calls, max iterations, approval pause,
    input pause, missing LLM result, or external cancellation.
    """

    LOOP_STATE_CLS: t.ClassVar[type[BaseLoopState]] = ReActLoopState

    def __init__(
        self,
        max_loop_iterations: int = 10,
        max_connection_retries: int = 3,
        enable_human_input: bool = True,
        guards: list[LoopGuard] | None = None,
    ) -> None:
        """Initialize the loop.

        Only config goes here. The agent calls ``bind()`` later to inject
        runtime dependencies (name, client, tool_executor, middleware_chain).

        Args:
            max_loop_iterations: Cap on iterations within one turn.
            max_connection_retries: Per-call transient-error retry budget.
            enable_human_input: Expose the native ask-the-user tool.
            guards: Loop guards steering the model back on track
                (schema-retry, repetition, budget, no-progress). ``None``
                installs ``default_guards()``; pass ``[]`` to disable.
        """
        super().__init__(
            max_connection_retries=max_connection_retries,
            enable_human_input=enable_human_input,
        )
        self.max_loop_iterations = max_loop_iterations
        self.guards: list[LoopGuard] = (
            default_guards() if guards is None else list(guards)
        )

    # -------- ENTRY POINT -----------------------------------------------------------
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
            eval_criteria: Accepted to satisfy the reasoning contract but
                ignored — this loop has no self-eval. Kept as an explicit
                parameter (not ``**kwargs``) so it is never forwarded to
                ``client.run()``.
            **kwargs: Provider-specific overrides forwarded to
                ``client.run()``.

        Yields:
            ``CoreEvent`` instances (model events, reasoning events,
            planning events, tool events, errors).
        """
        if not isinstance(loop_state, ReActLoopState):
            loop_state = ReActLoopState(**loop_state.model_dump())

        # Set loop state and register native runtime tools (human input,
        # update_plan) bound to this run's state.
        self._set_loop_state(loop_state)
        self._register_runtime_tools(loop_state)

        _log = log.child(run_id=ctx.run_id, session_id=ctx.session_id)

        # Per-call steering that must NOT enter the durable transcript.
        transient_guidance: list[CoreMessage] = []

        # Turn-scoped cache of successful tool results, keyed by
        # (tool, canonical args). An identical re-call is answered from
        # this cache instead of re-executing: deterministic — even when
        # compaction evicted the original result from the model's context,
        # the framework still remembers it ran.
        completed_calls: dict[str, str] = {}

        # Index into ctx.messages of the last final answer a guard vetoed
        # (marked interim). If the model then closes out its plan with a
        # bare update_plan call, that answer is promoted back to final
        # instead of forcing an extra LLM round that would re-generate
        # (and visibly repeat) the same text.
        vetoed_final_idx: int | None = None

        # Loop guards read the executor's live tool registry (runtime tools
        # included, since they were just registered above).
        guard_ctx = GuardContext(
            tools=self.tool_executor.tools,
            max_loop_iterations=self.max_loop_iterations,
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
                pause_events: list[CoreEvent] = []
                needs_approval = False
                async for item in self.tool_executor.execute_tool_call(
                    ctx=ctx,
                    records=pending,
                    cancellation_token=cancellation_token,
                ):
                    if isinstance(item, ToolMessage):
                        ctx.messages.append(item)
                        loop_state.tool_calls += 1
                        self._remember_completed_call(completed_calls, ctx, item)
                        continue
                    if isinstance(item, ToolApprovalEvent):
                        pause_events.append(item)
                        needs_approval = True
                        continue
                    if isinstance(item, UserInputRequestEvent):
                        pause_events.append(item)
                        continue
                    yield item

                if pause_events:
                    # Sync any plan drafted by the tools that DID run this
                    # round before pausing — loop_state (and plan_draft)
                    # does not survive the pause/resume boundary.
                    async for ev in self._sync_plan(ctx, loop_state):
                        yield ev
                    loop_state.finish_reason = (
                        "approval_needed" if needs_approval else "input_needed"
                    )
                    for ev in pause_events:
                        yield ev
                    yield self._reasoning_complete(loop_state)
                    return

                # The drained tools may have included an update_plan call.
                async for ev in self._sync_plan(ctx, loop_state):
                    yield ev

            if self._should_compact(ctx, prompts):
                async for event in self._run_mid_loop_compaction(ctx, prompts):
                    yield event

            loop_state.iteration += 1

            yield ReasoningIterationEvent(
                source=self.name,
                iteration=loop_state.iteration,
                max_iterations=self.max_loop_iterations,
            )

            # 1. LLM call (stream or non-stream — same contract). Transient
            #    guidance rides along for this call only.
            call = self._call_llm_stream if stream_tokens else self._call_llm
            paused_in_call = False
            async for event in call(
                ctx=ctx,
                prompts=prompts,
                loop_state=loop_state,
                cancellation_token=cancellation_token,
                output_format=output_format,
                transient_messages=transient_guidance or None,
                **kwargs,
            ):
                if isinstance(event, ToolApprovalEvent):
                    paused_in_call = True
                    yield event
                    continue
                yield event
            transient_guidance = []

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

            # 3. Record the assistant message in the transcript. Any new
            #    textual answer supersedes a previously vetoed one.
            assistant_msg = result.message
            ctx.messages.append(assistant_msg)
            if assistant_msg.text().strip():
                vetoed_final_idx = None

            # 4. Final answer — no tool calls, we're done... unless a guard
            #    vetoes it (e.g. an empty answer, an unfinished plan) and
            #    sends the model back with transient steering instead of
            #    ending the turn.
            if not assistant_msg.tool_calls:
                veto = self._guard_steering(
                    ctx, loop_state, guard_ctx, phase="final"
                )
                if veto:
                    # Mark the vetoed answer as interim: it stays in the
                    # transcript for the model's own context, but UIs hide
                    # it — the next answer supersedes it (otherwise the
                    # user sees the same answer 2-3 times, each a bit
                    # longer). Messages are frozen, so replace in place.
                    ctx.messages[-1] = assistant_msg.model_copy(
                        update={"interim": True}
                    )
                    vetoed_final_idx = len(ctx.messages) - 1
                    transient_guidance.extend(veto)
                    continue
                loop_state.finish_reason = result.finish_reason or "stop"
                break

            # 5. Split calls into cache hits (identical successful call
            #    already ran this turn — answer from the cache, never
            #    re-execute) and fresh calls promoted into records.
            fresh_calls: list["ToolCall"] = []
            for tc in assistant_msg.tool_calls:
                cached = None
                if self._dedup_eligible(tc.tool_name):
                    cached = completed_calls.get(
                        self._call_key(tc.tool_name, tc.parameters)
                    )
                if cached is None:
                    fresh_calls.append(tc)
                    continue
                _log.info(
                    "Answering repeated tool call from turn cache",
                    tool_name=tc.tool_name,
                    tool_call_id=tc.id,
                )
                ctx.messages.append(
                    ToolMessage(
                        source=self.name,
                        tool_call_id=tc.id,
                        tool_name=tc.tool_name,
                        success=True,
                        content=(
                            "[framework] Identical call already executed "
                            "this turn — result reused, tool NOT re-run:\n"
                            f"{cached}"
                        ),
                    )
                )

            records = self._register_tool_calls(ctx, fresh_calls)

            # 6. Execute. Pause events (approvals + user-input requests)
            #    are batched so the entire pause set arrives together at
            #    the end of the turn. Both pauses are pure record state:
            #    the consumer applies decisions/answers to ctx.tool_state
            #    and resumes.
            pause_events = []
            needs_approval = False
            async for item in self.tool_executor.execute_tool_call(
                ctx=ctx,
                records=records,
                cancellation_token=cancellation_token,
            ):
                if isinstance(item, ToolMessage):
                    ctx.messages.append(item)
                    loop_state.tool_calls += 1
                    self._remember_completed_call(completed_calls, ctx, item)
                    continue

                if isinstance(item, ToolApprovalEvent):
                    pause_events.append(item)
                    needs_approval = True
                    continue

                if isinstance(item, UserInputRequestEvent):
                    pause_events.append(item)
                    continue

                yield item

            if pause_events:
                # Sync any plan drafted by the tools that DID run this
                # round before pausing — loop_state (and plan_draft) does
                # not survive the pause/resume boundary, so deferring the
                # sync to the resumed segment would silently drop the plan.
                async for ev in self._sync_plan(ctx, loop_state):
                    yield ev
                loop_state.finish_reason = (
                    "approval_needed" if needs_approval else "input_needed"
                )
                for ev in pause_events:
                    yield ev
                yield self._reasoning_complete(loop_state)
                return

            # 7. Tools ran. Sync any model-managed plan update, then decide
            #    whether the next call needs a structural nudge. Capture the
            #    flag BEFORE _sync_plan consumes it.
            plan_updated_this_round = loop_state.plan_updated
            async for ev in self._sync_plan(ctx, loop_state):
                yield ev

            # One-shot finish: the model closed out its plan with only
            # update_plan calls. Two accepted shapes — forcing one more
            # LLM round just to (re)state the answer is friction small
            # models reliably fail at:
            #   a) text + update_plan in the same message → the text is
            #      the final answer;
            #   b) bare update_plan after a guard vetoed a final answer →
            #      promote that vetoed answer back to final (un-interim
            #      it) and end silently, instead of having the model
            #      visibly repeat the answer it already gave.
            if (
                plan_updated_this_round
                and ctx.plan is not None
                and not ctx.plan.has_unfinished_steps()
                and all(
                    tc.tool_name == "update_plan"
                    for tc in assistant_msg.tool_calls
                )
            ):
                if assistant_msg.text().strip():
                    loop_state.finish_reason = "stop"
                    break
                if vetoed_final_idx is not None:
                    vetoed = ctx.messages[vetoed_final_idx]
                    ctx.messages[vetoed_final_idx] = vetoed.model_copy(
                        update={"interim": False}
                    )
                    _log.info(
                        "Plan closed with bare update_plan; promoting the "
                        "guard-vetoed answer as final instead of asking "
                        "the model to repeat it"
                    )
                    loop_state.finish_reason = "stop"
                    break

            # Structural plan nudge: the model ran tools (made progress) but
            # did not update the plan, yet the plan still has unfinished
            # steps. Passed transiently into the NEXT LLM call — never
            # appended to ctx.messages, so repeated nudges don't accumulate
            # in the persisted transcript. Self-correcting: once the model
            # calls update_plan, the nudge stops.
            if (
                not plan_updated_this_round
                and ctx.plan is not None
                and ctx.plan.has_unfinished_steps()
            ):
                transient_guidance.append(
                    SystemMessage(source="plan-progress", content=_PLAN_NUDGE)
                )

            # 8. Loop guards inspect the finished round (schema failures,
            #    repetition, budget) and steer the next call transiently.
            transient_guidance.extend(
                self._guard_steering(ctx, loop_state, guard_ctx, phase="round")
            )

            if loop_state.scratchpad_updated:
                loop_state.scratchpad_updated = False
                yield ScratchpadUpdateEvent(
                    source=self.name,
                    scratchpad=loop_state.scratchpad.model_copy(deep=True),
                )

        else:
            loop_state.finish_reason = "max_iterations_exceeded"
            _log.warning(
                "Max reasoning loop iterations exceeded",
                max_loop_iterations=self.max_loop_iterations,
                final_iteration=loop_state.iteration,
            )

        yield self._reasoning_complete(loop_state)

    # -------- HELPERS -----------------------------------------------------------
    def _register_runtime_tools(self, loop_state: BaseLoopState) -> None:
        """Register the native human-input tool, then add ``update_plan``.

        Extends the base (which registers the human-input tool when enabled)
        with the self-directed ``update_plan`` tool, bound to *this* run's
        loop_state so the model can manage its own plan. Done here (not in
        ``__init__``) because the tool needs the per-run loop_state, which only
        exists now. Idempotent: skip if already present (e.g. on resume, or a
        user-registered one). Imported locally to avoid a loop<->tool import
        cycle.
        """
        super()._register_runtime_tools(loop_state)

        from ..tools.plan import UpdatePlanTool

        if UpdatePlanTool.TOOL_NAME not in self.tool_executor.tools:
            self.tool_executor.tools[UpdatePlanTool.TOOL_NAME] = UpdatePlanTool(
                loop_state=loop_state
            )

    # -------- TURN-SCOPED CALL DEDUP ---------------------------------------------
    @staticmethod
    def _call_key(tool_name: str, parameters: dict[str, t.Any]) -> str:
        return f"{tool_name}:{json.dumps(parameters, sort_keys=True, default=str)}"

    def _dedup_eligible(self, tool_name: str) -> bool:
        """Whether repeated identical calls to this tool are cacheable.

        Excluded: the native plan/ask tools (stateful — every call
        matters), runtime tools like ``bash`` (commands are often
        legitimately repeated: polling, mutations), and any tool that
        opts out via ``allow_repeated_calls = True``.
        """
        if tool_name == "update_plan" or tool_name in USER_INPUT_TOOL_NAMES:
            return False
        tool = self.tool_executor.tools.get(tool_name)
        if tool is None:
            return False
        if isinstance(tool, CoreRuntimeTool):
            return False
        return not getattr(tool, "allow_repeated_calls", False)

    def _remember_completed_call(
        self,
        completed_calls: dict[str, str],
        ctx: RunContext,
        message: ToolMessage,
    ) -> None:
        """Cache a successful tool result under its (tool, args) key."""
        if not message.success or not self._dedup_eligible(message.tool_name):
            return
        record = ctx.tool_state.get(message.tool_call_id)
        if record is None:
            return
        completed_calls[self._call_key(record.tool_name, record.parameters)] = (
            message.text()
        )

    def _guard_steering(
        self,
        ctx: RunContext,
        loop_state: BaseLoopState,
        guard_ctx: GuardContext,
        phase: t.Literal["round", "final"],
    ) -> list[CoreMessage]:
        """Run all guards for one hook point; collect steering messages.

        A guard that raises is logged and skipped — steering is best-effort
        and must never take down the turn.
        """
        steering: list[CoreMessage] = []
        for guard in self.guards:
            try:
                hook = (
                    guard.after_tool_round
                    if phase == "round"
                    else guard.on_final_answer
                )
                text = hook(ctx, loop_state, guard_ctx)
            except Exception as e:
                log.warning(
                    "Loop guard failed; skipping",
                    guard=type(guard).__name__,
                    err=str(e),
                )
                continue
            if text:
                steering.append(
                    SystemMessage(source="loop-guard", content=text)
                )
        return steering

    async def _sync_plan(
        self,
        ctx: RunContext,
        loop_state: BaseLoopState,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Promote a model-drafted plan onto ctx.plan and announce it.

        The ``update_plan`` tool sets ``loop_state.plan_draft`` and
        ``plan_updated``. When it has, copy the draft to ``ctx.plan`` and
        emit a ``PlanningEvent`` so the UI renders live plan progress.
        """
        if not loop_state.plan_updated:
            return
        loop_state.plan_updated = False
        if loop_state.plan_draft is not None:
            ctx.plan = loop_state.plan_draft
            yield PlanningEvent(source=self.name, phase="progress", plan=ctx.plan)

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
