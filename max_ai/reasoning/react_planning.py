"""
Pure ReAct+Planning reasoning loop.

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

from pydantic import BaseModel, Field

from ..base.reasoning import BaseReasoning, BaseLoopState

from ..loggers import ScopedLogger
from ..termination import CancellationToken

from ..core.messages import ToolMessage, UserMessage, SystemMessage, AssistantMessage
from ..core.event_type import (
    EvalEvent,
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


from ..types.completions import ChatCompletionResult

from .eval import EvalConfig

if t.TYPE_CHECKING:
    from ..core.messages import ToolCall


logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["ReActLoopPlanning"])


# -------- LOOP STATE -----------------------------------------------------------
class ReActLoopPlanningState(BaseLoopState):
    """ReAct-specific loop state.

    Currently identical to ``BaseLoopState``. Reserved as the
    extension point for ReAct-only fields (e.g. reflection markers,
    plan-step counters) without bleeding into other reasoning loops.
    """

    last_eval_issues: list[str] = Field(default_factory=list)
    last_eval_score: float | None = Field(default=None)
    step_failures: dict[int, int] = Field(default_factory=dict)


# -------- LOOP -----------------------------------------------------------
class ReActLoopPlanning(BaseReasoning):
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

    LOOP_STATE_CLS: t.ClassVar[type[BaseLoopState]] = ReActLoopPlanningState

    def __init__(
        self,
        max_loop_iterations: int = 10,
        max_connection_retries: int = 3,
        eval: EvalConfig | None = None,
        max_step_retries: int = 2,
    ) -> None:
        """Initialize the planning ReAct loop.

        Only config goes here. The agent calls ``bind()`` later to
        inject runtime dependencies (name, client, tool_executor,
        middleware_chain).

        Planning is intrinsic to this loop — it always produces a plan
        (use the plain ``ReActLoop`` when you don't want one). Self-eval is
        opt-in: pass an ``EvalConfig`` to enable it, leave ``eval=None`` to
        skip it.

        Args:
            max_loop_iterations: Cap on iterations within one turn.
            max_connection_retries: Per-call transient-error retry budget.
            eval: Self-evaluation config; ``None`` disables self-eval.
            max_step_retries: Failures a plan step may accumulate before it
                is marked failed and the loop replans.
        """
        super().__init__(max_connection_retries=max_connection_retries)
        self.max_loop_iterations = max_loop_iterations
        self.eval = eval
        self.max_step_retries = max_step_retries

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
            eval_criteria: Per-run self-eval criteria. Overrides the
                criteria set at construction; falls back to defaults
                when neither is given. Ignored unless self-eval runs.
            **kwargs: Provider-specific overrides forwarded to
                ``client.run()``.

        Yields:
            ``CoreEvent`` instances (model events, reasoning events,
            tool events, errors).
        """
        if not isinstance(loop_state, ReActLoopPlanningState):
            loop_state = ReActLoopPlanningState(**loop_state.model_dump())
        self._set_loop_state(loop_state)
        _log = log.child(run_id=ctx.run_id, session_id=ctx.session_id)

        # Planning is intrinsic to this loop. Plan only when none exists yet,
        # so a resume (where ctx.plan was already built and persisted) does
        # not throw away the in-flight plan and start over.
        if ctx.plan is None:
            async for event in self._planning_step(ctx, prompts, loop_state):
                yield event

        # Eval Wrapper. max_retries comes from the eval config; with no eval
        # config the loop runs exactly once (the `eval is None` breaks below).
        eval_attempt = 0
        eval_tokens_baseline = self._total_tokens(loop_state)
        max_eval_retries = self.eval.max_retries if self.eval is not None else 0
        while eval_attempt <= max_eval_retries:
            # Reset iteraction counter for the reasoning loop
            loop_state.iteration = 0

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

                if self._should_compact(ctx):
                    async for event in self._run_mid_loop_compaction(ctx, prompts):
                        yield event

                loop_state.iteration += 1

                yield ReasoningIterationEvent(
                    source=self.name,
                    iteration=loop_state.iteration,
                    max_iterations=self.max_loop_iterations,
                )

                if ctx.plan is not None:
                    step = ctx.plan.active_step()
                    if step is None:
                        step = ctx.plan.next_pending()
                        if step is not None:
                            step.status = "active"

                    if step is not None:
                        ctx.messages.append(
                            SystemMessage(
                                source="plan-progress",
                                content=f"You are working on step {step.id}: {step.description}",
                            )
                        )
                        yield PlanningEvent(
                            source=self.name, phase="progress", plan=ctx.plan
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

                    # 4.1 (piece 4, rule C): the active step finished as pure
                    # LLM output (no tool — e.g. "summarize"). Close it before
                    # exiting so the plan ends fully done, not stuck on the
                    # last step.
                    if ctx.plan is not None:
                        active = ctx.plan.active_step()
                        if active is not None:
                            if loop_state.step_failures.get(active.id, 0) > 0:
                                # A tool failed earlier on this step and the LLM
                                # then gave up with a final answer -> close the
                                # step as failed, not done (feedback 3.1).
                                ctx.plan.mark_failed(active.id)
                            else:
                                # Legit case: a pure-LLM step (no tool failure)
                                # finishing as final output -> done.
                                ctx.plan.mark_done(active.id)

                            yield PlanningEvent(
                                source=self.name, phase="progress", plan=ctx.plan
                            )
                    break

                # 5. Promote tool calls into records on ctx.tool_state.
                records = self._register_tool_calls(ctx, assistant_msg.tool_calls)

                # 6. Execute. Approval events are batched so the entire
                #    pause set arrives together at the end of the turn.
                approval_events: list[ToolApprovalEvent] = []
                tool_msgs_this_round: list[ToolMessage] = []
                async for item in self.tool_executor.execute_tool_call(
                    ctx=ctx,
                    records=records,
                    cancellation_token=cancellation_token,
                ):
                    if isinstance(item, ToolMessage):
                        ctx.messages.append(item)
                        tool_msgs_this_round.append(item)
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

                # 7. Tools ran. Loop back
                if loop_state.scratchpad_updated:
                    loop_state.scratchpad_updated = False
                    yield ScratchpadUpdateEvent(
                        source=self.name,
                        scratchpad=loop_state.scratchpad.model_copy(deep=True),
                    )

                if self.eval is not None and self.eval.intermediate:
                    failed = [m for m in tool_msgs_this_round if not m.success]
                    if failed:
                        errors_text = "\n".join(
                            f"- {m.tool_name}: {m.error}" for m in failed
                        )
                        ctx.messages.append(
                            UserMessage(
                                source="intermediate-eval",
                                content=(
                                    f"A tool call failed:\n{errors_text}\n"
                                    f"Reconsider your approach before continuing."
                                ),
                            )
                        )
                        yield EvalEvent(source=self.name, phase="intermediate")

                # 8. Advance the plan based on this round's tools.
                #    (piece 4, rule A): all tools succeeded → step done; on
                #    the next iteration piece 3 promotes the next pending step.
                #    (piece 5): a tool failed → count it; once a step exceeds
                #    max_step_retries, mark it failed and replan from here.
                if ctx.plan is not None and tool_msgs_this_round:
                    active = ctx.plan.active_step()
                    if active is not None:
                        if all(m.success for m in tool_msgs_this_round):
                            ctx.plan.mark_done(active.id)
                            yield PlanningEvent(
                                source=self.name, phase="progress", plan=ctx.plan
                            )
                        else:
                            n = loop_state.step_failures.get(active.id, 0) + 1
                            loop_state.step_failures[active.id] = n
                            if n >= self.max_step_retries:
                                ctx.plan.mark_failed(active.id)
                                reason = (
                                    f"step {active.id} ({active.description}) "
                                    f"failed {n} times"
                                )
                                async for ev in self._planning_step(
                                    ctx, prompts, loop_state, replan_reason=reason
                                ):
                                    yield ev

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

            # Eval step. Skipped entirely when no eval config was provided
            # or it explicitly disables self_eval.
            if self.eval is None or not self.eval.self_eval:
                break

            if loop_state.finish_reason != "stop":
                break

            eval_passed = False
            async for event in self._eval_step(ctx, prompts, loop_state, eval_criteria):
                if isinstance(event, EvalEvent) and event.phase == "complete":
                    eval_passed = event.passed or False
                yield event

            if eval_passed:
                break

            # Cost cap: stop retrying once eval retries have spent the
            # token budget, instead of silently re-running the loop.
            if self.eval.max_extra_tokens is not None:
                spent = self._total_tokens(loop_state) - eval_tokens_baseline
                if spent >= self.eval.max_extra_tokens:
                    _log.warning(
                        "Eval budget exhausted; stopping retries",
                        spent=spent,
                        budget=self.eval.max_extra_tokens,
                    )
                    yield EvalEvent(source=self.name, phase="skipped")
                    break

            eval_attempt += 1
            if eval_attempt > max_eval_retries:
                _log.warning(
                    "Max self-evaluation attempts exceeded",
                    max_eval_attempts=max_eval_retries,
                    final_attempt=eval_attempt,
                )
                break

            # Inject Feedback for Next Attempt
            issues_text = "\n".join(f"- {i}" for i in loop_state.last_eval_issues or [])
            ctx.messages.append(
                UserMessage(
                    source="self-eval",
                    content=f"Your response needs improvement:\n{issues_text}\nPlease try again.",
                )
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

    @staticmethod
    def _total_tokens(loop_state: BaseLoopState) -> int:
        """Accumulated billable tokens for the run (input + output).

        Cached tokens are excluded: they are discounted (~10% cost) and
        often not exposed by open-source/local providers, so counting
        them would make the eval budget inconsistent across backends.
        """
        return loop_state.tokens_input + loop_state.tokens_output

    def _reasoning_complete(self, loop_state: BaseLoopState) -> ReasoningCompleteEvent:
        """Build the terminal event for the current turn."""
        return ReasoningCompleteEvent(
            source=self.name,
            finish_reason=loop_state.finish_reason or "unknown",
            total_iterations=loop_state.iteration,
        )

    async def _planning_step(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
        loop_state: ReActLoopPlanningState,
        replan_reason: str | None = None,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Generate (or regenerate) the execution plan.

        Used in two situations, sharing the same machinery:
          * Initial planning before the main ReAct loop (``replan_reason``
            is None).
          * Replanning after a step failed too many times (piece 5):
            ``replan_reason`` carries the failure context so the LLM can
            produce a revised plan instead of repeating the dead end.
        """
        from .plan import AgentPlan

        yield PlanningEvent(source=self.name, phase="start")

        # The only difference between plan and replan: replanning injects the
        # failure context so the new plan avoids the dead end.
        if replan_reason is not None:
            ctx.messages.append(
                SystemMessage(
                    source="replan",
                    content=(
                        f"The previous plan could not be completed: "
                        f"{replan_reason}.\n"
                        f"Produce a revised plan from the current state."
                    ),
                )
            )

        try:
            # For simplicity, we reuse the same LLM call infrastructure as the main loop.
            # The planning prompt should be designed to elicit a plan in a single turn.
            result = t.cast(
                ChatCompletionResult,
                await self.client.run(
                    ctx=self._model_context(ctx),
                    prompts=prompts,
                    tools=None,  # Planning prompt should not include tool calls
                    output_format=AgentPlan,  # Expect the plan to be structured as an AgentPlan
                    stream=False,  # Planning is a single-turn call; no streaming
                ),
            )
        except Exception:
            yield PlanningEvent(source=self.name, phase="failed")
            return

        loop_state.record_completion(result)
        plan = result.message.structured_output
        if not isinstance(plan, AgentPlan):
            yield PlanningEvent(source=self.name, phase="failed")
            return

        # Serialize plan and passed into context
        line: list[str] = []
        for s in plan.steps:
            hint = f" [{s.tool_hint}]" if s.tool_hint else ""
            line.append(f"{s.id}. {s.description}{hint}")
        line.append(f"Rationale: {plan.rationale}")
        plan_text = "\n".join(line)

        ctx.messages.append(
            SystemMessage(
                source="planning",
                content=f"## Execution plan\n{plan_text}\n\nFollow this plan. You may adapt if needed.",
            )
        )
        ctx.plan = plan
        yield PlanningEvent(source=self.name, phase="complete", plan=plan)

    async def _eval_step(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
        loop_state: ReActLoopPlanningState,
        eval_criteria: list[str] | None = None,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Optional self-evaluation step after the main ReAct loop."""
        from .eval import EvalResult

        yield EvalEvent(source=self.name, phase="start")

        # Find last AssistantMessage in the transcript to evaluate
        last_assistant_msg = next(
            (m for m in reversed(ctx.messages) if isinstance(m, AssistantMessage)), None
        )
        if last_assistant_msg is None:
            yield EvalEvent(source=self.name, phase="skipped")
            return

        # Build Next Criteria
        defaults = [
            "Does the response fully address the original task?",
            "Are there missing steps or incomplete sections?",
            "Are there factual errors or unsupported claims?",
        ]

        config_criteria = self.eval.criteria if self.eval is not None else None
        criteria = (
            eval_criteria
            if eval_criteria is not None
            else (config_criteria or defaults)
        )
        criteria_text = "\n".join(f"- {c}" for c in criteria)

        # Inject the evaluation prompt with the assistant's last response and the criteria
        eval_ctx = ctx.model_copy(
            update={
                "messages": [
                    *ctx.messages,
                    UserMessage(
                        source="self-eval",
                        content=(
                            f"Evaluate your last response against these criteria:\n"
                            f"{criteria_text}\n"
                            f"Be honest and strict."
                        ),
                    ),
                ]
            }
        )

        try:
            result = t.cast(
                ChatCompletionResult,
                await self.client.run(
                    ctx=self._model_context(eval_ctx),
                    prompts=prompts,
                    tools=None,
                    output_format=EvalResult,
                    stream=False,
                ),
            )
        except Exception:
            yield EvalEvent(source=self.name, phase="failed")
            return

        loop_state.record_completion(result)
        eval_result = result.message.structured_output
        if not isinstance(eval_result, EvalResult):
            yield EvalEvent(source=self.name, phase="failed")
            return

        # Evalidate Pass/Fail
        checks = eval_result.checks
        score = sum(1 for c in checks if c.passed) / len(checks) if checks else 0.0
        threshold = self.eval.threshold if self.eval is not None else 0.8
        passed = score >= threshold

        # Update loop state
        loop_state.last_eval_score = score
        loop_state.last_eval_issues = eval_result.issues

        yield EvalEvent(
            source=self.name,
            phase="complete",
            score=score,
            passed=passed,
            result=eval_result,
        )
