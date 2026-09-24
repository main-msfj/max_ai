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

from pydantic import BaseModel, Field

from ....base.completion_gate import CompletionDecision
from ....base.reasoning import BaseLoopState, BaseReasoning, ReasoningConfig
from ....config import setting
from ....core.compaction.budget import client_max_output_tokens
from ....core.event_type import (
    CompletionRejectedEvent,
    CoreEvent,
    ReasoningCompleteEvent,
    ReasoningIterationEvent,
    TaskCompleteEvent,
    ToolApprovalEvent,
    UserInputRequestEvent,
)
from ....core.harness import messages as harness
from ....core.messages import (
    HARNESS_SOURCE,
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from ....core.primitives import FailureReason
from ....core.runtime import runtime_status
from ....core.termination import CancellationToken
from ....core.type_ref import type_ref
from ....loggers import ScopedLogger
from ....types.run_context import RunContext
from ....types.stacks import PromptCtx
from ....types.tool_call import ToolCallRecord
from ..guards import (
    BudgetGuard,
    GuardContext,
    LoopGuard,
    RepetitionGuard,
    SchemaRetryGuard,
)

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["ReactLoop"])

APPROVAL_NEEDED = "approval_needed"
ASK_USER = "input_needed"
TOOL_DENIED = "tool_denied"
MAX_ITERATIONS = "max_iterations"
OUTPUT_LIMIT = "output_limit"
MAX_CUT_OFFS = 2  # in a row; then the limit is too small for the task
HARNESS = HARNESS_SOURCE

def _without_broken_calls(message: AssistantMessage) -> AssistantMessage:
    """Drop tool calls whose arguments could not be parsed."""
    calls = [c for c in message.tool_calls if not c.parameters.get("parsing_error")]
    if len(calls) == len(message.tool_calls):
        return message
    return message.model_copy(update={"tool_calls": calls})


NO_LLM_RESULT = "no_result"
COMPLETED = "completed"
WAITING = "waiting"
CANCELLED = "cancelled"
STOP = "stop"


# -------- LOOP STATE -----------------------------------------------------------
class ReActLoopState(BaseLoopState):
    """Loop state for the ReAct loop.

    Currently identical to ``BaseLoopState``. Reserved as the extension
    point for loop-specific fields without bleeding into custom loops.
    """


# -------- LOOP -----------------------------------------------------------
class ReactLoopConfig(ReasoningConfig):
    """Configuration options for ``ReactLoop``."""
    max_loop_iterations: int | None = Field(
        default=None, ge=1, description="None = setting.max_loop_iterations.",
    )
    guards: list[dict[str, t.Any]] | None = Field(
        default=None, description="Serialized guards; None = the loop's defaults.",
    )


class ReactLoop(BaseReasoning):
    """ReAct loop: call the model, run any tool calls, repeat.

    Terminates on: no tool calls and the CompletionGate accepts, max
    iterations, approval pause, input pause, missing LLM result, or
    external cancellation.
    """

    LOOP_STATE_CLS: t.ClassVar[type[BaseLoopState]] = ReActLoopState
    component_schema = ReactLoopConfig
    component_provider_override = "maxai.reasoning.ReactLoop"

    def __init__(
        self,
        max_loop_iterations: int | None = None,
        max_connection_retries: int = 3,
        guards: t.Sequence[LoopGuard] | None = None,
    ) -> None:
        """Initialize the loop.

        Only config goes here. The agent calls ``bind()`` later to inject
        runtime dependencies (name, client, dispatcher, completion_bus).
        Tool availability (including ask_user) is decided by what's in the
        toolset, not by the loop — see Agent.

        Args:
            max_loop_iterations: Cap on iterations within one turn. ``None``
                follows ``setting.max_loop_iterations`` (env
                ``MAX_LOOP_ITERATIONS``), also once serialized.
            max_connection_retries: Per-call transient-error retry budget.
            guards: Mid-loop steering checks run after each tool round
                (see ``reasoning/guards.py``). Defaults to
                ``SchemaRetryGuard``/``RepetitionGuard``/``BudgetGuard`` —
                deliberately not the shared ``default_guards()`` (that one
                also includes ``NoProgressGuard``/``PlanCompletionGuard``,
                whose jobs live in ``RuntimeCompletionGate`` for this loop).
                Pass ``[]`` to disable guards entirely.
        """
        super().__init__(max_connection_retries=max_connection_retries)
        self.max_loop_iterations = max_loop_iterations
        # None keeps "the defaults": a stored agent picks up improved defaults.
        self._custom_guards = guards is not None
        self.guards: list[LoopGuard] = (
            [SchemaRetryGuard(), RepetitionGuard(), BudgetGuard()]
            if guards is None
            else list(guards)
        )

    @property
    def max_loop_iterations(self) -> int:
        """Perform the ``max loop iterations`` operation for ``ReactLoop``."""
        return self._max_loop_iterations or setting.max_loop_iterations

    @max_loop_iterations.setter
    def max_loop_iterations(self, value: int | None) -> None:
        """Perform the ``max loop iterations`` operation for ``ReactLoop``.

Parameters
----------
value : int | None
    Value supplied for ``value``."""
        if value is not None and value < 1:
            raise ValueError("max_loop_iterations must be positive")
        self._max_loop_iterations = value

    def _to_config(self) -> ReactLoopConfig:
        """Build the serializable configuration for ``ReactLoop``."""
        return ReactLoopConfig(
            max_loop_iterations=self._max_loop_iterations,
            max_connection_retries=self.max_connection_retries,
            guards=(
                [guard.serialize().model_dump(exclude_none=True) for guard in self.guards]
                if self._custom_guards else None
            ),
        )

    @classmethod
    def _from_config(cls, config: ReactLoopConfig) -> "ReactLoop":
        """Create an instance from its configuration for ``ReactLoop``.

Parameters
----------
config : ReactLoopConfig
    Value supplied for ``config``."""
        guards = None
        if config.guards is not None:
            guards = [LoopGuard.deserialize(guard) for guard in config.guards]
        return cls(
            max_loop_iterations=config.max_loop_iterations,
            max_connection_retries=config.max_connection_retries,
            guards=guards,
        )

    def _register_tool_calls(
        self, ctx: RunContext, tool_calls: list[ToolCall]
    ) -> list[ToolCallRecord]:
        """Create and register a ``ToolCallRecord`` for each tool call.

        Adding to ``ctx.tool_state`` raises if a record with the same id
        already exists — the LLM should never reuse ids in one turn, así
        que una colisión es un bug real que vale la pena que reviente.
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

    async def _run_tools(
        self,
        ctx: RunContext,
        loop_state: ReActLoopState,
        records: list[ToolCallRecord],
        cancellation_token: CancellationToken | None = None,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Execute tool records, fold results into the transcript, pass
        through every other event live.

        On pause (approval, input, or a tool denial), sets
        ``loop_state.finish_reason`` — callers check it after the loop
        and end the turn themselves.

        A denied tool (explicit user rejection or static permission
        policy) always stops this batch here, even if other records in
        it resolved cleanly — we can't know whether a later step
        depends on the denied one. The caller decides what happens
        next (see ``_tool_denied_followup``): a single constrained
        model call gets a chance to acknowledge the denial via
        ``ask_user``/``update_plan`` before the model can propose any
        other action, rather than silently chaining into unrelated
        approval requests.
        """
        _log = log.child(run_id=ctx.run_id, session_id=ctx.session_id)
        pause_events: list[CoreEvent] = []
        needs_approval = False

        async for item in self._execute_tools(
            ctx=ctx, records=records, cancellation_token=cancellation_token
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

        denied = self._denied_records(records)

        if not pause_events and not denied:
            return

        if denied:
            loop_state.finish_reason = TOOL_DENIED
            _log.info(
                "Tool-Denied-Pausing-Turn",
                iteration=loop_state.iteration,
                denied=[r.tool_name for r in denied],
            )
        else:
            # Caller decides whether to end the turn — this helper only
            # sets the reason, never yields the terminal event itself.
            loop_state.finish_reason = APPROVAL_NEEDED if needs_approval else ASK_USER
            _log.info(
                "Paused-For-Approval",
                iteration=loop_state.iteration,
                needs_approval=needs_approval,
            )

        for ev in pause_events:
            yield ev

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
                Mutated in place. MUST be a fresh instance per call —
                ``finish_reason`` isn't part of the metrics snapshot, so
                reusing one across a pause/resume leaks a stale pause
                reason into the next turn.
            stream_tokens: If True, use ``_call_llm_stream``; otherwise
                ``_call_llm``.
            cancellation_token: External cancellation signal.
            output_format: Optional schema of the final answer. The loop
                works in free text; once the gates accept the answer, one
                extra call without tools shapes it into this schema and
                attaches it as the final message's ``structured_output``.
            **kwargs: Provider-specific overrides forwarded to
                ``client.run()``.

        Yields:
            ``CoreEvent`` instances (model events, reasoning events,
            planning events, tool events, errors).
        """
        if not isinstance(loop_state, ReActLoopState):  # type:ignore : defensive
            raise TypeError(
                f"loop_state must be ReActLoopState, got {type(loop_state)}"
            )

        self._set_loop_state(loop_state)
        _log = log.child(run_id=ctx.run_id, session_id=ctx.session_id)

        # Steering from guards.after_tool_round() or a tool-denial followup,
        # if any — consumed by the next model call only, never written to
        # ctx.messages. tools_override likewise applies to that one call.
        pending_steering: list[SystemMessage] | None = None
        tools_override: list[t.Any] | None = None
        cut_offs = 0  # consecutive replies cut at the output limit

        # Resume preamble: finish whatever was left pending from a previous
        # pause, before the model sees anything new. Only ever has content
        # right after a resume — runs once, not per iteration.
        pending = [r for r in ctx.tool_state.records.values() if not r.is_consumed]
        if pending:
            async for item in self._run_tools(
                ctx, loop_state, pending, cancellation_token
            ):
                yield item
            if loop_state.finish_reason in (APPROVAL_NEEDED, ASK_USER):
                yield self._reasoning_complete(loop_state)
                return
            if loop_state.finish_reason == TOOL_DENIED:
                followup = self._tool_denied_followup(self._denied_records(pending))
                if followup is None:
                    yield self._reasoning_complete(loop_state)
                    return
                pending_steering, tools_override = followup
                loop_state.finish_reason = "unknown"

        # A gate pause keeps the proposed answer. Recheck it before spending
        # another model iteration, even when the model budget is exhausted.
        if (
            loop_state.last_completion_decision is not None
            and loop_state.last_completion_decision.status == WAITING
        ):
            decision = await self._check_completion(ctx, loop_state, cancellation_token)
            if decision.status == COMPLETED:
                yield TaskCompleteEvent(source=self.name, decision=decision)
            if decision.status in (COMPLETED, WAITING):
                yield self._reasoning_complete(loop_state)
                return

        # Main loop: call the model, run any tool calls, repeat until termination.
        while True:
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            if loop_state.iteration >= self.max_loop_iterations:
                loop_state.finish_reason = MAX_ITERATIONS
                _log.warning("Max iterations reached", iteration=loop_state.iteration)
                msg = UserMessage(source=HARNESS, content=harness.MAX_ITERATIONS_REACHED)
                ctx.messages.append(msg)
                break

            # Keep plan/summary current in the prompt, then compact if the
            # strategy asks for it; both run before every model call.
            self._refresh_session_state(ctx, prompts)
            async for event in self._compact_if_needed(ctx, prompts):
                yield event

            loop_state.iteration += 1
            yield ReasoningIterationEvent(
                source=self.name,
                iteration=loop_state.iteration,
                max_iterations=self.max_loop_iterations,
            )

            # Call Model
            paused_in_call: bool = False
            call = self._call_llm_stream if stream_tokens else self._call_llm
            async for event in call(
                ctx=ctx,
                prompts=prompts,
                loop_state=loop_state,
                cancellation_token=cancellation_token,
                transient_messages=pending_steering,
                tools_override=tools_override,
                **kwargs,
            ):
                if isinstance(event, ToolApprovalEvent):
                    paused_in_call = True
                    yield event
                    continue
                yield event  # yield any other event (ToolMessage, UserInputRequestEvent, etc.)
            pending_steering = None
            tools_override = None

            if paused_in_call:
                loop_state.finish_reason = APPROVAL_NEEDED
                _log.info("LLM-Paused-For-Approval", iteration=loop_state.iteration)
                yield self._reasoning_complete(loop_state)
                return

            # 2. Check for termination condition
            # no tool calls, max iterations, or external cancellation.
            result = loop_state.last_result
            if result is None:
                loop_state.finish_reason = NO_LLM_RESULT
                _log.warning("No LLM result captured", iteration=loop_state.iteration)
                yield self._reasoning_complete(loop_state)
                return

            # 3. Tracked assistant messages
            assistant_msg = result.message
            if result.finish_reason == "length":
                # Cut off at the output limit: a half-written tool call must not
                # run, and the model must learn why, or it retries the same way.
                assistant_msg = _without_broken_calls(assistant_msg)
                if not assistant_msg.tool_calls:
                    cut_offs += 1
                    _log.warning("Output cut off", iteration=loop_state.iteration, in_a_row=cut_offs)
                    if assistant_msg.text().strip():
                        ctx.messages.append(assistant_msg)
                    if cut_offs >= MAX_CUT_OFFS:
                        loop_state.finish_reason = OUTPUT_LIMIT
                        break
                    limit = client_max_output_tokens(self.client)
                    pending_steering = [SystemMessage(
                        source=HARNESS, content=harness.output_cut_off(limit),
                    )]
                    continue
            cut_offs = 0
            ctx.messages.append(assistant_msg)

            # 4. Check if the LLM produced any tool calls. If so, register them and execute them.
            if assistant_msg.tool_calls:
                records = self._register_tool_calls(ctx, assistant_msg.tool_calls)
                async for item in self._run_tools(
                    ctx, loop_state, records, cancellation_token
                ):
                    yield item
                if loop_state.finish_reason in (APPROVAL_NEEDED, ASK_USER):
                    yield self._reasoning_complete(loop_state)
                    return
                if loop_state.finish_reason == TOOL_DENIED:
                    followup = self._tool_denied_followup(self._denied_records(records))
                    if followup is None:
                        yield self._reasoning_complete(loop_state)
                        return
                    pending_steering, tools_override = followup
                    loop_state.finish_reason = "unknown"
                    continue
                pending_steering = self._run_guards(ctx, loop_state)
                continue

            # 5. If not tool calls, the loop is done. The CompletionGate is triggered.
            # Framework-level blockers (cancelled/pending) are enforced here,
            # before any gate is asked — a gate never has to remember to check them.
            gate_result = await self._check_completion(
                ctx, loop_state, cancellation_token
            )

            if gate_result.status == COMPLETED:
                if output_format is not None:
                    async for event in self._format_final_answer(
                        ctx, prompts, loop_state, output_format, cancellation_token, **kwargs
                    ):
                        yield event
                await self._apply_final_response_middleware(ctx)
                yield TaskCompleteEvent(source=self.name, decision=gate_result)
                break

            if gate_result.status == WAITING:
                _log.info("Gate waiting", reasons=gate_result.reasons)
                yield self._reasoning_complete(loop_state)
                return

            # Only "incomplete" left: let the model try again.
            yield CompletionRejectedEvent(source=self.name, decision=gate_result)
            continue

        # 6. Loop Finished: emit the terminal ReasoningCompleteEvent
        _log.info(
            "Loop-Finished",
            iteration=loop_state.iteration,
            finish_reason=loop_state.finish_reason,
        )
        yield self._reasoning_complete(loop_state)

    async def _apply_final_response_middleware(self, ctx: RunContext) -> None:
        """Let middleware read or map the accepted answer before it is delivered."""
        if not self.middleware_chain:
            return
        for index in range(len(ctx.messages) - 1, -1, -1):
            message = ctx.messages[index]
            if isinstance(message, AssistantMessage):
                mw = self._middleware_context(ctx)
                ctx.messages[index] = await self.middleware_chain.final_response(mw, message)
                return

    async def _format_final_answer(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
        loop_state: ReActLoopState,
        output_format: type[BaseModel],
        cancellation_token: CancellationToken | None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[CoreEvent, None]:
        """Shape the accepted answer into ``output_format`` with one call
        without tools. The transcript keeps the prose answer (the model's
        context for later turns); the object goes on its ``structured_output``."""
        index = next(
            (i for i in range(len(ctx.messages) - 1, -1, -1)
             if isinstance(ctx.messages[i], AssistantMessage)),
            None,
        )
        if index is None:
            return
        accepted_result = loop_state.last_result
        instruction = SystemMessage(source=HARNESS, content=harness.FORMAT_FINAL_ANSWER)
        async for event in self._call_llm(
            ctx=ctx,
            prompts=prompts,
            loop_state=loop_state,
            cancellation_token=cancellation_token,
            output_format=output_format,
            transient_messages=[instruction],
            tools_override=[],
            **kwargs,
        ):
            yield event
        shaped = loop_state.last_result
        loop_state.last_result = accepted_result
        if shaped is None or shaped.message.structured_output is None:
            log.warning("Final answer not shaped", output_format=output_format.__name__)
            return
        shape = shaped.message.structured_output
        ctx.messages[index] = ctx.messages[index].model_copy(
            update={"structured_output": shape, "structured_output_type": type_ref(type(shape))}
        )

    async def _check_completion(
        self,
        ctx: RunContext,
        loop_state: ReActLoopState,
        cancellation_token: CancellationToken | None,
    ) -> CompletionDecision:
        """Perform the internal ``check completion`` operation for ``ReactLoop``.

Parameters
----------
ctx : RunContext
    Value supplied for ``ctx``.
loop_state : ReActLoopState
    Value supplied for ``loop_state``.
cancellation_token : CancellationToken | None
    Value supplied for ``cancellation_token``."""
        blocked = runtime_status(
            ctx,
            cancelled=bool(cancellation_token and cancellation_token.is_cancelled()),
        )
        decision = (
            blocked
            if blocked is not None
            else await self.completion_bus.check_final_response(ctx)
        )
        loop_state.last_completion_decision = decision
        if decision.status == CANCELLED:
            raise asyncio.CancelledError()
        if decision.status == COMPLETED:
            result = loop_state.last_result
            loop_state.finish_reason = (
                result.finish_reason if result else None
            ) or STOP
        elif decision.status == WAITING:
            loop_state.finish_reason = WAITING
        else:
            content = "; ".join(decision.reasons) or "Task not yet complete."
            ctx.messages.append(SystemMessage(source=HARNESS, content=content))
        return decision

    def _run_guards(
        self, ctx: RunContext, loop_state: ReActLoopState
    ) -> list[SystemMessage] | None:
        """Run every guard's ``after_tool_round``, collect any steering.

        Returned messages are meant for ``transient_messages`` on the next
        model call only — never written to ``ctx.messages``.
        """
        if not self.guards:
            return None
        guard_ctx = GuardContext(
            tools=self.tool_catalog, max_loop_iterations=self.max_loop_iterations
        )
        texts = [
            steering
            for guard in self.guards
            if (steering := guard.after_tool_round(ctx, loop_state, guard_ctx))
        ]
        if not texts:
            return None
        return [SystemMessage(source=HARNESS, content=" ".join(texts))]

    @staticmethod
    def _denied_records(records: list[ToolCallRecord]) -> list[ToolCallRecord]:
        """Perform the internal ``denied records`` operation for ``ReactLoop``.

Parameters
----------
records : list[ToolCallRecord]
    Value supplied for ``records``."""
        return [
            r
            for r in records
            if r.result is not None
            and r.result.failure_reason == FailureReason.APPROVAL_DENIED
        ]

    _TOOL_DENIED_FOLLOWUP_TOOLS: t.ClassVar[tuple[str, ...]] = (
        "ask_user",
        "update_plan",
    )

    def _tool_denied_followup(
        self, denied: list[ToolCallRecord]
    ) -> tuple[list[SystemMessage], list[t.Any]] | None:
        """One constrained model call after a denial, instead of a silent pause.

        Restricts the next call to ``ask_user``/``update_plan`` only — the
        model cannot reopen the approval chain it was just denied, only
        acknowledge the denial to the user or close out the plan. Returns
        ``None`` if ``ask_user`` isn't registered on this agent: without it
        the model has no way to ask anything, so the caller falls back to
        the old hard pause instead of risking an unconstrained call.
        """
        catalog = self.tool_catalog
        if "ask_user" not in catalog:
            return None
        tools = [
            catalog[name]
            for name in self._TOOL_DENIED_FOLLOWUP_TOOLS
            if name in catalog
        ]
        return [SystemMessage(source=HARNESS, content=harness.tool_denied(denied))], tools

    def _reasoning_complete(self, loop_state: ReActLoopState) -> ReasoningCompleteEvent:
        """Build the terminal event for the current turn."""
        return ReasoningCompleteEvent(
            source=self.name,
            finish_reason=loop_state.finish_reason or "unknown",
            total_iterations=loop_state.iteration,
        )
