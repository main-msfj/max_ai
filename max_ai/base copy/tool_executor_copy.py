"""
Tool executor for agent runs.

Resolves, approves, validates and executes tool calls. The executor is
side-effect-free with respect to the LLM transcript — it never appends
to ``llm_messages``. The caller (``ReActLoopSelfDirected`` or ``Agent``)
collects ``ToolMessage`` instances from this generator and decides
where they go.

Tool calls flow as ``ToolCallRecord`` instances (already tracked in
``ctx.tool_state``). Their status drives dispatch:

- ``PENDING_APPROVAL`` + tool requires approval → emit
  ``ToolApprovalEvent`` and return; caller pauses the run.
- the native ask-the-user tool → never executed: fresh records move to
  ``INPUT_NEEDED`` + emit ``UserInputRequestEvent`` (caller pauses);
  answered records (``apply_user_answer``) complete from the stored
  answer.
- ``PENDING_APPROVAL`` + tool is auto-approve → ``auto_approve()``
  the record, then proceed.
- ``APPROVED`` / ``AUTO_APPROVED`` → ``start_execution()`` then run.
- ``REJECTED`` → emit a failure result, never run.
- ``EXECUTING`` (stale) → caller must resolve via ``fail_stale``
  before passing here. The executor refuses to act on stale records.
- ``CONSUMED`` → caller filtered, should not arrive here.

Parallel execution uses a bounded semaphore. Each tool's timeout
counts only from the moment it acquires the semaphore — time spent
waiting for a slot does not consume the timeout.

Middleware integration:

The middleware pipeline wraps the **execution step only**
(``tool.execute()``). Resolution, approval evaluation, and
parameter validation happen outside the chain so middleware can
observe / transform the actual side-effecting call without seeing
the boilerplate. Each tool execution gets its own ``execute()`` call
on the chain with ``action="tool_call"`` and ``data=ToolCallRecord``.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import typing as t
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from ..base.middleware import CoreMiddleware
from ..base.tools import CoreRuntimeTool, CoreTool, ToolContext
from ..core.event_type import (
    CoreEvent,
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
    ToolProgressEvent,
    UserInputRequestEvent,
)
from ..core.messages import ToolMessage
from ..core.primitives import FailureReason
from ..executor.local import LocalExecutor
from ..loggers import ScopedLogger
from ..middleware.chain import MiddlewareChain
from ..termination import CancellationToken
from ..types.run_context import RunContext
from ..types.tool_call import ToolCallRecord, ToolResult
from ..types.tools import ToolApprovalMode
from .environment import Environment
from .executor import CoreExecutor

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["ToolExecutor"])


# What the executor yields back to the caller.
ToolExecutorYield = t.Union[ToolMessage, CoreEvent]

# Name of the native ask-the-user tool. Kept as a string constant (not an
# import of AskUserTool) to avoid a tools -> base -> tools import cycle;
# a test asserts it matches AskUserTool.TOOL_NAME. The legacy set covers
# records persisted before the rename so paused runs still resume.
USER_INPUT_TOOL_NAME = "ask_user"
USER_INPUT_TOOL_NAMES = frozenset({USER_INPUT_TOOL_NAME, "structure_human_in_loop"})


def _frame_user_answer(record: ToolCallRecord) -> str:
    """Turn a stored ask_user answer into a self-describing tool result.

    A raw answer like ``"Vuelo"`` arrives at the model as a bare tool
    message with no framing; small models then misread free-form replies
    as invalid ("your answer blocks me") or forget which question it
    answered. Deterministic framing fixes that: the result restates the
    question, and a reply that is not one of the offered options is
    explicitly labeled as a valid free-form answer to interpret.
    """
    answer = (record.user_answer or "").strip()
    question = (
        record.input_question
        or str(record.parameters.get("question") or "")
    ).strip()
    q_part = f" to your question {question!r}" if question else ""

    if not answer:
        return (
            f"[framework] The user skipped this question"
            f"{f' ({question!r})' if question else ''} without answering. "
            "Do not re-ask it — proceed using your best judgment and the "
            "context you already have."
        )

    options = record.input_options or []
    # Options render as "Label — description"; the UI submits the label.
    labels = {o.strip() for o in options}
    labels |= {o.split("—", 1)[0].strip() for o in options}
    if options and answer not in labels:
        return (
            f"The user replied in their own words{q_part}: {answer!r}\n"
            "[framework] This free-form reply IS the answer. Interpret it "
            "in context and continue the task — do not treat it as "
            "invalid, do not say it blocks you, and do not re-ask."
        )
    return f"The user's answer{q_part}: {answer!r}"


# -------- INTERNAL OUTCOMES -----------------------------------------------------------
@dataclass
class _ResolutionOutcome:
    """Result of looking up a tool by name."""

    tool: CoreTool | None
    error_msg: str | None = None


@dataclass
class _ApprovalDecision:
    """What the executor should do with a record's approval state."""

    proceed: bool
    pending_event: ToolApprovalEvent | None = None
    rejection_msg: str | None = None


# -------- TOOL EXECUTOR -----------------------------------------------------------
class ToolExecutor:
    _ACTION = "tool_call"

    def __init__(
        self,
        tools: list[CoreTool] | None = None,
        middlewares: list[CoreMiddleware] | None = None,
        agent_name: str = "unknown",
        max_concurrent_tools: int = 5,
        executor: CoreExecutor | None = None,
        runtime_executor: CoreExecutor | None = None,
        runtime_deps: dict[str, t.Any] | None = None,
        environment: Environment | None = None,
    ) -> None:
        """Initialize the executor.

        Args:
            tools: Tools available to the agent. Indexed by name.
            middlewares: Middleware applied around each individual
                tool execution.
            agent_name: Used for logging and as the ``source`` field
                of emitted events and ``ToolMessage`` instances.
            max_concurrent_tools: Bound on parallel tool execution.
            executor: Strategy for normal local tools. Defaults to
                ``LocalExecutor``.
            runtime_executor: Strategy for ``CoreRuntimeTool`` instances
                such as bash. Defaults to ``executor``.
        """
        self.environment = environment
        self.tools: dict[str, CoreTool] = {t.name: t for t in (tools or [])}
        self.mw_chain = MiddlewareChain(middlewares=middlewares or [])
        self.agent_name = agent_name
        self.max_concurrent_tools = max_concurrent_tools
        self.executor: CoreExecutor = executor or LocalExecutor()
        self.runtime_executor: CoreExecutor = runtime_executor or self.executor
        self.runtime_deps = dict(runtime_deps or {})

    # -------- PUBLIC ENTRY POINT -----------------------------------------------------------
    async def execute_tool_call(
        self,
        ctx: RunContext,
        records: list[ToolCallRecord],
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[ToolExecutorYield, None]:
        """Execute the given tool call records.

        Single record → sequential. Multiple → bounded parallel.

        Yields:
            ``ToolMessage`` (back to the LLM transcript) and ``CoreEvent``
            (observability / UI).
        """
        if not records:
            return

        if len(records) == 1:
            async for item in self._process_record(ctx, records[0], cancellation_token):
                yield item
            return

        async for item in self._execute_parallel(ctx, records, cancellation_token):
            yield item

    # -------- SEQUENTIAL FLOW -----------------------------------------------------------
    async def _process_record(
        self,
        ctx: RunContext,
        record: ToolCallRecord,
        cancellation_token: CancellationToken | None,
    ) -> AsyncGenerator[ToolExecutorYield, None]:
        """Run the full pipeline for a single record."""
        _log = log.child(
            agent_name=self.agent_name,
            run_id=ctx.run_id,
            tool_name=record.tool_name,
            tool_call_id=record.id,
        )

        # 1. Refuse stale records — caller must fail_stale them first.
        if record.is_stale_execution:
            _log.error("Refusing to execute stale record")
            async for item in self._yield_failure(
                record,
                error=(
                    "Tool call is in EXECUTING from a previous run. "
                    "Caller must resolve via ToolState.fail_stale() first."
                ),
            ):
                yield item
            return

        # 2. Refuse already-consumed records (caller should have filtered).
        if record.is_consumed:
            _log.warning("Skipping already-consumed record")
            return

        # 3. A rejected approval must never look like a real execution.
        # It still becomes a ToolMessage so the LLM can explain that the
        # requested action was not performed.
        if record.is_rejected:
            async for item in self._yield_failure(
                record,
                error=record.approval_reason or "User declined approval",
            ):
                yield item
            return

        # 3b. Native ask-the-user tool — elicitation is state, never
        # execution (mirrors the approval pause). Fresh record → mark
        # INPUT_NEEDED + emit UserInputRequestEvent and pause; answered
        # record (post apply_user_answer) → synthesize the result from
        # the stored answer. The tool's execute() is never invoked, so
        # no executor timeout can kill a waiting question.
        if record.tool_name in USER_INPUT_TOOL_NAMES:
            async for item in self._handle_user_input_record(record):
                yield item
            return

        coerced_skill_name = self._coerce_bash_skill_name_to_read_skill(record)
        if coerced_skill_name is not None:
            _log.info(
                "Coerced bash skill-name command into read_skill",
                skill_name=coerced_skill_name,
            )

        skill_name_misuse = self._skill_name_misuse(record)
        if skill_name_misuse is not None:
            async for item in self._yield_skill_name_misuse(record, skill_name_misuse):
                yield item
            return

        # 3. Observability: the model attempted a tool call. Emit this
        # before resolution so missing-tool failures still have a visible
        # call event in traces/UI.
        if record.tool_name != "bash":
            yield ToolCallEvent(
                source=self.agent_name,
                tool_name=record.tool_name,
                parameters=record.parameters,
                tool_call_id=record.id,
            )

        # 4. Resolve the tool.
        resolution = self._resolve_tool(record)
        if resolution.tool is None:
            _log.error("Tool not found in registry", err=resolution.error_msg)
            async for item in self._yield_failure(
                record, error=resolution.error_msg or "Unknown tool"
            ):
                yield item
            return
        tool = resolution.tool

        # 5. Validate parameters against the tool's JSON schema before
        # asking for approval. Invalid or policy-blocked calls should not
        # create an approval prompt.
        validation = tool.validate_parameters(record)
        if not validation.is_tool_valid:
            err = f"Parameter validation failed: {validation.msg_error}"
            _log.error(err)
            async for item in self._yield_failure(
                record, error=err, reason=FailureReason.INVALID_PARAMETERS
            ):
                yield item
            return

        # 6. Evaluate approval.
        decision = self._evaluate_approval(tool, record)
        if decision.pending_event is not None:
            yield decision.pending_event
            return  # Pause: caller waits for user approval response.
        if not decision.proceed:
            async for item in self._yield_failure(
                record, error=decision.rejection_msg or "Rejected"
            ):
                yield item
            return

        # 7. Cancellation check before doing anything expensive.
        if cancellation_token and cancellation_token.is_cancelled():
            _log.info("Cancelled before execution")
            record.start_execution()  # so mark_consumed contract holds
            async for item in self._consume_and_yield(
                record,
                ToolResult.cancelled_before_start(record.id),
            ):
                yield item
            return

        if tool.name == "bash":
            yield ToolProgressEvent(
                source=self.agent_name,
                tool_name=tool.name,
                content="Running skill",
                tool_call_id=record.id,
            )

        # 8. Run the tool through middleware.
        async for item in self._run_through_middleware(
            ctx, tool, record, cancellation_token
        ):
            yield item

    # -------- TOOL RESOLUTION -----------------------------------------------------------
    def _resolve_tool(self, record: ToolCallRecord) -> _ResolutionOutcome:
        tool = self.tools.get(record.tool_name)
        if tool is not None:
            return _ResolutionOutcome(tool=tool)
        return _ResolutionOutcome(
            tool=None,
            error_msg=f"Tool '{record.tool_name}' not found in tool registry.",
        )

    def _coerce_bash_skill_name_to_read_skill(
        self, record: ToolCallRecord
    ) -> str | None:
        skill_names = self.runtime_deps.get("skill_names") or []
        if not isinstance(skill_names, list):
            return None

        if record.tool_name != "bash":
            return None
        command = record.parameters.get("command")
        if not isinstance(command, str):
            return None
        normalized = command.strip().strip('\'"')
        skill_name = normalized if normalized in skill_names else None
        if skill_name is None:
            try:
                first_token = shlex.split(command, posix=True)[0]
            except (IndexError, ValueError):
                first_token = ""
            if first_token in skill_names:
                skill_name = first_token

        if skill_name is not None:
            record.parameters["command"] = f"read_skill {skill_name}"
            record.parameters.pop("args", None)
            return skill_name
        return None

    def _skill_name_misuse(self, record: ToolCallRecord) -> str | None:
        skill_names = self.runtime_deps.get("skill_names") or []
        if not isinstance(skill_names, list):
            return None

        if record.tool_name in skill_names:
            return record.tool_name
        return None

    # -------- APPROVAL EVALUATION -----------------------------------------------------------
    def _evaluate_approval(
        self, tool: CoreTool, record: ToolCallRecord
    ) -> _ApprovalDecision:
        """Decide what to do with this record's approval state."""
        # Auto-approval path.
        if tool.approval_mode != ToolApprovalMode.ASK_APPROVED:
            if record.is_pending_approval:
                record.auto_approve()
            return _ApprovalDecision(proceed=True)

        # Manual approval required.
        if record.is_pending_approval:
            return _ApprovalDecision(
                proceed=False,
                pending_event=ToolApprovalEvent(
                    source=self.agent_name,
                    tool_name=record.tool_name,
                    parameters=record.parameters,
                    tool_call_id=record.id,
                    reason_for_approval=(
                        f"Approval required for tool '{record.tool_name}'"
                    ),
                ),
            )

        if record.is_rejected:
            return _ApprovalDecision(
                proceed=False,
                rejection_msg=(record.approval_reason or "User declined approval"),
            )

        if record.is_actionable:
            return _ApprovalDecision(proceed=True)

        # Unexpected: caller should have filtered consumed/executing.
        return _ApprovalDecision(
            proceed=False,
            rejection_msg=f"Cannot proceed from status {record.status}",
        )

    # -------- TOOL EXECUTION (THROUGH MIDDLEWARE) -----------------------------------------------------------
    async def _run_through_middleware(
        self,
        ctx: RunContext,
        tool: CoreTool,
        record: ToolCallRecord,
        cancellation_token: CancellationToken | None,
    ) -> AsyncGenerator[ToolExecutorYield, None]:
        _log = log.child(
            agent_name=self.agent_name,
            run_id=ctx.run_id,
            tool_name=record.tool_name,
            tool_call_id=record.id,
        )

        record.start_execution()

        # Build the per-call ToolContext once. The executor never sees
        # the full RunContext — it only gets what tools legitimately need.
        events: asyncio.Queue = asyncio.Queue()
        tool_ctx = ToolContext(
            run_id=ctx.run_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id or "",
            deps=dict(self.runtime_deps),
            emit_event=events.put_nowait,
            environment=self.environment,
        )

        async def func(rec: ToolCallRecord) -> ToolResult:
            executor = self._executor_for(tool)
            return await executor.run(tool, rec, tool_ctx, cancellation_token)

        async def produce():
            try:
                async for item in self.mw_chain.execute(
                    action=self._ACTION, ctx=ctx, data=record, func=func,
                ):
                    events.put_nowait(item)
            except Exception as error:
                events.put_nowait(ToolResult.execution_error(
                    record.id, f"Middleware chain failure: {error}"
                ))
            finally:
                events.put_nowait(None)

        result: ToolResult | None = None
        producer = asyncio.create_task(produce())
        try:
            while (item := await events.get()) is not None:
                if isinstance(item, CoreEvent):
                    yield item
                elif isinstance(item, ToolResult):
                    result = item
                else:
                    _log.warning("Unexpected tool stream item", item_type=type(item).__name__)
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

        if result is None:
            _log.error("Middleware chain produced no result")
            result = ToolResult.execution_error(record.id, "Tool produced no result.")

        async for item in self._consume_and_yield(record, result):
            yield item

    def _executor_for(self, tool: CoreTool) -> CoreExecutor:
        if isinstance(tool, CoreRuntimeTool):
            return self.runtime_executor
        return self.executor

    # -------- PARALLEL EXECUTION -----------------------------------------------------------
    async def _execute_parallel(
        self,
        ctx: RunContext,
        records: list[ToolCallRecord],
        cancellation_token: CancellationToken | None,
    ) -> AsyncGenerator[ToolExecutorYield, None]:
        """Run multiple records in parallel with bounded concurrency.

        UI events stream as they occur and carry call IDs. Tool messages stay
        ordered by the input records so the model transcript is deterministic.

        The timeout for each tool starts when it acquires the semaphore.
        Time spent waiting for a slot does NOT consume the timeout.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_tools)
        live_events: asyncio.Queue = asyncio.Queue()

        async def runner(rec: ToolCallRecord) -> list[ToolExecutorYield]:
            buffer: list[ToolExecutorYield] = []
            async with semaphore:
                async for item in self._process_record(ctx, rec, cancellation_token):
                    if isinstance(item, CoreEvent):
                        live_events.put_nowait(item)
                    else:
                        buffer.append(item)
            return buffer

        tasks = [asyncio.create_task(runner(r)) for r in records]
        async def collect():
            try:
                return await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                live_events.put_nowait(None)

        collector = asyncio.create_task(collect())
        try:
            while (event := await live_events.get()) is not None:
                yield event
            results = await collector
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if not collector.done():
                collector.cancel()
            await asyncio.gather(collector, *tasks, return_exceptions=True)

        for record, outcome in zip(records, results):
            if isinstance(outcome, BaseException):
                # The runner itself crashed (not a tool-level failure —
                # those are captured into a ToolResult). Surface as a
                # synthetic failure for this record.
                log.child(
                    agent_name=self.agent_name,
                    tool_name=record.tool_name,
                    tool_call_id=record.id,
                ).error("Parallel runner crashed", err=str(outcome))

                # If the record already moved past PENDING in the runner,
                # we may need to consume it; otherwise just emit failure.
                async for item in self._yield_failure(
                    record,
                    error=f"Parallel runner failure: {outcome}",
                ):
                    yield item
                continue

            for item in outcome:
                yield item

    # -------- ELICITATION (ask-the-user) -----------------------------------------------------------
    async def _handle_user_input_record(
        self,
        record: ToolCallRecord,
    ) -> AsyncGenerator[ToolExecutorYield, None]:
        """Resolve an ask-the-user record without executing anything.

        Three cases:

        - **Answered** (``user_answer`` set by ``apply_user_answer``):
          complete the call — the answer *is* the tool result. Emits the
          normal ``ToolMessage`` + ``ToolCallResponseEvent`` pair.
        - **Fresh** (``PENDING_APPROVAL``): transition to ``INPUT_NEEDED``
          with the question/options from the call parameters and emit
          ``UserInputRequestEvent``. The caller pauses the turn.
        - **Still waiting** (``INPUT_NEEDED``, re-dispatched without an
          answer): re-emit the event — idempotent pause.
        """
        if record.user_answer is not None:
            if record.is_pending_approval:
                # Defensive: answer arrived through a path that never went
                # through INPUT_NEEDED (e.g. pre-seeded). Approve so
                # start_execution() holds.
                record.auto_approve()
            record.start_execution()
            result = ToolResult.success_result(
                record.id, _frame_user_answer(record)
            )
            async for item in self._consume_and_yield(record, result):
                yield item
            return

        if record.is_pending_approval:
            # Sanitize model-supplied options: keep non-empty strings only;
            # an empty/invalid list degrades to a free-text question.
            raw_options = record.parameters.get("options")
            options = None
            if isinstance(raw_options, list):
                options = [
                    o for o in raw_options if isinstance(o, str) and o.strip()
                ] or None
            record.await_user_input(
                question=str(record.parameters.get("question") or ""),
                options=options,
            )

        yield UserInputRequestEvent(
            source=self.agent_name,
            tool_call_id=record.id,
            question=record.input_question or "",
            options=record.input_options,
        )

    # -------- HELPERS -----------------------------------------------------------
    async def _yield_failure(
        self,
        record: ToolCallRecord,
        error: str,
        reason: FailureReason = FailureReason.EXECUTION_ERROR,
    ) -> AsyncGenerator[ToolExecutorYield, None]:
        """Emit a failure for a record that never reached execution.

        Used when the executor short-circuits before invoking the tool:
        tool not found, validation failed, rejected by user, cancelled
        pre-execution, runner crashed. The record is force-consumed with
        a failure ``ToolResult`` so it lands in a terminal state and
        doesn't leak back into ``actionable_calls`` or ``pending_approvals``.
        """
        result = ToolResult.tool_failure(record.id, error=error, reason=reason)

        if record.is_executing:
            record.mark_consumed(result)
        elif not record.is_consumed:
            record.force_consume(result)

        yield ToolMessage.error_message(
            tool_call_id=record.id,
            tool_name=record.tool_name,
            error=error,
            source=self.agent_name,
        )
        yield ToolCallResponseEvent(
            source=self.agent_name,
            tool_call_id=record.id,
            tool_result=result,
        )

    async def _yield_skill_name_misuse(
        self,
        record: ToolCallRecord,
        skill_name: str,
    ) -> AsyncGenerator[ToolExecutorYield, None]:
        """Handle a model calling a skill capability id as a tool or shell command."""
        message = (
            "This skill capability was not executed. Retry by calling the bash "
            f"tool with command: read_skill {skill_name}. Do not mention "
            "this retry instruction to the user."
        )
        result = ToolResult.execution_error(
            record.id,
            message,
        )
        if not record.is_consumed:
            record.force_consume(result)

        yield ToolMessage.error_message(
            tool_call_id=record.id,
            tool_name=record.tool_name,
            error=message,
            source=self.agent_name,
        )
        yield ToolCallResponseEvent(
            source=self.agent_name,
            tool_call_id=record.id,
            tool_result=result,
        )

    async def _consume_and_yield(
        self,
        record: ToolCallRecord,
        result: ToolResult,
    ) -> AsyncGenerator[ToolExecutorYield, None]:
        """Consume the record with the result, then emit the LLM message + UI event."""
        record.mark_consumed(result)

        if result.success:
            content = (
                str(result.result)
                if result.result is not None
                else "Tool executed but returned no content."
            )
            yield ToolMessage.success_message(
                tool_call_id=record.id,
                tool_name=record.tool_name,
                content=content,
                source=self.agent_name,
            )
        else:
            yield ToolMessage.error_message(
                tool_call_id=record.id,
                tool_name=record.tool_name,
                error=result.error or "Tool execution failed.",
                source=self.agent_name,
            )

        yield ToolCallResponseEvent(
            source=self.agent_name,
            tool_call_id=record.id,
            tool_result=result,
        )
