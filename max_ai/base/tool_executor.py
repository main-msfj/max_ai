"""
Tool executor for agent runs.

Resolves, approves, validates and executes tool calls. The executor is
side-effect-free with respect to the LLM transcript — it never appends
to ``llm_messages``. The caller (``ReActLoop`` or ``Agent``)
collects ``ToolMessage`` instances from this generator and decides
where they go.

Tool calls flow as ``ToolCallRecord`` instances (already tracked in
``ctx.tool_state``). Their status drives dispatch:

- ``PENDING_APPROVAL`` + tool requires approval → emit
  ``ToolApprovalEvent`` and return; caller pauses the run.
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
import typing as t
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from pydantic import ValidationError

from .executor import CoreExecutor
from ..base.tools import CoreTool, ToolContext
from ..base.middleware import CoreMiddleware
from ..middleware.chain import MiddlewareChain
from ..executor.local import LocalExecutor

from ..loggers import ScopedLogger
from ..termination import CancellationToken

from ..types.run_context import RunContext
from ..types.tools import ToolApprovalMode
from ..types.tool_call import ToolCallRecord, ToolResult

from ..core.messages import ToolMessage
from ..core.event_type import (
    CoreEvent,
    ToolCallEvent,
    ToolApprovalEvent,
    ToolCallResponseEvent,
)


logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="ToolExecutor")


# What the executor yields back to the caller.
ToolExecutorYield = t.Union[ToolMessage, CoreEvent]


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
    ) -> None:
        """Initialize the executor.

        Args:
            tools: Tools available to the agent. Indexed by name.
            middlewares: Middleware applied around each individual
                tool execution.
            agent_name: Used for logging and as the ``source`` field
                of emitted events and ``ToolMessage`` instances.
            max_concurrent_tools: Bound on parallel tool execution.
            executor: Strategy that actually runs the tool. Defaults
                to ``LocalExecutor``. Choosing a different strategy
                (Docker, MCP, etc.) changes where every tool of this
                agent runs — the choice is per-agent, not per-tool.
        """
        self.tools: dict[str, CoreTool] = {t.name: t for t in (tools or [])}
        self.mw_chain = MiddlewareChain(middlewares=middlewares or [])
        self.agent_name = agent_name
        self.max_concurrent_tools = max_concurrent_tools
        self.executor: CoreExecutor = executor or LocalExecutor()

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

        # 3. Resolve the tool.
        resolution = self._resolve_tool(record)
        if resolution.tool is None:
            _log.error("Tool not found in registry", err=resolution.error_msg)
            async for item in self._yield_failure(
                record, error=resolution.error_msg or "Unknown tool"
            ):
                yield item
            return
        tool = resolution.tool

        # 4. Observability: tool call begins.
        yield ToolCallEvent(
            source=self.agent_name,
            tool_name=record.tool_name,
            parameters=record.parameters,
            tool_call_id=record.id,
        )

        # 5. Evaluate approval.
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

        # 6. Validate parameters against the tool's JSON schema.
        validation = tool.validate_parameters(record)
        if not validation.is_tool_valid:
            err = f"Parameter validation failed: {validation.msg_error}"
            _log.error(err)
            async for item in self._yield_failure(record, error=err):
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
        tool_ctx = ToolContext(
            run_id=ctx.run_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id or "",
        )

        async def func(rec: ToolCallRecord) -> ToolResult:
            return await self.executor.run(tool, rec, tool_ctx, cancellation_token)

        result: ToolResult | None = None
        try:
            async for item in self.mw_chain.execute(
                action=self._ACTION,
                ctx=ctx,
                data=record,
                func=func,
            ):
                if isinstance(item, CoreEvent):
                    yield item
                    continue
                if isinstance(item, ToolResult):
                    result = item
                    continue
                _log.warning(
                    "Middleware chain yielded unexpected item type",
                    item_type=type(item).__name__,
                )
        except Exception as e:
            _log.error("Middleware chain failed", err=str(e))
            result = ToolResult.execution_error(
                record.id, f"Middleware chain failure: {e}"
            )

        if result is None:
            _log.error("Middleware chain produced no result")
            result = ToolResult.execution_error(record.id, "Tool produced no result.")

        async for item in self._consume_and_yield(record, result):
            yield item

    # async def _invoke_tool(
    #     self,
    #     tool: CoreTool,
    #     record: ToolCallRecord,
    #     ctx: RunContext,
    #     cancellation_token: CancellationToken | None,
    # ) -> ToolResult:
    #     """Actually call ``tool.execute(...)`` with timeout + cancellation + try/except.

    #     Returns a ``ToolResult`` — never raises (every exception is
    #     captured and turned into a ``tool_failure``).
    #     """
    #     timeout = tool.timeout_seconds or self.waiting_timeout
    #     tool_ctx = ToolContext(
    #         run_id=ctx.run_id,
    #         session_id=ctx.session_id or "",
    #     )

    #     try:
    #         task = asyncio.create_task(
    #             tool.execute(record, tool_ctx, cancellation_token)
    #         )
    #         if cancellation_token is not None:
    #             cancellation_token.link_future(task)
    #         return await asyncio.wait_for(task, timeout=timeout)

    #     except asyncio.TimeoutError:
    #         return ToolResult.timeout(record.id, timeout_seconds=timeout)

    #     except asyncio.CancelledError:
    #         return ToolResult.cancelled_during_execution(record.id)

    #     except ValidationError as ve:
    #         return ToolResult.invalid_parameters(record.id, str(ve))

    #     except Exception as e:
    #         return ToolResult.execution_error(record.id, str(e))

    # -------- PARALLEL EXECUTION -----------------------------------------------------------
    async def _execute_parallel(
        self,
        ctx: RunContext,
        records: list[ToolCallRecord],
        cancellation_token: CancellationToken | None,
    ) -> AsyncGenerator[ToolExecutorYield, None]:
        """Run multiple records in parallel with bounded concurrency.

        Each task buffers its yielded items so the overall stream stays
        ordered by the input ``records`` list — UI / LLM never sees
        events from tool B interleaved into tool A's lifecycle.

        The timeout for each tool starts when it acquires the semaphore.
        Time spent waiting for a slot does NOT consume the timeout.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_tools)

        async def runner(rec: ToolCallRecord) -> list[ToolExecutorYield]:
            buffer: list[ToolExecutorYield] = []
            async with semaphore:
                async for item in self._process_record(ctx, rec, cancellation_token):
                    buffer.append(item)
            return buffer

        tasks = [asyncio.create_task(runner(r)) for r in records]
        results: list[list[ToolExecutorYield] | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

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

    # -------- HELPERS -----------------------------------------------------------
    async def _yield_failure(
        self,
        record: ToolCallRecord,
        error: str,
    ) -> AsyncGenerator[ToolExecutorYield, None]:
        """Emit a failure for a record that never reached execution.

        Used when the executor short-circuits before invoking the tool:
        tool not found, validation failed, rejected by user, cancelled
        pre-execution, runner crashed. The record is force-consumed with
        a failure ``ToolResult`` so it lands in a terminal state and
        doesn't leak back into ``actionable_calls`` or ``pending_approvals``.
        """
        result = ToolResult.tool_failure(record.id, error=error)

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
