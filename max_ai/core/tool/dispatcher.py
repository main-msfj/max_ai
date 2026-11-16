"""ToolDispatcher: where the registry and the sandbox meet. It receives a
tool call, checks it, waits for approval, runs it and closes its record."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterable

from ...base.middleware import MiddlewareContext, ToolRequest
from ...base.tools import CoreTool, ToolContext
from ...loggers import ScopedLogger
from ...types.tool_call import ToolCallRecord, ToolResult
from ..event_type import (
    CoreEvent,
    ToolApprovalEvent,
    ToolAutoApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
    UserInputRequestEvent,
)
from ..messages import ToolMessage
from ..termination import CancellationToken
from .hook import ToolHook
from .registry import ToolRegistry

if TYPE_CHECKING:
    from ...base.sandbox import SandboxCore
    from ...types.run_context import RunContext
    from ..middleware.chain import MiddlewareChain

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["ToolDispatcher"])

Emit = Callable[[Any], None]

# What the model reads. General on purpose: who denied and why is the hook's to say.
DENIED = "The tool call was denied."
INTERRUPTED = (
    "The tool call was interrupted and may not have finished. Check before calling it again."
)
CANNOT_RUN = "The tool call could not be run. Call the tool again if you still need it."
NOT_ANSWERED = "The user did not answer."
CANCELLED = "The user cancelled the call."


class ToolDispatcher:
    """Runs the calls of one user: build it with the tools and that user's
    sandbox (``sandbox.bind(identifier)``); without a sandbox it runs the
    tools that work on the host.

    ``hook`` is asked at the four moments of a call: before, permission,
    after and failed. ``middleware`` watches what the tools answer. A call
    that waits for a person returns None: the application answers with
    ``record.approve()`` / ``record.reject()`` and dispatches it again.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        sandbox: SandboxCore | None = None,
        *,
        hook: ToolHook | None = None,
        middleware: MiddlewareChain | None = None,
        deps: dict[str, Any] | None = None,
        timeout: float = 360,
        max_parallel: int = 8,
    ) -> None:
        self.registry = registry
        self.sandbox = sandbox
        self.hook = hook or ToolHook()
        self.middleware = middleware
        self.source = "tool_dispatcher"  # who the events say they come from
        self.deps = deps or {}  # the developer's own objects, handed to every tool
        self.timeout = timeout  # seconds one call may run
        self.max_parallel = max_parallel  # read-only calls that may run at once
        self._active: set[str] = set()

    # -------- MANY CALLS ---------------------------------------------------------
    async def dispatch_many(
        self,
        records: Iterable[ToolCallRecord],
        ctx: RunContext,
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncIterator[CoreEvent | ToolMessage]:
        """Stream tool events and transcript results, preserving call order.
        Read-only calls next to each other run at once."""
        # None marks the end of the stream.
        queue: asyncio.Queue[CoreEvent | ToolMessage | None] = asyncio.Queue()
        limit = asyncio.Semaphore(self.max_parallel)

        async def run(record: ToolCallRecord) -> ToolResult | None:
            async with limit:
                return await self.dispatch(record, ctx, queue.put_nowait, cancellation_token)

        async def produce() -> None:
            try:
                for batch in self._batches(list(records)):
                    if cancellation_token is not None and cancellation_token.is_cancelled():
                        raise asyncio.CancelledError()
                    results = await asyncio.gather(*(run(record) for record in batch))
                    # Results enter the transcript in call order, whatever finished first.
                    for record, result in zip(batch, results):
                        if result is not None:
                            queue.put_nowait(self.tool_message(record, result))
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(produce())
        try:
            while (item := await queue.get()) is not None:
                yield item
            await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _batches(self, records: list[ToolCallRecord]) -> list[list[ToolCallRecord]]:
        """Consecutive read-only calls form one batch that runs at once; any
        other call runs alone, so "write A, read A" keeps its order."""
        batches: list[list[ToolCallRecord]] = []
        previous_read_only = False
        for record in records:
            tool = self.registry.get(record.tool_name)
            read_only = tool is not None and tool.read_only
            if read_only and previous_read_only:
                batches[-1].append(record)
            else:
                batches.append([record])
            previous_read_only = read_only
        return batches

    # -------- THE CONVERSATION ---------------------------------------------------
    @staticmethod
    def tool_message(record: ToolCallRecord, result: ToolResult) -> ToolMessage:
        """The message that answers ``record``'s call in the conversation."""
        return ToolMessage(
            source=record.tool_name,
            tool_call_id=record.id,
            tool_name=record.called_as or record.tool_name,
            success=result.success,
            error=result.error or None,
            content=(
                json.dumps(result.result, ensure_ascii=False, default=str)
                if result.success
                else result.error or "Tool failed"
            ),
        )

    @classmethod
    def close_unanswered(cls, ctx: RunContext) -> None:
        """Close every open question. Call it before a new user message:
        each tool call needs its answer first, or the provider rejects it."""
        for record in ctx.tool_state.pending_user_input:
            result = ToolResult.tool_failure(record.id, NOT_ANSWERED)
            record.force_consume(result)
            ctx.messages.append(cls.tool_message(record, result))

    @classmethod
    def answer_unanswered(cls, ctx: RunContext, records: Iterable[ToolCallRecord]) -> None:
        """Give each of ``records`` the ToolMessage it lacks. A call that never
        started is closed as cancelled, so it does not run on the next turn."""
        answered = {m.tool_call_id for m in ctx.messages if isinstance(m, ToolMessage)}
        for record in records:
            if record.id in answered:
                continue
            if not record.is_consumed and not record.is_executing:
                record.force_consume(record.result or ToolResult.cancelled_before_start(record.id))
            if record.result is not None:
                ctx.messages.append(cls.tool_message(record, record.result))

    @classmethod
    def close_interrupted(cls, ctx: RunContext) -> None:
        """The user stopped the turn: every open call gets its answer, so
        the conversation can go on."""
        cls.close_unanswered(ctx)
        state = ctx.tool_state
        open_calls = [r for r in state.records.values() if not r.is_consumed]
        for record in open_calls:
            if record.is_executing:
                state.fail_stale(record.id, CANCELLED)
        cls.answer_unanswered(ctx, open_calls)

    # -------- ONE CALL -----------------------------------------------------------
    async def dispatch(
        self,
        record: ToolCallRecord,
        ctx: RunContext,
        emit: Emit | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult | None:
        """Run one call. None means it waits for a person: approve or answer
        it and dispatch it again."""
        emit = emit or _silent
        if record.is_consumed:
            return record.result
        # This same call is running right now: that run will answer it.
        if record.id in self._active:
            log.child(tool_call_id=record.id).warning("Tool call dispatched while it runs")
            return None
        # Left running by a run that died: nobody will finish it, so close it.
        if record.is_executing:
            record.mark_stale_as_failed(INTERRUPTED)
            log.child(tool_call_id=record.id).warning("Tool call left running was closed")
            assert record.result is not None
            emit(
                ToolCallResponseEvent(
                    source=self.source, tool_call_id=record.id, tool_result=record.result
                )
            )
            return record.result
        self._active.add(record.id)
        try:
            return await self._dispatch(record, ctx, emit, cancellation_token)
        finally:
            self._active.discard(record.id)

    async def _dispatch(
        self,
        record: ToolCallRecord,
        ctx: RunContext,
        emit: Emit,
        cancellation_token: CancellationToken | None,
    ) -> ToolResult | None:
        emit(
            ToolCallEvent(
                source=self.source,
                tool_name=record.tool_name,
                parameters=record.parameters,
                tool_call_id=record.id,
            )
        )
        if record.is_rejected:
            reason = record.approval_reason or "User declined approval"
            return self._finish(record, ToolResult.approval_denied(record.id, reason), emit)
        if cancellation_token is not None and cancellation_token.is_cancelled():
            return self._finish(record, ToolResult.cancelled_before_start(record.id), emit)

        # Can this call run at all?
        tool = self._check(record)
        if isinstance(tool, ToolResult):
            return self._finish(record, tool, emit)
        tool_ctx = ToolContext(ctx, self.sandbox, emit, dict(self.deps))

        # 1. before: the hook may stop the call here.
        if (answered := await self.hook.before(tool, record, tool_ctx)) is not None:
            return self._finish(record, answered, emit)

        # 2. permission: run it, deny it, or wait for the user.
        ready = await self._permission(tool, record, tool_ctx, emit)
        if isinstance(ready, ToolResult):
            return self._finish(record, ready, emit)
        if not ready:
            return None

        # 3 and 4. Run it; the hook sees how it went.
        result = await self._execute(tool, record, tool_ctx, emit, cancellation_token)
        react = self.hook.after if result.success else self.hook.failed
        result = await react(tool, record, tool_ctx, result)
        return self._finish(record, result, emit)

    def _check(self, record: ToolCallRecord) -> CoreTool | ToolResult:
        """The tool this call names, or why the call cannot run."""
        tool = self.registry.get(record.tool_name)
        if tool is None:
            return ToolResult.execution_error(record.id, "Unknown tool")
        try:
            validation = tool.validate_parameters(record)
        except (ValueError, TypeError) as error:
            return ToolResult.invalid_parameters(record.id, str(error))
        if not validation.is_tool_valid:
            return ToolResult.invalid_parameters(
                record.id, validation.msg_error or "Invalid arguments"
            )
        if tool.sandbox and self.sandbox is None:
            return ToolResult.execution_error(record.id, f"{tool.name} needs a sandbox")
        return tool

    async def _permission(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext, emit: Emit
    ) -> bool | ToolResult:
        """True: run it now. False: it waits for a person. A result: denied."""
        # A person will see this call anyway: no approval on top of it.
        if tool.waits_for_user and record.result is None:
            if not record.is_awaiting_input:
                record.await_user_input()
            emit(
                UserInputRequestEvent(
                    source=self.source,
                    tool_call_id=record.id,
                    tool_name=record.tool_name,
                    parameters=record.parameters,
                )
            )
            return False
        # Asked once per call: after the user approves, the record no longer waits.
        if record.is_pending_approval:
            answer = await self.hook.permission(tool, record, tool_ctx)
            if answer is None:
                emit(
                    ToolApprovalEvent(
                        source=self.source,
                        tool_name=record.tool_name,
                        parameters=record.parameters,
                        tool_call_id=record.id,
                        reason_for_approval=(
                            record.parameters.get("description")
                            or f"Approval required for '{record.tool_name}'"
                        ),
                    )
                )
                return False
            if answer is False:
                return ToolResult.approval_denied(record.id, DENIED)
            record.auto_approve()
            emit(
                ToolAutoApprovalEvent(
                    source=self.source, tool_call_id=record.id, tool_name=record.tool_name
                )
            )
        if not record.is_actionable:
            return ToolResult.execution_error(record.id, CANNOT_RUN)
        return True

    async def _execute(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_ctx: ToolContext,
        emit: Emit,
        cancellation_token: CancellationToken | None,
    ) -> ToolResult:
        """Run the tool, with the middleware around it. Only calls that
        passed ``before`` and ``permission`` get here."""
        # Middleware sees the call about to run; it may answer for it.
        chain = self.middleware
        mw = request = None
        if chain:
            mw = MiddlewareContext(ctx=tool_ctx.run_context, agent=self.source, emit=emit)
            request = ToolRequest(record=record)
            if (blocked := await chain.tool_request(mw, request)) is not None:
                return blocked

        async def invoke() -> object:
            # object: a user's tool may return anything; checked below.
            if record.result is not None:  # a person already produced it
                return record.result
            return await tool.execute(record, tool_ctx, cancellation_token)

        record.start_execution()
        limit = self.timeout
        try:
            task = asyncio.create_task(invoke())
            if cancellation_token is not None:
                cancellation_token.link_future(task)
            # A tool stops itself at its own limit (bash inside the sandbox);
            # this is only the net for one that never returns.
            limit = getattr(tool, "timeout", None) or self.timeout
            result = await asyncio.wait_for(task, timeout=limit + 30)
            if not isinstance(result, ToolResult) or result.tool_call_id != record.id:
                result = ToolResult.execution_error(record.id, "Invalid tool result")
        except asyncio.TimeoutError:
            result = ToolResult.timeout(record.id, limit)
        except asyncio.CancelledError:
            # Not a hook's business: close the record and let the cancel go on.
            self._finish(record, ToolResult.cancelled_during_execution(record.id), emit)
            raise
        except Exception as error:
            result = ToolResult.execution_error(record.id, str(error))
        if chain and mw is not None and request is not None:
            result = await chain.tool_response(mw, request, result)
        return result

    def _finish(self, record: ToolCallRecord, result: ToolResult, emit: Emit) -> ToolResult:
        """Close the record with ``result`` and tell the outside."""
        # The result is built after the tool ran: stamp the real start and end.
        now = datetime.now(timezone.utc)
        result = result.model_copy(
            update={"started_at": record.started_at or now, "completed_at": now}
        )
        if record.is_executing:
            record.mark_consumed(result)
        else:
            record.force_consume(result)
        emit(ToolCallResponseEvent(source=self.source, tool_call_id=record.id, tool_result=result))
        return result


def _silent(event: Any) -> None:
    """Nobody listens: used when a call is dispatched without ``emit``."""
