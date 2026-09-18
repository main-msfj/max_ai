"""Tool approval and execution routing for the agent runtime."""

import asyncio
import json
from typing import TYPE_CHECKING

from .registry import ToolRegistry
from ...base.tools import ToolContext
from ..messages import ToolMessage
from ..event_type import (
    ToolApprovalEvent, ToolCallEvent, ToolCallResponseEvent, UserInputRequestEvent,
)
from ...termination import CancellationToken
from ...types.tool_call import ToolCallRecord, ToolResult
from ...types.tools import ToolApprovalMode

if TYPE_CHECKING:
    from ..environment.manager import EnvironmentManager


class ToolDispatcher:
    """Resolve, validate, request approval and invoke a tool adapter.

    Explicit host tools run in-process. Other tools run through the manager's
    executor and conversation session, without a host fallback.
    Approval is supplied by trusted application code via record.approve/reject.
    Pending approval returns None and leaves the record resumable.
    """

    def __init__(self, registry: ToolRegistry, *, source: str = "tool_dispatcher",
                 manager: "EnvironmentManager | None" = None):
        self.registry = registry
        self.source = source
        self.manager = manager
        self._active: set[str] = set()

    async def dispatch_many(self, records, context, cancellation_token=None):
        """Stream tool events and transcript results, preserving call order."""
        queue = asyncio.Queue()
        sentinel = object()
        call_context = ToolContext(
            context.run_id, session_id=context.session_id, user_id=context.user_id,
            retry_count=context.retry_count, deps=dict(context.deps),
            emit_event=queue.put_nowait,
        )

        async def produce():
            try:
                for record in records:
                    if cancellation_token is not None and cancellation_token.is_cancelled():
                        raise asyncio.CancelledError()
                    result = await self.dispatch(record, call_context, cancellation_token)
                    if result is not None:
                        queue.put_nowait(ToolMessage(
                            source=record.tool_name, tool_call_id=record.id,
                            tool_name=record.tool_name, success=result.success,
                            error=result.error,
                            content=json.dumps(result.result, ensure_ascii=False, default=str)
                            if result.success else result.error or "Tool failed",
                        ))
            finally:
                queue.put_nowait(sentinel)

        task = asyncio.create_task(produce())
        try:
            while (item := await queue.get()) is not sentinel:
                yield item
            await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def dispatch(
        self,
        record: ToolCallRecord,
        context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult | None:
        if record.id in self._active or record.is_executing:
            raise ValueError(
                "Tool call is already executing; resolve stale state first"
            )
        if record.is_consumed:
            return record.result
        self._active.add(record.id)
        try:
            return await self._dispatch(record, context, cancellation_token)
        finally:
            self._active.discard(record.id)

    async def _dispatch(self, record, context, cancellation_token):
        def emit(event):
            if context.emit_event is not None:
                context.emit_event(event)

        def finish(result):
            if record.is_executing:
                record.mark_consumed(result)
            else:
                record.force_consume(result)
            emit(
                ToolCallResponseEvent(
                    source=self.source, tool_call_id=record.id, tool_result=result
                )
            )
            return result

        emit(
            ToolCallEvent(
                source=self.source,
                tool_name=record.tool_name,
                parameters=record.parameters,
                tool_call_id=record.id,
            )
        )
        if record.is_rejected:
            return finish(
                ToolResult.execution_error(
                    record.id, record.approval_reason or "User declined approval"
                )
            )
        if cancellation_token is not None and cancellation_token.is_cancelled():
            return finish(ToolResult.cancelled_before_start(record.id))
        tool = self.registry.get(record.tool_name)
        if tool is None:
            return finish(ToolResult.execution_error(record.id, "Unknown tool"))
        try:
            validation = tool.validate_parameters(record)
        except (ValueError, TypeError) as error:
            return finish(ToolResult.invalid_parameters(record.id, str(error)))
        if not validation.is_tool_valid:
            return finish(
                ToolResult.invalid_parameters(
                    record.id, validation.msg_error or "Invalid arguments"
                )
            )
        from ...capabilities.tools.ask_user import AskUserTool

        if isinstance(tool, AskUserTool):
            if record.user_answer is not None:
                return finish(ToolResult.success_result(record.id, {
                    "question": record.input_question,
                    "answer": record.user_answer,
                }))
            if record.is_pending_approval:
                # Options already passed schema validation (each a
                # {label, description} dict) by the time we get here.
                raw_options = record.parameters.get("options")
                display_options = None
                if isinstance(raw_options, list):
                    sanitized: list[str] = []
                    for o in raw_options:
                        label = o.get("label") if isinstance(o, dict) else None
                        if isinstance(label, str) and label.strip():
                            text = label.strip()
                            description = o.get("description")
                            if isinstance(description, str) and description.strip():
                                text = f"{text} — {description.strip()}"
                            sanitized.append(text)
                    display_options = sanitized or None
                record.await_user_input(
                    record.parameters["question"], display_options,
                )
            if not record.is_awaiting_input:
                return finish(ToolResult.execution_error(
                    record.id, "ask_user requires pending input or a stored answer",
                ))
            emit(UserInputRequestEvent(
                source=self.source, tool_call_id=record.id,
                question=record.input_question or "", options=record.input_options,
            ))
            return None
        if record.is_awaiting_input:
            raise ValueError("User input must be resolved before dispatch")
        try:
            approval_mode = ToolApprovalMode(tool.approval_mode)
            from ...capabilities.tools.bash import BashTool

            if isinstance(tool, BashTool):
                permission = tool.permission_for(record.parameters["command"])
                if permission == "deny":
                    return finish(ToolResult.execution_error(record.id, "Command denied by Bash permissions"))
                approval_mode = (
                    ToolApprovalMode.AUTO_APPROVED if permission == "allow"
                    else ToolApprovalMode.ASK_APPROVED
                )
        except ValueError:
            return finish(
                ToolResult.execution_error(record.id, "Unknown approval mode")
            )
        if record.is_pending_approval:
            if approval_mode == ToolApprovalMode.ASK_APPROVED:
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
                return None
            record.auto_approve()
        if not record.is_actionable:
            raise ValueError(f"Cannot dispatch from {record.status}")
        if approval_mode == ToolApprovalMode.ASK_APPROVED and record.was_auto_approved:
            return finish(ToolResult.execution_error(
                record.id, "Current permissions require manual approval; submit a new tool call",
            ))
        if not self.registry.runs_on_host(tool.name) and self.manager is None:
            return finish(ToolResult.execution_error(
                record.id, "A runtime tool requires an EnvironmentManager",
            ))

        # Per-call dependencies must not leak into concurrent calls or resumes.
        deps = dict(context.deps)
        deps["tool_call_id"] = record.id
        deps.pop("tool_reference", None)
        reference = self.registry.reference(tool.name)
        if reference is not None:
            deps["tool_reference"] = reference
        call_context = ToolContext(
            context.run_id, session_id=context.session_id, user_id=context.user_id,
            retry_count=context.retry_count, deps=deps, emit_event=context.emit_event,
        )

        async def invoke():
            if self.registry.runs_on_host(tool.name):
                return await tool.execute(record, call_context, cancellation_token)
            async with self.manager.acquire(context.user_id, context.session_id) as session:
                return await self.manager.executor.run_tool(
                    session, tool, record, call_context, cancellation_token,
                )

        record.start_execution()
        try:
            task = asyncio.create_task(
                invoke()
            )
            if cancellation_token is not None:
                cancellation_token.link_future(task)
            result = await asyncio.wait_for(task, timeout=tool.timeout_seconds)
            if not isinstance(result, ToolResult) or result.tool_call_id != record.id:
                result = ToolResult.execution_error(record.id, "Invalid tool result")
        except asyncio.TimeoutError:
            result = ToolResult.timeout(record.id, tool.timeout_seconds)
        except asyncio.CancelledError:
            finish(ToolResult.cancelled_during_execution(record.id))
            raise
        except Exception as error:
            result = ToolResult.execution_error(record.id, str(error))
        return finish(result)
