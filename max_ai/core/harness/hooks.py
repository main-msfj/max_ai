"""The hooks of our harness: what it does around the model and around each tool."""

import logging
import os
import typing as t
from collections.abc import Sequence

from ...base.loop import LoopHook, LoopState
from ...base.middleware import CoreMiddleware, MiddlewareContext, ToolRequest
from ...base.sandbox import SandboxCore
from ...base.tools import CoreTool, ToolContext
from ...capabilities.mcp import MCPClientManager
from ...capabilities.memory import Memory
from ...types.run_context import RunContext
from ...types.tool_call import ToolCallRecord, ToolResult
from ..tool.hook import ToolHook
from .permissions import Permissions

logger = logging.getLogger(__name__)


class HarnessLoopHook(LoopHook):
    """Around the model: the memory index stays fresh and the MCP tasks
    that ended are told."""

    TASK_ENDED: t.ClassVar[str] = "The task you left running with {tool} ended. Its result:\n{result}"
    TASK_LOST: t.ClassVar[str] = (
        "The task you left running with {tool} is no longer on its server; its result was lost."
    )

    def __init__(
        self,
        *,
        memory: Memory | None = None,
        sandbox: SandboxCore | None = None,  # already bound to the user
        mcp: MCPClientManager | None = None,
        warn_before: int = 3,
        max_repeats: int = 3,
    ) -> None:
        super().__init__(warn_before=warn_before, max_repeats=max_repeats)
        self.memory = memory
        self.sandbox = sandbox
        self.mcp = mcp

    async def refresh_memory(self, ctx: RunContext) -> None:
        """Read the memory index again; also called when a session opens."""
        if self.memory is not None and self.sandbox is not None:
            ctx.memory_index = await self.memory.index(self.sandbox)

    async def after_compact(self, ctx: RunContext, state: LoopState) -> None:
        # The model forgot what it saved: show it the memory as it is now.
        await self.refresh_memory(ctx)

    async def notes(self, ctx: RunContext, state: LoopState) -> list[str]:
        # 1. The usual notes, then the tasks that ended on the MCP servers.
        notes = await super().notes(ctx, state)
        if self.mcp is None or not ctx.mcp_tasks:
            return notes
        try:
            ended = await self.mcp.collect_tasks(ctx)
        except Exception:
            logger.exception("The MCP tasks could not be read")
            return notes
        # 2. One note per task, with its result or saying it was lost.
        for task, result in ended:
            if result is None:
                notes.append(self.TASK_LOST.format(tool=task.tool_name))
                continue
            texts = [str(getattr(part, "text", "")) for part in result.content]
            text = "\n".join(text for text in texts if text) or "(no text)"
            notes.append(self.TASK_ENDED.format(tool=task.tool_name, result=text))
        return notes


class HarnessToolHook(ToolHook):
    """Around each tool: a read-only memory is not written, and the
    middlewares (tracing among them) see the call and its result."""

    WRITERS: t.ClassVar[frozenset[str]] = frozenset({"write_file", "edit_file"})
    MEMORY_READ_ONLY: t.ClassVar[str] = (
        "The memory is read-only here: {path} was not written. Go on without saving it."
    )

    def __init__(
        self,
        *,
        middlewares: Sequence[CoreMiddleware] = (),
        memory: Memory | None = None,
        agent: str = "",
        permissions: Permissions | None = None,  # the user's rules, read from setting.json
    ) -> None:
        self.permissions = permissions or Permissions()
        self.middlewares = list(middlewares)
        self.memory = memory
        self.agent = agent

    async def before(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext
    ) -> ToolResult | None:
        # 1. A read-only memory is never written by the file tools.
        path = self.memory_path(record, tool_ctx)
        if path is not None and self.memory is not None and self.memory.read_only:
            return ToolResult.tool_failure(record.id, self.MEMORY_READ_ONLY.format(path=path))
        # 2. A middleware may answer in the tool's place; one that fails is skipped.
        context, request = self.context(tool_ctx), ToolRequest(record=record)
        for middleware in self.middlewares:
            try:
                answer = await middleware.on_tool_request(context, request)
            except Exception:
                logger.exception("Middleware %s failed before a tool", type(middleware).__name__)
                continue
            if answer is not None:
                return answer
        return await super().before(tool, record, tool_ctx)

    async def permission(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext
    ) -> bool | None:
        # The rules decide first; without one, the tool's own approval mode.
        decision = self.permissions.decide(record.tool_name, record.parameters)
        if isinstance(decision, str):
            return await super().permission(tool, record, tool_ctx)
        return decision

    async def after(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext, result: ToolResult
    ) -> ToolResult:
        result = await self.tell(record, tool_ctx, result)
        return await super().after(tool, record, tool_ctx, result)

    async def failed(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext, result: ToolResult
    ) -> ToolResult:
        result = await self.tell(record, tool_ctx, result)
        return await super().failed(tool, record, tool_ctx, result)

    async def tell(
        self, record: ToolCallRecord, tool_ctx: ToolContext, result: ToolResult
    ) -> ToolResult:
        """Each middleware sees the result, last one first; one that fails is skipped."""
        context, request = self.context(tool_ctx), ToolRequest(record=record)
        for middleware in reversed(self.middlewares):
            try:
                result = await middleware.on_tool_response(context, request, result)
            except Exception:
                logger.exception("Middleware %s failed after a tool", type(middleware).__name__)
        return result

    def context(self, tool_ctx: ToolContext) -> MiddlewareContext:
        made = MiddlewareContext(ctx=tool_ctx.run_context, agent=self.agent)
        if tool_ctx.emit_event is not None:
            made.emit = tool_ctx.emit_event
        return made

    @staticmethod
    def memory_path(record: ToolCallRecord, tool_ctx: ToolContext) -> str | None:
        """Where the call writes, when that is inside the memory folder."""
        sandbox = tool_ctx.sandbox
        path = record.parameters.get("path")
        if record.tool_name not in HarnessToolHook.WRITERS or sandbox is None or not path:
            return None
        # A relative path starts where the model is.
        start = tool_ctx.run_context.cwd or str(sandbox.layout.workspace)
        full = os.path.normpath(os.path.join(start, str(path)))
        memory = str(sandbox.layout.memory)
        return full if full == memory or full.startswith(memory + os.sep) else None
