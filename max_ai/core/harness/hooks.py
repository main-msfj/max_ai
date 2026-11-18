"""The hooks of our harness: what it does around the model and around each tool."""

import logging
import os
import typing as t

from ...base.loop import LoopHook, LoopState
from ...base.sandbox import SandboxCore
from ...base.tools import CoreTool, ToolContext
from ...capabilities.mcp import MCPClientManager
from ...capabilities.memory import Memory
from ...capabilities.tools.memory import SaveMemoryTool
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
    """Around each tool: a read-only memory is not written, and the user's
    rules decide the permission."""

    WRITERS: t.ClassVar[frozenset[str]] = frozenset({"write_file", "edit_file"})
    SAVER: t.ClassVar[str] = SaveMemoryTool.TOOL_NAME
    MEMORY_READ_ONLY: t.ClassVar[str] = (
        "The memory is read-only here: {path} was not written. Go on without saving it."
    )

    def __init__(
        self,
        *,
        memory: Memory | None = None,
        agent: str = "",
        permissions: Permissions | None = None,  # the user's rules, read from setting.json
    ) -> None:
        self.permissions = permissions or Permissions()
        self.memory = memory
        self.agent = agent

    async def before(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext
    ) -> ToolResult | None:
        # 1. A read-only memory is never written: not by the file tools, not by its own.
        path = self.memory_path(record, tool_ctx)
        if record.tool_name == self.SAVER:
            path = f"{record.parameters.get('name')}.md"
        if path is not None and self.memory is not None and self.memory.read_only:
            return ToolResult.tool_failure(record.id, self.MEMORY_READ_ONLY.format(path=path))
        return await super().before(tool, record, tool_ctx)

    async def permission(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext
    ) -> bool | None:
        # The rules decide first; without one, the tool's own approval mode.
        decision = self.permissions.decide(record.tool_name, record.parameters)
        if isinstance(decision, str):
            return await super().permission(tool, record, tool_ctx)
        return decision

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
