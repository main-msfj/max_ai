"""
Routing execution strategy.

Splits tool execution across two backends by tool *type*:

  - ``CoreRuntimeTool`` instances (e.g. ``BashTool``) — and therefore the
    skill scripts they run — go to the ``sandbox`` executor (Docker).
  - Every other ``CoreTool`` (FunctionAsTool / ``@tool`` wrappers,
    WorkspaceTool, knowledge tools, …) runs in-process via the ``local``
    executor.

Rationale: ordinary tools are developer-authored code, trusted at the
same level as the rest of the application — paying the cost of a
container per call buys nothing. The variable, model-driven surface is
the *skill scripts* the LLM chooses to run through ``bash``; only that
needs isolation. Aligning the trust boundary with the executor split
keeps the common path fast and the risky path contained.

The decision is structural, not name-based: ``isinstance(tool,
CoreRuntimeTool)`` is the whole routing rule. A tool opts into the
sandbox by subclassing ``CoreRuntimeTool``.

Lifecycle:
  - ``bind_to_workspace`` fans out to both backends. Binding Docker is
    cheap (it only records the workspace root and ensures the host
    layout exists); the expensive step (image build) stays lazy.
  - ``connect`` connects the local backend eagerly. The sandbox backend
    is connected lazily on its first ``run`` (via its own
    ``_ensure_connected``), so an agent that never invokes a runtime
    tool never builds or boots a container.
  - ``disconnect`` fans out to both so container cleanup always runs.

Contract: ``run`` always returns a ``ToolResult`` — never raises. A
sandbox-bound tool with no configured sandbox backend is surfaced as a
failure result, not an exception.
"""

from __future__ import annotations


from pathlib import Path

from ..base.executor import CoreExecutor
from ..termination import CancellationToken
from ..types.tool_call import ToolCallRecord, ToolResult
from ..base.tools import CoreRuntimeTool, CoreTool, ToolContext


class RoutingExecutor(CoreExecutor):
    """Dispatch each tool to the local or sandbox backend by type."""

    def __init__(
        self,
        local: CoreExecutor,
        sandbox: CoreExecutor | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            local: Executor for in-process tools. Required.
            sandbox: Executor for ``CoreRuntimeTool`` instances. Optional —
                when ``None``, an agent with no runtime tools runs fully
                local. If a runtime tool is invoked without a sandbox
                backend configured, the call fails with a clear result.
        """
        super().__init__(default_timeout=local.default_timeout)
        self.local = local
        self.sandbox = sandbox

    # -------- ROUTING -----------------------------------------------------------
    @staticmethod
    def _needs_sandbox(tool: CoreTool) -> bool:
        """A tool needs the sandbox iff it is a ``CoreRuntimeTool``."""
        return isinstance(tool, CoreRuntimeTool)

    def _select(self, tool: CoreTool) -> CoreExecutor | None:
        """Return the backend that should run ``tool`` (may be None)."""
        if self._needs_sandbox(tool):
            return self.sandbox
        return self.local

    # -------- LIFECYCLE -----------------------------------------------------------
    async def bind_to_workspace(self, workspace_registry_root: str | Path) -> None:
        """Bind both backends to the shared workspace root."""
        await self.local.bind_to_workspace(workspace_registry_root)
        if self.sandbox is not None:
            await self.sandbox.bind_to_workspace(workspace_registry_root)

    async def connect(self) -> None:
        """Connect the local backend eagerly; leave the sandbox lazy.

        The sandbox backend connects on its first ``run`` through its own
        ``_ensure_connected``, so no container work happens unless a
        runtime tool is actually invoked.
        """
        await self.local.connect()

    async def disconnect(self) -> None:
        """Disconnect both backends so container cleanup always runs."""
        try:
            await self.local.disconnect()
        finally:
            if self.sandbox is not None:
                await self.sandbox.disconnect()

    # -------- EXECUTION -----------------------------------------------------------
    async def run(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        """Dispatch one tool call to the appropriate backend.

        Never raises: a runtime tool with no sandbox backend configured is
        returned as a failed ``ToolResult`` so the orchestrator can keep
        the ``ToolCallRecord`` lifecycle intact.
        """
        backend = self._select(tool)
        if backend is None:
            return ToolResult.execution_error(
                record.id,
                (
                    f"Tool {tool.name!r} requires a sandboxed runtime, but no "
                    "sandbox executor is configured. Provide a sandbox-capable "
                    "executor (e.g. DockerExecutor) to run runtime tools/skills."
                ),
            )
        return await backend.run(
            tool,
            record,
            tool_context,
            cancellation_token=cancellation_token,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"local={type(self.local).__name__}, "
            f"sandbox={type(self.sandbox).__name__ if self.sandbox else None})"
        )