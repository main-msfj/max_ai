"""
Tool execution strategy.

A ``CoreExecutor`` decides *where* and *how* a tool runs — locally in
the agent's process, in a Docker container, against an MCP server,
or anywhere else. The orchestration (resolve / approve / validate /
record lifecycle) stays in ``ToolExecutor``; this layer only owns the
final ``tool.execute(...)`` step and its safety wrappers (timeout,
cancellation, exception capture).

Contract:

- ``run`` always returns a ``ToolResult`` — never raises. Every
  exception, timeout, or cancellation is captured and surfaced as a
  ``ToolResult`` with ``success=False`` and an appropriate
  ``failure_reason``. The orchestrator relies on this guarantee to
  preserve the ``ToolCallRecord`` lifecycle.
- ``default_timeout`` is the per-tool fallback when the tool itself
  doesn't declare ``timeout_seconds``. Each strategy ships its own
  reasonable default — local executors can be aggressive (~300s),
  Docker-style executors usually need more headroom for boot.
"""

from __future__ import annotations

import typing as t
from pathlib import Path
from abc import ABC, abstractmethod

from pydantic import BaseModel
from .component import CoreLifecycleComponent
from ..termination import CancellationToken
from ..base.tools import CoreTool, ToolContext
from ..types.tool_call import ToolCallRecord, ToolResult

if t.TYPE_CHECKING:
    from .workspace import WorkSpaceRegistry


class CoreExecutor(CoreLifecycleComponent[BaseModel], ABC):
    """Abstract execution strategy.

    Subclasses implement ``run`` for a specific runtime: in-process
    asyncio, Docker container, MCP server, remote service, etc.
    """

    def __init__(self, default_timeout: int = 300) -> None:
        """Initialize the executor.

        Args:
            default_timeout: Fallback timeout in seconds when the tool
                does not declare its own ``timeout_seconds``. Each
                strategy can ship a different default — local can be
                tight, container-based strategies usually need more.
        """
        super().__init__()
        self.default_timeout = default_timeout

    @abstractmethod
    async def bind_to_workspace(self, workspace_registry_root: str | Path) -> None:
        """Bind Workspace to Executor"""
        ...


    @abstractmethod
    async def run(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        """Execute one tool call.

        Args:
            tool: The resolved ``CoreTool`` instance.
            record: The lifecycle record for this call. Already in
                ``EXECUTING`` state when this is invoked.
            tool_context: Per-call context the tool may need
                (run/session ids, etc.). The orchestrator builds this
                so the executor doesn't need access to the whole
                ``RunContext``.
            cancellation_token: External cancellation signal. Strategies
                should propagate it to whatever async primitive they
                use so a cancelled run doesn't leak resources.

        Returns:
            A ``ToolResult``. Always — even on timeout, cancellation,
            or unhandled exceptions, the strategy must convert them
            to a failure result with the right ``failure_reason``.
        """
        ...
