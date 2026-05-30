"""
Local in-process tool executor.

Runs tools in the same Python process as the agent, on the asyncio
event loop. Coroutine tools are awaited directly; sync tools are
dispatched to the default thread pool. Suitable for tools that are
trusted, side-effect-light, and don't need isolation.

Trade-offs:
- Fast: zero overhead beyond a coroutine schedule.
- No isolation: a misbehaving tool can corrupt the agent's process.
- No resource limits beyond ``timeout_seconds``.

Use ``DockerExecutor`` or similar when running untrusted code or
when you need filesystem / network isolation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ..base.executor import CoreExecutor
from ..base.tools import CoreTool, ToolContext
from ..termination import CancellationToken
from ..types.tool_call import ToolCallRecord, ToolResult


class LocalExecutorConfig(BaseModel):
    default_timeout: int = 300

class LocalExecutor(CoreExecutor):
    component_schema = LocalExecutorConfig
    component_type = "executor"

    """In-process executor — the default.

    Wraps ``tool.execute(...)`` with timeout, cancellation linking,
    and exception capture. Every failure mode is mapped to a
    ``ToolResult`` so the caller never has to handle exceptions.
    """

    def _to_config(self) -> LocalExecutorConfig:
        return LocalExecutorConfig(default_timeout=self.default_timeout)

    @classmethod
    def _from_config(cls, config: LocalExecutorConfig) -> "LocalExecutor":
        return cls(default_timeout=config.default_timeout)

    async def bind_to_workspace(self, workspace_registry_root: str | Path) -> None:
        """Bind workspace root for API compatibility with sandbox executors."""
        self.workspace_root = Path(workspace_registry_root).expanduser().resolve()

    async def run(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        await self._ensure_connected()
        timeout = tool.timeout_seconds or self.default_timeout

        try:
            task = asyncio.create_task(
                tool.execute(record, tool_context, cancellation_token)
            )
            if cancellation_token is not None:
                cancellation_token.link_future(task)
            return await asyncio.wait_for(task, timeout=timeout)

        except asyncio.TimeoutError:
            return ToolResult.timeout(record.id, timeout_seconds=timeout)

        except asyncio.CancelledError:
            return ToolResult.cancelled_during_execution(record.id)

        except ValidationError as ve:
            return ToolResult.invalid_parameters(record.id, str(ve))

        except Exception as e:
            return ToolResult.execution_error(record.id, str(e))
