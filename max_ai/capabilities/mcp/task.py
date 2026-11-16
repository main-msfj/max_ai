"""MCP tasks: a tool call the server runs in the background (SEP-2663)."""

from __future__ import annotations

import asyncio
import time
import typing as t

import mcp_types
from mcp.client.extension import ClaimContext, ClientExtension, ResultClaim
from mcp_types import CallToolResult, RequestParams, Result
from pydantic import BaseModel, ConfigDict, Field

from ...errors.mcp import MCPToolError


class MCPTask(BaseModel):
    """A task still running on a server; kept in ``RunContext.mcp_tasks``."""

    task_id: str
    server_id: str
    tool_name: str
    tool_call_id: str = ""
    expires_at: float | None = None  # time.time() when the server drops the result


class MCPTaskRunning(MCPToolError):
    """The call became a task that did not end within the wait."""

    def __init__(self, task_id: str, ttl_ms: int | None) -> None:
        super().__init__(f"Task {task_id} is still running.")
        self.task_id = task_id
        self.expires_at = None if ttl_ms is None else time.time() + ttl_ms / 1000


class TaskState(Result):
    """What the server says about a task; ``result`` only when completed."""

    model_config = ConfigDict(populate_by_name=True)
    task_id: str = Field(alias="taskId")
    status: t.Literal["working", "input_required", "completed", "failed", "cancelled"]
    ttl_ms: int | None = Field(default=None, alias="ttlMs")
    poll_interval_ms: int | None = Field(default=None, alias="pollIntervalMs")
    status_message: str | None = Field(default=None, alias="statusMessage")
    result: dict[str, t.Any] | None = None
    error: dict[str, t.Any] | None = None


class TaskCreated(TaskState):
    result_type: t.Literal["task"] = Field(alias="resultType")


class TaskParams(RequestParams):
    model_config = ConfigDict(populate_by_name=True)
    task_id: str = Field(serialization_alias="taskId")


class GetTask(mcp_types.Request[TaskParams, t.Literal["tasks/get"]]):
    method: t.Literal["tasks/get"] = "tasks/get"
    params: TaskParams
    name_param = "taskId"


class CancelTask(mcp_types.Request[TaskParams, t.Literal["tasks/cancel"]]):
    method: t.Literal["tasks/cancel"] = "tasks/cancel"
    params: TaskParams
    name_param = "taskId"


class TasksExtension(ClientExtension):
    """Tells a server this client runs tasks, and waits a while for each one."""

    identifier = "io.modelcontextprotocol/tasks"

    def __init__(self, wait: float = 60) -> None:
        self.wait = wait  # seconds to wait before leaving the task running

    def claims(self) -> t.Sequence[ResultClaim[t.Any]]:
        return (ResultClaim(result_type="task", model=TaskCreated, resolve=self._resolve),)

    async def _resolve(self, created: TaskCreated, ctx: ClaimContext) -> CallToolResult:
        deadline, delay = time.monotonic() + self.wait, 0.5
        ceiling = (created.poll_interval_ms or 5000) / 1000
        while True:
            state = await ctx.session.send_request(
                GetTask(params=TaskParams(task_id=created.task_id)), TaskState
            )
            if state.status != "working":
                return self.finished(state)
            left = deadline - time.monotonic()
            if left <= 0:
                raise MCPTaskRunning(created.task_id, created.ttl_ms)
            await asyncio.sleep(min(delay, left))
            delay = min(delay * 2, ceiling)

    @staticmethod
    def finished(state: TaskState) -> CallToolResult:
        """A task that ended, as the result of an ordinary call."""
        if state.status == "completed":
            return CallToolResult.model_validate(state.result or {})
        if state.status == "input_required":
            reason = "it asks for input this client cannot give"
        else:
            reason = (state.error or {}).get("message") or state.status_message or state.status
        return CallToolResult(
            content=[mcp_types.TextContent(type="text", text=f"The task did not finish: {reason}")],
            is_error=True,
        )
