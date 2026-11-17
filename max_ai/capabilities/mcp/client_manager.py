"""Lifecycle manager for MCP SDK v2 clients and discovered tools."""

from __future__ import annotations

import asyncio
import time
import typing as t
from dataclasses import dataclass, field

import httpx
import httpx2
from mcp import Client, MCPError
from mcp.types import CallToolResult, Result

from ...base.tools import CoreTool
from ...core.retry import RetryPolicy, retrying
from ...errors.mcp import (
    MCPConnectionLost,
    MCPServerNotFoundError,
    MCPServerRegistrationError,
    MCPServerUnavailableError,
    MCPToolError,
)
from ...types.tools import ToolApprovalMode
from ._model import MCPServerConfig, StdioMCPServerConfig
from .task import CancelTask, GetTask, MCPTask, TaskParams, TasksExtension, TaskState
from .tool import MCPResourceTool, MCPTool
from .transport import create_mcp_client

if t.TYPE_CHECKING:
    from ...types.run_context import RunContext

ProgressFn = t.Callable[[float, float | None, str | None], None]


@dataclass
class _ManagedServer:
    """_ManagedServer represents structured data used by the capability system."""

    config: MCPServerConfig
    worker: asyncio.Task[None] | None = None
    requests: asyncio.Queue[t.Any] = field(default_factory=asyncio.Queue)
    tools: list[CoreTool] = field(default_factory=list)
    connecting: asyncio.Task[None] | None = None  # The connection attempt in progress
    failed_at: float | None = None  # When the last attempt failed (time.monotonic()) and why.
    error: str | None = None


class MCPClientManager:
    """Own MCP clients in dedicated tasks so SDK calls share their lifecycle.

    ``retry`` covers connecting (network errors, within
    ``connect_timeout_seconds``) and sending read-only calls again when the
    connection drops.
    """

    def __init__(
        self,
        retry: RetryPolicy | None = None,
        stdio_commands: t.Sequence[str] | None = None,
        *,
        idle_timeout: float = 360,
        max_timeout: float = 600,
        task_wait: float = 60,
    ) -> None:
        # Seconds a call may go without a progress notice, and in total.
        self.idle_timeout = idle_timeout
        self.max_timeout = max_timeout
        self.task_wait = task_wait  # seconds to wait for a task before leaving it running
        self._servers: dict[str, _ManagedServer] = {}
        self.retry = retry or RetryPolicy()
        # A stdio server is a program run on this machine. None allows any
        # command; a list allows only those (an empty one, none at all).
        self.stdio_commands = None if stdio_commands is None else set(stdio_commands)

    @property
    def server_ids(self) -> list[str]:
        return list(self._servers)

    def add_server(self, config: MCPServerConfig) -> None:
        if not isinstance(config, MCPServerConfig) or type(config) is MCPServerConfig:
            raise TypeError("Use StdioMCPServerConfig or HTTPServerConfig")
        if (
            isinstance(config, StdioMCPServerConfig)
            and self.stdio_commands is not None
            and config.command not in self.stdio_commands
        ):
            raise MCPServerRegistrationError(
                f"MCP server {config.server_id!r} runs {config.command!r} on this machine, "
                "and that command is not allowed here."
            )
        if config.server_id in self._servers:
            raise MCPServerRegistrationError(f"MCP server already registered: {config.server_id}")
        self._servers[config.server_id] = _ManagedServer(config=config)

    async def connect(self, server_id: str) -> None:
        """Open the connection unless it is up; raise if the server is unavailable.

        The attempt (network errors are retried) is bounded by
        ``connect_timeout_seconds`` and shared by concurrent callers. After a
        failure the server is not retried for ``retry_after_seconds``; its
        last known tools stay registered and answer that it is unavailable.
        """
        server = self._get_server(server_id)
        if server.worker is not None and not server.worker.done():
            return
        if server.connecting is None:
            if (
                server.failed_at is not None
                and time.monotonic() - server.failed_at < server.config.retry_after_seconds
            ):
                raise self._unavailable(server)
            attempt = asyncio.create_task(self._open(server))
            server.connecting = attempt

            def finished(task: asyncio.Task[None]) -> None:
                if server.connecting is task:
                    server.connecting = None
                if not task.cancelled():
                    task.exception()  # retrieved here even if every waiter left

            attempt.add_done_callback(finished)
        await asyncio.shield(server.connecting)

    async def _open(self, server: _ManagedServer) -> None:
        server.worker = None  # a dead connection is replaced

        async def attempt() -> tuple[asyncio.Task[None], list[CoreTool]]:
            ready = asyncio.get_running_loop().create_future()
            worker = asyncio.create_task(self._serve(server, ready))
            try:
                return worker, await ready
            except BaseException:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                raise

        timeout = server.config.connect_timeout_seconds
        try:
            worker, tools = await asyncio.wait_for(
                retrying(attempt, policy=self.retry, transient=self._unreachable),
                timeout,
            )
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise  # cancelled: not the server's fault
            server.failed_at = time.monotonic()
            server.error = (
                f"no answer within {timeout:g}s"
                if isinstance(exc, TimeoutError)
                else str(exc) or type(exc).__name__
            )
            raise self._unavailable(server) from exc
        server.worker = worker
        server.tools = tools
        server.failed_at = server.error = None

    @staticmethod
    def _unavailable(server: _ManagedServer) -> MCPServerUnavailableError:
        return MCPServerUnavailableError(
            f"MCP server {server.config.server_id!r} is unavailable ({server.error}); "
            f"it is retried after {server.config.retry_after_seconds:g}s. "
            "Try again later or continue without it."
        )

    async def _serve(self, server: _ManagedServer, ready: asyncio.Future) -> None:
        try:
            extensions = [TasksExtension(self.task_wait)]
            async with create_mcp_client(server.config, extensions) as client:
                ready.set_result(await self._discover_server_tools(server.config, client))
                # This task owns the connection; each call runs in its own
                # subtask so parallel tool calls don't wait for each other.
                calls: set[asyncio.Task[None]] = set()

                async def answer(method: str, args: tuple, future: asyncio.Future) -> None:
                    """Run one MCP request and resolve its waiting future.

                    Parameters
                    ----------
                    method : str
                        Client method to call.
                    args : tuple
                        Positional arguments for the client method.
                    future : asyncio.Future
                        Future that receives the result or raised exception.
                    """
                    try:
                        # send_request (tasks) is the session's, the rest the client's.
                        target = client.session if method == "send_request" else client
                        result = await getattr(target, method)(*args)
                        if not future.done():
                            future.set_result(result)
                    except asyncio.CancelledError:
                        if asyncio.current_task().cancelling():
                            # The connection is closing: the caller wasn't cancelled.
                            if not future.done():
                                future.set_exception(MCPConnectionLost("MCP connection closed"))
                            raise
                        # Cancelled inside the client (the transport failed), not by us.
                        if not future.done():
                            future.set_exception(
                                MCPConnectionLost(
                                    "The MCP server connection interrupted the request."
                                )
                            )
                    except BaseException as exc:
                        if not future.done():
                            future.set_exception(exc)

                try:
                    while True:
                        request = await server.requests.get()
                        if request is None:
                            break
                        method, args, future = request
                        if future.done():
                            continue
                        call = asyncio.create_task(answer(method, args, future))
                        calls.add(call)
                        call.add_done_callback(calls.discard)
                finally:
                    for call in list(calls):
                        call.cancel()
                    await asyncio.gather(*calls, return_exceptions=True)
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise
        finally:
            while not server.requests.empty():
                request = server.requests.get_nowait()
                if request is not None:
                    future = request[2]
                    if not future.done():
                        future.set_exception(MCPConnectionLost("MCP connection closed"))

    async def connect_all(self) -> dict[str, MCPServerUnavailableError]:
        """Connect every server in parallel; one failing never touches the others.

        Returns the unavailable servers. Raises if one of them is ``required``.
        """
        server_ids = self.server_ids
        results = await asyncio.gather(
            *(self.connect(server_id) for server_id in server_ids),
            return_exceptions=True,
        )
        failures: dict[str, MCPServerUnavailableError] = {}
        for server_id, result in zip(server_ids, results):
            if isinstance(result, MCPServerUnavailableError):
                failures[server_id] = result
            elif isinstance(result, BaseException):
                raise result
        for server_id, error in failures.items():
            if self._servers[server_id].config.required:
                raise error
        return failures

    def get_tools(self) -> list[CoreTool]:
        return [tool for server in self._servers.values() for tool in server.tools]

    async def disconnect(self, server_id: str) -> None:
        server = self._get_server(server_id)
        if server.connecting is not None:
            server.connecting.cancel()
            await asyncio.gather(server.connecting, return_exceptions=True)
        server.failed_at = server.error = None
        worker = server.worker
        server.worker = None
        server.tools = []
        if worker is not None:
            await server.requests.put(None)
            # A connection that already died keeps its error: closing must not raise it.
            await asyncio.gather(worker, return_exceptions=True)

    async def disconnect_all(self) -> None:
        for server_id in reversed(self.server_ids):
            await self.disconnect(server_id)

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, t.Any],
        *,
        read_only: bool = False,
        on_progress: ProgressFn | None = None,
    ) -> t.Any:
        """Call a tool. It may run ``idle_timeout`` seconds without a progress
        notice, each notice starting that count again, and ``max_timeout`` in all."""
        alive = asyncio.Event()

        async def progress(progress: float, total: float | None, message: str | None) -> None:
            alive.set()
            if on_progress is not None:
                on_progress(progress, total, message)

        call = asyncio.ensure_future(
            self._request(
                server_id,
                "call_tool",
                tool_name,
                arguments,
                self.max_timeout,
                progress,
                again=read_only,
            )
        )
        try:
            async with asyncio.timeout(self.max_timeout):
                while not call.done():
                    alive.clear()
                    waiter = asyncio.ensure_future(alive.wait())
                    done, _ = await asyncio.wait(
                        {call, waiter},
                        timeout=self.idle_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    waiter.cancel()
                    if not done:
                        raise TimeoutError
            return call.result()
        except TimeoutError as exc:
            raise MCPToolError(
                f"MCP tool {tool_name!r} of {server_id!r} stopped answering."
            ) from exc
        finally:
            call.cancel()

    async def collect_tasks(
        self, ctx: RunContext
    ) -> list[tuple[MCPTask, CallToolResult | None]]:
        """Ask the servers about the tasks of ``ctx``; return and forget the ones
        that ended. None as the result: the server no longer has it."""
        ended: list[tuple[MCPTask, CallToolResult | None]] = []
        for task_id, kept in list(ctx.mcp_tasks.items()):
            task = MCPTask.model_validate(kept)
            result: CallToolResult | None = None
            if task.expires_at is None or time.time() < task.expires_at:
                try:
                    state = await asyncio.wait_for(self._request(
                        task.server_id, "send_request",
                        GetTask(params=TaskParams(task_id=task_id)), TaskState, again=True,
                    ), self.idle_timeout)
                except MCPError:
                    state = None  # the server does not know the task any more
                except (MCPToolError, MCPServerUnavailableError, TimeoutError):
                    continue  # the server can't be asked now: try on the next round
                if state is not None and state.status == "working":
                    continue
                result = TasksExtension.finished(state) if state is not None else None
            del ctx.mcp_tasks[task_id]
            ended.append((task, result))
        return ended

    async def cancel_task(self, ctx: RunContext, task_id: str) -> None:
        """Ask the server to stop a task of ``ctx`` and forget it."""
        kept = ctx.mcp_tasks.pop(task_id, None)
        if kept is None:
            return
        task = MCPTask.model_validate(kept)
        try:
            await asyncio.wait_for(self._request(
                task.server_id, "send_request",
                CancelTask(params=TaskParams(task_id=task_id)), Result,
            ), self.idle_timeout)
        except (MCPError, MCPToolError, MCPServerUnavailableError, TimeoutError):
            pass  # already gone, or unreachable: nothing left to stop

    async def read_resource(self, server_id: str, uri: str) -> t.Any:
        timeout = self.idle_timeout
        try:
            return await asyncio.wait_for(
                self._request(server_id, "read_resource", uri, again=True),
                timeout,
            )
        except TimeoutError as exc:
            raise MCPToolError(
                f"MCP server {server_id!r} did not return {uri!r} within {timeout:g}s."
            ) from exc

    async def _request(
        self,
        server_id: str,
        method: str,
        *args: t.Any,
        again: bool = False,
    ) -> t.Any:
        server = self._get_server(server_id)

        async def send() -> t.Any:
            if server.worker is None or server.worker.done():
                await self.connect(server_id)  # reconnects a dropped connection
            future = asyncio.get_running_loop().create_future()
            await server.requests.put((method, args, future))
            return await future

        # Lost mid-request, it may have run: only read-only calls go again.
        policy = self.retry if again else RetryPolicy(attempts=1)
        return await retrying(
            send,
            policy=policy,
            transient=lambda error: isinstance(error, MCPConnectionLost),
        )

    @staticmethod
    def _unreachable(error: BaseException) -> bool:
        """The server couldn't be reached (network, process, timeout): worth
        trying again. anyio wraps these in exception groups."""
        if isinstance(error, BaseExceptionGroup):
            return any(MCPClientManager._unreachable(inner) for inner in error.exceptions)
        # The HTTP transports use httpx2; its errors are not httpx's.
        return isinstance(
            error, (OSError, TimeoutError, httpx.TransportError, httpx2.TransportError)
        )

    def _get_server(self, server_id: str) -> _ManagedServer:
        try:
            return self._servers[server_id]
        except KeyError as exc:
            raise MCPServerNotFoundError(f"Unknown MCP server: {server_id}") from exc

    async def _discover_server_tools(
        self,
        config: MCPServerConfig,
        client: Client,
    ) -> list[CoreTool]:
        discovered: list[CoreTool] = []
        for tool_def in await self._list_all_tools(client):
            discovered.append(
                MCPTool(
                    mcp_tool_name=tool_def.name,
                    mcp_tool_description=tool_def.description or tool_def.name,
                    mcp_tool_schema=tool_def.input_schema,
                    client_manager=self,
                    server_id=config.server_id,
                    approval_mode=self._approval_mode_for_tool(config, tool_def),
                    read_only=config.trusted and self._is_read_only(tool_def),
                )
            )
        resources, templates = await self._list_resources(client)
        if resources or templates:
            discovered.append(
                MCPResourceTool(
                    client_manager=self,
                    server_id=config.server_id,
                    available_resources=resources,
                    resource_templates=templates,
                    approval_mode=config.effective_resource_approval_mode,
                )
            )
        return discovered

    @staticmethod
    def _is_read_only(tool_def: t.Any) -> bool:
        """The server says the tool only reads (MCP ``readOnlyHint``)."""
        annotations = getattr(tool_def, "annotations", None)
        return bool(
            annotations is not None
            and annotations.read_only_hint is True
            and annotations.destructive_hint is not True
        )

    @staticmethod
    def _approval_mode_for_tool(config: MCPServerConfig, tool_def: t.Any) -> ToolApprovalMode:
        explicit = config.tool_approval_modes.get(tool_def.name)
        if explicit is not None:
            return explicit
        annotations = getattr(tool_def, "annotations", None)
        # What a server says about itself counts only if the config trusts it.
        if annotations is None or not config.trusted:
            return config.approval_mode
        if (
            annotations.read_only_hint is True
            and annotations.destructive_hint is not True
            and annotations.open_world_hint is not True
        ):
            return ToolApprovalMode.AUTO_APPROVED
        if (
            annotations.destructive_hint is True
            or annotations.open_world_hint is True
            or annotations.read_only_hint is False
        ):
            return ToolApprovalMode.ASK_APPROVED
        return config.approval_mode

    @staticmethod
    async def _list_all_tools(client: Client) -> list[t.Any]:
        tools: list[t.Any] = []
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tools

    @staticmethod
    async def _list_resources(client: Client) -> tuple[list[t.Any], list[t.Any]]:
        resources: list[t.Any] = []
        templates: list[t.Any] = []
        try:
            cursor: str | None = None
            while True:
                page = await client.list_resources(cursor=cursor)
                resources.extend(page.resources)
                cursor = page.next_cursor
                if cursor is None:
                    break
        except MCPError:
            pass
        try:
            cursor = None
            while True:
                page = await client.list_resource_templates(cursor=cursor)
                templates.extend(page.resource_templates)
                cursor = page.next_cursor
                if cursor is None:
                    break
        except MCPError:
            pass
        return resources, templates
