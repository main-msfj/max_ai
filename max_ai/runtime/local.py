"""Native, shared-filesystem runtime executor."""

from __future__ import annotations

import asyncio
import math
import os
from uuid import uuid4

from ..base.runtime_executor import ExecutionSession, Executor
from ..base.tools import ToolContext
from .process import run_process


class LocalExecutor(Executor):
    def __init__(self, *, max_output_bytes: int = 1 << 20):
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.max_output_bytes = max_output_bytes
        self._sessions: dict[str, ExecutionSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closed: dict[str, ExecutionSession] = {}

    async def connect(self, workspace, user_id: str, conversation_id: str) -> ExecutionSession:
        directory = workspace.materialize(user_id, conversation_id)
        session = ExecutionSession(uuid4().hex, user_id, conversation_id, workspace,
                                   str(directory.root), directory)
        self._sessions[session.id] = session
        self._locks[session.id] = asyncio.Lock()
        return session

    def _check(self, session):
        if self._sessions.get(session.id) is not session:
            raise ValueError("session does not belong to this executor")

    async def execute_argv(self, session, argv, *, stdin=None, timeout=60, cancellation_token=None):
        self._check(session)
        async with self._locks[session.id]:
            env = os.environ.copy()
            env["WORKSPACE"] = session.workspace_path
            return await run_process(argv, cwd=session.workspace_path, env=env, stdin=stdin,
                                     timeout=timeout, max_output_bytes=self.max_output_bytes,
                                     cancellation_token=cancellation_token)

    async def execute(self, session, command, *, timeout=60, cancellation_token=None):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command cannot be empty")
        # Clear shell startup variables (notably BASH_ENV) inherited from the host.
        return await self.execute_argv(session, [
            "/usr/bin/env", "-i", "PATH=/usr/bin:/bin",
            f"WORKSPACE={session.workspace_path}",
            "/bin/bash", "--noprofile", "--norc", "-c", command,
        ],
                                       timeout=timeout, cancellation_token=cancellation_token)

    async def run_tool(self, session, tool, record, context, cancellation_token=None):
        self._check(session)
        from .binding import SessionEnvironment
        deps = dict(context.deps)
        directory = session.handle
        deps.update(runtime_root=session.workspace_path,
                    conversation_dir=str(directory.conversation_dir),
                    skills_dir=str(directory.skill_dir),
                    filesystem_root=str(session.workspace.base_root),
                    workspace_filesystem=session.workspace.get_filesystem())
        ctx = ToolContext(context.run_id, session_id=session.conversation_id, user_id=session.user_id,
                          retry_count=context.retry_count, deps=deps, emit_event=context.emit_event,
                          environment=SessionEnvironment(self, session))
        task = asyncio.create_task(tool.execute(record, ctx, cancellation_token))
        if cancellation_token is not None:
            cancellation_token.link_future(task)
        timeout = getattr(tool, "timeout_seconds", None)
        if timeout is None or not math.isfinite(timeout) or timeout <= 0:
            timeout = 60
        try:
            return await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            from ..types.tool_call import ToolResult
            return ToolResult.timeout(record.id, timeout_seconds=timeout)

    async def sync(self, session, direction):
        self._check(session)

    async def disconnect(self, session):
        self._check(session)

    async def clean(self, session):
        if self._sessions.get(session.id) is not session:
            if self._closed.get(session.id) is session:
                return
            raise ValueError("session does not belong to this executor")
        self._sessions.pop(session.id, None)
        self._locks.pop(session.id, None)
        self._closed[session.id] = session

    async def rebuild(self, session):
        self._check(session)
        workspace = session.workspace
        user_id = session.user_id
        conversation_id = session.conversation_id
        await self.clean(session)
        return await self.connect(workspace, user_id, conversation_id)
