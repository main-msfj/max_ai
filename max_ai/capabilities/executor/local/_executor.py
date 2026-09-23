"""Native, shared-filesystem executor — no sandbox isolation, no containers."""

from __future__ import annotations

import asyncio
import math
import os

from ....base.executor import ExecutionSession, ExecutorBase
from ....base.tools import ToolContext
from ....core.executor.process import run_process
from ....ids import short_id
from ._model import LocalExecutorConfig


class LocalExecutor(ExecutorBase):
    component_provider_override = "max_ai.capabilities.executor.local.LocalExecutor"
    component_schema = LocalExecutorConfig

    def __init__(self, *, max_output_bytes: int = 1 << 20) -> None:
        super().__init__()
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.max_output_bytes = max_output_bytes
        self._sessions: dict[str, ExecutionSession] = {}
        self._closed: dict[str, ExecutionSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _to_config(self) -> LocalExecutorConfig:
        return LocalExecutorConfig(max_output_bytes=self.max_output_bytes)

    @classmethod
    def _from_config(cls, config: LocalExecutorConfig) -> "LocalExecutor":
        return cls(max_output_bytes=config.max_output_bytes)

    def _check(self, session: ExecutionSession) -> None:
        if self._sessions.get(session.id) is not session:
            raise ValueError("session does not belong to this executor")

    async def connect(self, workspace, user_id, conversation_id) -> ExecutionSession:
        directory = workspace.materialize(user_id, conversation_id)
        session = ExecutionSession(
            short_id(), user_id, conversation_id, workspace,
            str(directory.root), handle=directory,
        )
        self._sessions[session.id] = session
        self._locks[session.id] = asyncio.Lock()
        return session

    async def disconnect(self, session: ExecutionSession) -> None:
        self._check(session)

    async def clean(self, session: ExecutionSession) -> None:
        if self._sessions.get(session.id) is not session:
            if self._closed.get(session.id) is session:
                return
            raise ValueError("session does not belong to this executor")
        self._sessions.pop(session.id, None)
        self._locks.pop(session.id, None)
        self._closed[session.id] = session

    async def rebuild(self, session: ExecutionSession) -> ExecutionSession:
        self._check(session)
        workspace, user_id, conversation_id = (
            session.workspace, session.user_id, session.conversation_id,
        )
        await self.clean(session)
        return await self.connect(workspace, user_id, conversation_id)

    async def execute_argv(
        self, session, argv, *, stdin=None, timeout=60, cancellation_token=None
    ):
        self._check(session)
        lock = self._locks.setdefault(session.id, asyncio.Lock())
        async with lock:
            env = os.environ.copy()
            env["WORKSPACE"] = session.workspace_path
            return await run_process(
                argv,
                cwd=session.workspace_path,
                env=env,
                stdin=stdin,
                timeout=timeout,
                max_output_bytes=self.max_output_bytes,
                cancellation_token=cancellation_token,
            )

    async def execute(self, session, command, *, timeout=60, cancellation_token=None):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command cannot be empty")
        # Clear shell startup variables (notably BASH_ENV) inherited from the host.
        return await self.execute_argv(
            session,
            [
                "/usr/bin/env",
                "-i",
                "PATH=/usr/bin:/bin",
                f"WORKSPACE={session.workspace_path}",
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                command,
            ],
            timeout=timeout,
            cancellation_token=cancellation_token,
        )

    async def run_tool(self, session, tool, record, context, cancellation_token=None):
        deps = dict(context.deps)
        directory = session.handle
        deps.update(
            runtime_root=session.workspace_path,
            workspace_dir=str(directory.workspace_dir),
            skills_dir=str(directory.skill_dir),
            filesystem_root=str(session.workspace.base_root),
            workspace_filesystem=session.workspace.get_filesystem(),
        )
        ctx = ToolContext(
            context.run_id,
            session_id=session.conversation_id,
            user_id=session.user_id,
            retry_count=context.retry_count,
            deps=deps,
            emit_event=context.emit_event,
        )
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
            from ....types.tool_call import ToolResult

            return ToolResult.timeout(record.id, timeout_seconds=timeout)
