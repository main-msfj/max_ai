"""Native, shared-filesystem executor — no sandbox isolation, no containers."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from ....base.executor import ExecutionResult, ExecutionSession, ExecutorBase
from ....base.tools import ToolContext
from ....config import setting
from ....core.executor.process import run_process
from ....core.ids import short_id
from ....types.tool_call import ToolResult
from ._model import LocalExecutorConfig

if TYPE_CHECKING:
    from ....base.tools import CoreTool
    from ....base.workspace import WorkspaceBase
    from ....core.termination import CancellationToken
    from ....types.tool_call import ToolCallRecord


class LocalExecutor(ExecutorBase):
    """Runs on the host, in the user's workspace. No isolation.

    Commands are off by default: the Agent gets no bash tool. With
    ``allow_commands=True`` bash runs on this machine and the user approves
    every command.
    """

    component_provider_override = "max_ai.capabilities.executor.local.LocalExecutor"
    component_schema = LocalExecutorConfig

    def __init__(self, *, allow_commands: bool = False, max_output_bytes: int = 1 << 20) -> None:
        """Initialize ``LocalExecutor``.

        Parameters
        ----------
        allow_commands : bool
            Give the agent bash on this machine, with approval for each command.
        max_output_bytes : int
            Most bytes kept from a command's stdout and from its stderr.
        """
        super().__init__()
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.allow_commands = allow_commands
        self.max_output_bytes = max_output_bytes
        self._sessions: dict[str, ExecutionSession] = {}
        self._closed: dict[str, ExecutionSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def runs_commands(self) -> bool:
        return self.allow_commands

    def describe_environment(self) -> str | None:
        """Where commands run, for the system prompt."""
        if not self.allow_commands:
            return None
        return (
            "Commands run on the user's own machine, not in a sandbox, and the "
            "user approves each one; prefer the file tools when they are enough."
        )

    def _to_config(self) -> LocalExecutorConfig:
        return LocalExecutorConfig(
            allow_commands=self.allow_commands, max_output_bytes=self.max_output_bytes
        )

    @classmethod
    def _from_config(cls, config: LocalExecutorConfig) -> LocalExecutor:
        return cls(allow_commands=config.allow_commands, max_output_bytes=config.max_output_bytes)

    def _check(self, session: ExecutionSession) -> None:
        if self._sessions.get(session.id) is not session:
            raise ValueError("session does not belong to this executor")

    async def connect(
        self, workspace: WorkspaceBase, user_id: str, conversation_id: str
    ) -> ExecutionSession:
        """Use the user's workspace folder on the host."""
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
        """Forget the session; calling it twice is fine."""
        if self._sessions.get(session.id) is not session:
            if self._closed.get(session.id) is session:
                return
            raise ValueError("session does not belong to this executor")
        self._sessions.pop(session.id, None)
        self._locks.pop(session.id, None)
        self._closed[session.id] = session

    async def rebuild(self, session: ExecutionSession) -> ExecutionSession:
        """A new session over the same workspace."""
        self._check(session)
        workspace, user_id, conversation_id = (
            session.workspace, session.user_id, session.conversation_id,
        )
        await self.clean(session)
        return await self.connect(workspace, user_id, conversation_id)

    async def execute_argv(
        self,
        session: ExecutionSession,
        argv: list[str],
        *,
        stdin: str | None = None,
        timeout: float = 60,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        """Run ``argv`` from the workspace, one command at a time per session."""
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

    async def execute(
        self,
        session: ExecutionSession,
        command: str,
        *,
        timeout: float,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        """Run ``command`` with bash; run_process kills its whole process group."""
        if not command.strip():
            raise ValueError("command cannot be empty")
        # A clean environment: the host's PATH, HOME and locale so its tools
        # work, but none of its secrets (API keys) and no BASH_ENV.
        return await self.execute_argv(
            session,
            [
                "/usr/bin/env",
                "-i",
                f"PATH={os.environ.get('PATH', '/usr/bin:/bin')}",
                f"HOME={os.environ.get('HOME', '/tmp')}",
                f"LANG={os.environ.get('LANG', 'C.UTF-8')}",
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

    async def run_tool(
        self,
        session: ExecutionSession,
        tool: CoreTool,
        record: ToolCallRecord,
        context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        """Run the tool in this process, with the workspace paths in its deps."""
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
        timeout = setting.tool_timeout_seconds
        try:
            return await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return ToolResult.timeout(record.id, timeout_seconds=timeout)
