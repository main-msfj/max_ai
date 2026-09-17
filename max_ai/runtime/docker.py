"""Docker runtime executor using the Docker CLI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

from ..base.runtime_executor import ExecutionSession
from .process import run_process
from .remote import RemoteExecutor


@dataclass
class _Container:
    name: str
    started: bool = False
    may_exist: bool = False


class DockerExecutor(RemoteExecutor):
    def __init__(self, *, image: str = "maxai-runtime:latest", network: str = "none",
                 max_output_bytes: int = 1 << 20):
        if network not in {"none", "unrestricted"}:
            raise ValueError("network must be 'none' or 'unrestricted'")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.image, self.network, self.max_output_bytes = image, network, max_output_bytes
        super().__init__()
        self._sessions = {}
        self._locks = {}
        self._closed = {}

    def _check(self, session):
        if self._sessions.get(session.id) is not session:
            raise ValueError("session does not belong to this executor")

    async def connect(self, workspace, user_id, conversation_id):
        directory = workspace.materialize(user_id, conversation_id)
        handle = _Container(f"maxai-runtime-{uuid4().hex}")
        session = ExecutionSession(uuid4().hex, user_id, conversation_id, workspace,
                                   f"/workspaces/{user_id}", handle)
        self._sessions[session.id] = session
        self._locks[session.id] = asyncio.Lock()
        try:
            await self._start(session, directory.root)
            return session
        except BaseException:
            await self.clean(session)
            raise

    async def _start(self, session, root):
        h = session.handle
        if "," in str(root):
            raise ValueError("Docker workspace paths cannot contain commas")
        args = ["run", "--detach", "--pull=never", "--name", h.name,
                "--network=none" if self.network == "none" else "--network=bridge",
                "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--init", "--pids-limit=128", "--memory=512m", "--cpus=1",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m", "--mount",
                f"type=bind,src={root},dst={session.workspace_path}",
                "--workdir", session.workspace_path, "--env", f"WORKSPACE={session.workspace_path}",
                self.image, "sleep", "infinity"]
        h.may_exist = True
        try:
            result = await run_process(["docker", *args], timeout=30, max_output_bytes=self.max_output_bytes)
            if result.exit_code != 0:
                raise RuntimeError(result.stderr or "Docker startup failed")
            h.started = True
            verify = await self._docker_exec(session, ["sh", "-c", "test \"$(id -u)\" -ne 0 && test -w \"$WORKSPACE\""])
            if verify.exit_code != 0:
                raise RuntimeError("Docker image user must be non-root and workspace-writable")
        except BaseException:
            await self._destroy_container(session)
            raise

    async def _docker_exec(self, session, argv, **kwargs):
        stdin = kwargs.get("stdin")
        interactive = ["-i"] if stdin is not None else []
        return await run_process(["docker", "exec", *interactive, "--workdir", session.workspace_path,
                                  session.handle.name, *argv], max_output_bytes=self.max_output_bytes, **kwargs)

    async def execute_argv(self, session, argv, *, stdin=None, timeout=60, cancellation_token=None):
        self._check(session)
        async with self._locks[session.id]:
            try:
                if not session.handle.started:
                    await self._start(session, session.workspace.materialize(session.user_id, session.conversation_id).root)
                result = await self._docker_exec(session, list(argv), stdin=stdin, timeout=timeout,
                                                 cancellation_token=cancellation_token)
                if result.timed_out:
                    await self._destroy_container(session)
                return result
            except (asyncio.CancelledError,):
                await self._destroy_container(session)
                raise
            except Exception:
                await self._destroy_container(session)
                raise

    async def execute(self, session, command, *, timeout=60, cancellation_token=None):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command cannot be empty")
        return await self.execute_argv(session, ["bash", "--noprofile", "--norc", "-c", command],
                                       timeout=timeout, cancellation_token=cancellation_token)

    async def sync(self, session, direction):
        self._check(session)

    async def disconnect(self, session):
        self._check(session)

    async def _remove_owned(self, handle):
        if not handle.may_exist:
            return
        result = await run_process(["docker", "rm", "--force", handle.name], timeout=15,
                                   max_output_bytes=self.max_output_bytes)
        if result.exit_code != 0 and "No such container" not in result.stderr:
            raise RuntimeError(result.stderr or "Docker cleanup failed")
        handle.may_exist = False

    async def _destroy_container(self, session):
        handle = session.handle
        if handle.may_exist:
            cleanup = asyncio.create_task(self._remove_owned(handle))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            handle.started = False

    async def clean(self, session):
        if self._sessions.get(session.id) is not session:
            if self._closed.get(session.id) is session:
                return
            raise ValueError("session does not belong to this executor")
        h = session.handle
        if h.may_exist:
            try:
                cleanup = asyncio.create_task(self._remove_owned(h))
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            else:
                h.started = False
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
