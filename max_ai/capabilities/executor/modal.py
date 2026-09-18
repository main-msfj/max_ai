"""Modal Sandbox executor with explicit workspace upload/download.

SDK references: https://modal.com/docs/guide/sandbox-spawn and
https://modal.com/docs/guide/sandbox-files (reviewed 2026-09-15).
The optional SDK is imported only when connecting. Images must contain max_ai,
native tool dependencies, Bash, and a non-root user with writable /workspaces.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from uuid import uuid4

from ...base.executor import ExecutionResult
from ...base.executor import ExecutionSession
from .remote import RemoteExecutor
from .sync import apply_snapshot, snapshot


@dataclass
class _Sandbox:
    sandbox: object
    baseline: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False


class ModalExecutor(RemoteExecutor):
    def __init__(self, *, image, app_name: str = "maxai-runtime",
                 network: str = "none", lifetime: int = 3600,
                 max_output_bytes: int = 1 << 20):
        if network not in {"none", "unrestricted"}:
            raise ValueError("network must be 'none' or 'unrestricted'")
        if not 1 <= lifetime <= 86400 or max_output_bytes <= 0:
            raise ValueError("Invalid lifetime or output limit")
        self.image, self.app_name, self.network = image, app_name, network
        self.lifetime, self.max_output_bytes = lifetime, max_output_bytes
        self._sessions: dict[str, ExecutionSession] = {}

    def _check(self, session):
        if self._sessions.get(session.id) is not session:
            raise ValueError("Session does not belong to this executor")

    async def connect(self, workspace, user_id, conversation_id):
        try:
            import modal
        except ImportError as error:
            raise RuntimeError("Install Modal with: uv sync --extra runtime-modal") from error
        workspace.materialize(user_id, conversation_id)
        app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
        image = modal.Image.from_registry(self.image) if isinstance(self.image, str) else self.image
        root = f"/workspaces/{user_id}"
        sandbox = await modal.Sandbox.create.aio(
            "sleep", "infinity", app=app, image=image,
            timeout=self.lifetime, workdir="/workspaces",
            env={"WORKSPACE": root}, block_network=self.network == "none",
        )
        session = ExecutionSession(uuid4().hex, user_id, conversation_id, workspace,
                                   root, _Sandbox(sandbox))
        self._sessions[session.id] = session
        try:
            check = await sandbox.exec.aio(
                "sh", "-c", 'test "$(id -u)" -ne 0 && mkdir -p "$WORKSPACE" && test -w "$WORKSPACE"',
                timeout=30,
            )
            if await check.wait.aio() != 0:
                raise RuntimeError("Modal image needs a non-root user with writable /workspaces")
            return session
        except BaseException:
            cleanup = asyncio.create_task(self.clean(session))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            raise

    async def _command(self, session, argv, *, stdin=None, timeout=60, limit=None):
        self._check(session)
        handle = session.handle
        if handle.closed:
            raise RuntimeError("Modal sandbox is closed; rebuild it before executing")
        if not argv or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Command and finite positive timeout required")
        invocation = uuid4().hex
        process = await handle.sandbox.exec.aio(
            "python", "-m", "max_ai.capabilities.executor.modal_command",
            workdir=session.workspace_path, timeout=math.ceil(timeout + 30),
        )
        payload = {"id": invocation, "argv": argv, "stdin": stdin, "timeout": timeout,
                   "max_output_bytes": limit or self.max_output_bytes}

        async def communicate():
            process.stdin.write(json.dumps(payload).encode())
            process.stdin.write_eof()
            await process.stdin.drain.aio()
            stdout, stderr, code = await asyncio.gather(
                process.stdout.read.aio(), process.stderr.read.aio(), process.wait.aio(),
            )
            if code != 0:
                raise RuntimeError(stderr or "Modal command supervisor failed")
            return ExecutionResult(**json.loads(stdout))

        task = asyncio.create_task(communicate())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            async def cancel_remote():
                await handle.sandbox.filesystem.write_text.aio(
                    "cancel", f"/tmp/maxai-cancel-{invocation}",
                )
                # The SDK's remote timeout is a second bound if the command
                # supervisor fails. Keep the filesystem until sync completes.
                await task
            cleanup = asyncio.create_task(cancel_remote())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def execute_argv(self, session, argv, *, stdin=None, timeout=60, cancellation_token=None):
        self._check(session)
        if cancellation_token is not None and cancellation_token.is_cancelled():
            raise asyncio.CancelledError
        async with session.handle.lock:
            task = asyncio.create_task(self._command(session, argv, stdin=stdin, timeout=timeout))
            if cancellation_token is not None:
                cancellation_token.link_future(task)
            return await task

    async def execute(self, session, command, *, timeout=60, cancellation_token=None):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Command cannot be empty")
        return await self.execute_argv(session, ["bash", "--noprofile", "--norc", "-c", command],
                                       timeout=timeout, cancellation_token=cancellation_token)

    async def sync(self, session, direction):
        self._check(session)
        if direction not in {"to_environment", "to_workspace"}:
            raise ValueError("Unknown synchronization direction")
        async with session.handle.lock:
            root = session.workspace.materialize(session.user_id, session.conversation_id).root
            local = snapshot(root)
            remote_result = await self._command(
                session, ["python", "-m", "max_ai.capabilities.executor.sync", "snapshot", session.workspace_path],
                timeout=120, limit=24 << 20,
            )
            if remote_result.exit_code != 0 or remote_result.truncated or remote_result.timed_out:
                raise RuntimeError(remote_result.stderr or "Workspace snapshot failed")
            remote = json.loads(remote_result.stdout)
            if not isinstance(remote, dict):
                raise ValueError("Invalid remote workspace snapshot")
            source, destination = (local, remote) if direction == "to_environment" else (remote, local)
            baseline = session.handle.baseline
            desired = dict(destination)
            changes = set(source) | set(baseline)
            for name in changes:
                before, incoming, current = baseline.get(name), source.get(name), destination.get(name)
                if incoming == before:
                    continue
                if current != before and current != incoming:
                    raise RuntimeError(f"Workspace sync conflict: {name}")
                if incoming is None:
                    desired.pop(name, None)
                else:
                    desired[name] = incoming
            if desired != destination:
                if direction == "to_environment":
                    result = await self._command(
                        session, ["python", "-m", "max_ai.capabilities.executor.sync", "apply", session.workspace_path],
                        stdin=json.dumps({"desired": desired, "expected": destination}), timeout=120,
                    )
                    if result.exit_code != 0 or result.timed_out:
                        raise RuntimeError(result.stderr or "Workspace upload failed")
                else:
                    apply_snapshot(root, desired, destination)
            # Advance only synchronized paths. Unrelated concurrent changes
            # still differ from the baseline and transfer on the next pass.
            for name in set(source) | set(desired) | set(baseline):
                if source.get(name) == desired.get(name):
                    if name in source:
                        baseline[name] = source[name]
                    else:
                        baseline.pop(name, None)

    async def disconnect(self, session):
        self._check(session)
        # No detach before clean: Modal invalidates the detached handle.

    async def clean(self, session):
        self._check(session)
        handle = session.handle
        if not handle.closed:
            await handle.sandbox.terminate.aio(wait=True)
            await handle.sandbox.detach.aio()
            handle.closed = True

    async def rebuild(self, session):
        self._check(session)
        await self.sync(session, "to_workspace")
        await self.clean(session)
        return await self.connect(session.workspace, session.user_id, session.conversation_id)
