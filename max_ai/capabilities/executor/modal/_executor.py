"""Modal Sandbox executor with explicit workspace upload/download.

SDK references: https://modal.com/docs/guide/sandbox-spawn and
https://modal.com/docs/guide/sandbox-files (reviewed 2026-09-15).
The optional SDK is imported only when connecting. Without ``image`` the
executor builds one (Debian + max_ai + its dependencies + ``packages``);
a custom image must contain max_ai, Bash, and /workspaces writable by ``uid``.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ....base.executor import ExecutionResult, ExecutionSession
from ....core.executor.remote import RemoteExecutor
from ....core.ids import short_id
from ._model import ModalExecutorConfig
from .sync import apply_snapshot, snapshot


@dataclass
class _Sandbox:
    """_Sandbox represents structured data used by the capability system."""
    sandbox: object
    baseline: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False


class ModalExecutor(RemoteExecutor):
    """ModalExecutor provides the Modalexecution implementation."""
    component_schema = ModalExecutorConfig
    component_provider_override = "max_ai.capabilities.executor.modal.ModalExecutor"

    def __init__(self, *, image=None, packages: list[str] | None = None,
                 app_name: str = "maxai-runtime",
                 network: str = "none", lifetime: int = 3600,
                 max_output_bytes: int = 1 << 20, uid: int = 1000):
        """Initialize ``ModalExecutor``.

Parameters
----------
image
    Value supplied for ``image``.
packages : list[str] | None
    Value supplied for ``packages``.
app_name : str
    Value supplied for ``app_name``.
network : str
    Value supplied for ``network``.
lifetime : int
    Value supplied for ``lifetime``.
max_output_bytes : int
    Value supplied for ``max_output_bytes``.
uid : int
    Value supplied for ``uid``."""
        if network not in {"none", "unrestricted"}:
            raise ValueError("network must be 'none' or 'unrestricted'")
        if not 1 <= lifetime <= 86400 or max_output_bytes <= 0:
            raise ValueError("Invalid lifetime or output limit")
        if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
            raise ValueError("uid must be a positive (non-root) user id")
        if packages and image is not None:
            raise ValueError("packages only apply to the default image; install them in yours")
        self.image, self.app_name, self.network = image, app_name, network
        self.packages = list(packages or [])
        self.lifetime, self.max_output_bytes, self.uid = lifetime, max_output_bytes, uid
        self._sessions: dict[str, ExecutionSession] = {}

    def _to_config(self) -> ModalExecutorConfig:
        """Build the serializable configuration for ``ModalExecutor``."""
        if self.image is not None and not isinstance(self.image, str):
            raise TypeError("Modal image must be a registry string to serialize")
        return ModalExecutorConfig(
            image=self.image, packages=self.packages, app_name=self.app_name, network=self.network,
            lifetime=self.lifetime, max_output_bytes=self.max_output_bytes, uid=self.uid,
        )

    def describe_environment(self) -> str:
        """Describe the environment exposed by ``ModalExecutor``."""
        parts = ["Commands run in an isolated Linux sandbox (Modal) as a non-root user"]
        if self.network == "none":
            parts.append("with no network access: pip install and downloads fail")
        else:
            parts.append("with internet access")
        text = " ".join(parts) + "."
        if self.image is None:
            extra = f", plus {', '.join(self.packages)}" if self.packages else ""
            text += f" Python 3.11 is available{extra}; these are already installed."
        return text

    def _default_image(self, modal):
        """Debian + max_ai's own dependencies + ``packages`` (Modal caches the
        build), with this machine's max_ai source mounted at start."""
        requires = [r for r in importlib.metadata.requires("maxai") or [] if "extra ==" not in r]
        source = Path(__file__).resolve().parents[3]  # .../max_ai
        return (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install(*requires, *self.packages)
            .run_commands(
                f"useradd --uid {self.uid} --create-home agent",
                f"mkdir /workspaces && chown {self.uid}:{self.uid} /workspaces",
            )
            .env({"PYTHONPATH": "/opt/maxai", "HOME": "/home/agent"})
            .add_local_dir(source, "/opt/maxai/max_ai", ignore=["**/__pycache__"])
        )

    @classmethod
    def _from_config(cls, config: ModalExecutorConfig) -> "ModalExecutor":
        """Create an instance from its configuration for ``ModalExecutor``.

Parameters
----------
config : ModalExecutorConfig
    Value supplied for ``config``."""
        return cls(**config.model_dump())

    def _as_user(self, *argv: str) -> tuple[str, ...]:
        """Modal starts every sandbox process as root (the image's USER is
        ignored), so each one drops to ``uid`` before running."""
        return ("setpriv", f"--reuid={self.uid}", f"--regid={self.uid}", "--clear-groups", "--", *argv)

    def _check(self, session):
        """Perform the internal ``check`` operation for ``ModalExecutor``.

Parameters
----------
session
    Value supplied for ``session``."""
        if self._sessions.get(session.id) is not session:
            raise ValueError("Session does not belong to this executor")

    async def connect(self, workspace, user_id, conversation_id):
        """Open required resources for ``ModalExecutor``.

Parameters
----------
workspace
    Value supplied for ``workspace``.
user_id
    Value supplied for ``user_id``.
conversation_id
    Value supplied for ``conversation_id``."""
        try:
            import modal
        except ImportError as error:
            raise RuntimeError("Install Modal with: uv sync --extra runtime-modal") from error
        workspace.materialize(user_id, conversation_id)
        app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
        if self.image is None:
            image = self._default_image(modal)
        elif isinstance(self.image, str):
            image = modal.Image.from_registry(self.image)
        else:
            image = self.image
        root = f"/workspaces/{user_id}"
        sandbox = await modal.Sandbox.create.aio(
            "sleep", "infinity", app=app, image=image,
            timeout=self.lifetime, workdir="/workspaces",
            env={"WORKSPACE": root}, block_network=self.network == "none",
        )
        session = ExecutionSession(short_id(), user_id, conversation_id, workspace,
                                   root, _Sandbox(sandbox))
        self._sessions[session.id] = session
        try:
            check = await sandbox.exec.aio(
                *self._as_user("sh", "-c", 'test "$(id -u)" -ne 0 && mkdir -p "$WORKSPACE" && test -w "$WORKSPACE"'),
                timeout=30,
            )
            if await check.wait.aio() != 0:
                raise RuntimeError(
                    f"Modal image needs setpriv and /workspaces writable by uid {self.uid}"
                )
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
        """Perform the internal ``command`` operation for ``ModalExecutor``.

Parameters
----------
session
    Value supplied for ``session``.
argv
    Value supplied for ``argv``.
stdin
    Value supplied for ``stdin``.
timeout
    Value supplied for ``timeout``.
limit
    Value supplied for ``limit``."""
        self._check(session)
        handle = session.handle
        if handle.closed:
            raise RuntimeError("Modal sandbox is closed; rebuild it before executing")
        if not argv or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Command and finite positive timeout required")
        invocation = short_id()
        process = await handle.sandbox.exec.aio(
            *self._as_user("python", "-m", "max_ai.capabilities.executor.modal.modal_command"),
            workdir=session.workspace_path, timeout=math.ceil(timeout + 30),
        )
        payload = {"id": invocation, "argv": argv, "stdin": stdin, "timeout": timeout,
                   "max_output_bytes": limit or self.max_output_bytes}

        async def communicate():
            """Perform the ``communicate`` operation for ``ModalExecutor``."""
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
                """Perform the ``cancel remote`` operation for ``ModalExecutor``."""
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
        """Run an argument vector for ``ModalExecutor``.

Parameters
----------
session
    Value supplied for ``session``.
argv
    Value supplied for ``argv``.
stdin
    Value supplied for ``stdin``.
timeout
    Value supplied for ``timeout``.
cancellation_token
    Value supplied for ``cancellation_token``."""
        self._check(session)
        if cancellation_token is not None and cancellation_token.is_cancelled():
            raise asyncio.CancelledError
        async with session.handle.lock:
            task = asyncio.create_task(self._command(session, argv, stdin=stdin, timeout=timeout))
            if cancellation_token is not None:
                cancellation_token.link_future(task)
            return await task

    async def execute(self, session, command, *, timeout=60, cancellation_token=None):
        """Execute the requested operation for ``ModalExecutor``.

Parameters
----------
session
    Value supplied for ``session``.
command
    Value supplied for ``command``.
timeout
    Value supplied for ``timeout``.
cancellation_token
    Value supplied for ``cancellation_token``."""
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Command cannot be empty")
        return await self.execute_argv(session, ["bash", "--noprofile", "--norc", "-c", command],
                                       timeout=timeout, cancellation_token=cancellation_token)

    async def sync(self, session, direction):
        """Perform the ``sync`` operation for ``ModalExecutor``.

Parameters
----------
session
    Value supplied for ``session``.
direction
    Value supplied for ``direction``."""
        self._check(session)
        if direction not in {"to_environment", "to_workspace"}:
            raise ValueError("Unknown synchronization direction")
        async with session.handle.lock:
            root = session.workspace.materialize(session.user_id, session.conversation_id).root
            local = snapshot(root)
            remote_result = await self._command(
                session, ["python", "-m", "max_ai.capabilities.executor.modal.sync", "snapshot", session.workspace_path],
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
                        session, ["python", "-m", "max_ai.capabilities.executor.modal.sync", "apply", session.workspace_path],
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
        """Release resources held for ``ModalExecutor``.

Parameters
----------
session
    Value supplied for ``session``."""
        self._check(session)
        # No detach before clean: Modal invalidates the detached handle.

    async def clean(self, session):
        """Remove temporary resources owned for ``ModalExecutor``.

Parameters
----------
session
    Value supplied for ``session``."""
        self._check(session)
        handle = session.handle
        if not handle.closed:
            await handle.sandbox.terminate.aio(wait=True)
            await handle.sandbox.detach.aio()
            handle.closed = True

    async def rebuild(self, session):
        """Recreate the runtime environment for ``ModalExecutor``.

Parameters
----------
session
    Value supplied for ``session``."""
        self._check(session)
        await self.sync(session, "to_workspace")
        await self.clean(session)
        return await self.connect(session.workspace, session.user_id, session.conversation_id)
