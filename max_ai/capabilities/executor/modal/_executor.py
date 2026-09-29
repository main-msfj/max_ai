"""Modal Sandbox executor with explicit workspace upload/download.

SDK references: https://modal.com/docs/guide/sandbox-spawn and
https://modal.com/docs/guide/sandbox-files (reviewed 2026-09-15).
The optional SDK is imported only when connecting. By default the image is
max_ai's runtime Dockerfile; with your own ``image`` or ``dockerfile`` the
executor adds max_ai (from ``framework``), the agent user and /workspaces.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ....base.executor import ExecutionResult, ExecutionSession, SyncDirection
from ....core.executor.remote import (
    DEFAULT_FRAMEWORK,
    FRAMEWORK_PYTHON,
    RemoteExecutor,
    kill_argv,
    supervised,
)
from ....core.ids import short_id
from ._model import ModalExecutorConfig
from .sync import apply_snapshot, snapshot

if TYPE_CHECKING:
    from ....base.workspace import WorkspaceBase
    from ....core.termination import CancellationToken

_RUNTIME_DOCKERFILE = str(Path(__file__).resolve().parents[1] / "docker" / "Dockerfile")


@dataclass
class _Sandbox:
    """One session's Modal sandbox and its last synced workspace state."""

    sandbox: Any  # modal.Sandbox; the SDK is optional
    baseline: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False


class ModalExecutor(RemoteExecutor):
    """Runs commands in a Modal sandbox, one per conversation."""

    component_schema = ModalExecutorConfig
    component_provider_override = "max_ai.capabilities.executor.modal.ModalExecutor"

    def __init__(
        self,
        *,
        image: Any = None,
        dockerfile: str | None = None,
        framework: str = DEFAULT_FRAMEWORK,
        packages: list[str] | None = None,
        app_name: str = "maxai-runtime",
        network: str = "packages",
        allow_list: list[str] | None = None,
        lifetime: int = 3600,
        max_output_bytes: int = 1 << 20,
        uid: int = 1000,
    ) -> None:
        """Initialize ``ModalExecutor``.

        Parameters
        ----------
        image : str | modal.Image | None
            A registry image (``"python:3.12-slim"``) or a ``modal.Image``.
        dockerfile : str | None
            Path to your own Dockerfile. With neither ``image`` nor ``dockerfile``,
            max_ai's runtime Dockerfile is used (Python, uv, Node/npm, git, ripgrep).
            Your image only needs Linux, bash, git and python3 with pip and venv:
            the executor adds max_ai, the agent user and /workspaces on top.
        framework : str
            pip requirement that installs max_ai inside the sandbox.
        packages : list[str] | None
            Extra pip packages baked into the image (optional: the agent can also
            install what it needs when the network allows it).
        app_name : str
            Modal app the sandboxes belong to; created if missing.
        network : str
            "packages" (default): only package registries (pip, uv, npm) plus
            ``allow_list``. "internet": every site, or only ``allow_list`` when
            given. "none": no network.
        allow_list : list[str] | None
            Extra domains the sandbox may reach; ``*.`` wildcards allowed.
        lifetime : int
            Seconds a sandbox may live before Modal stops it (1 to 86400).
        max_output_bytes : int
            Most bytes kept from a command's stdout and from its stderr.
        uid : int
            Non-root user id commands run as.
        """
        self._init_network(network, allow_list)
        if not 1 <= lifetime <= 86400 or max_output_bytes <= 0:
            raise ValueError("Invalid lifetime or output limit")
        if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
            raise ValueError("uid must be a positive (non-root) user id")
        if image is not None and dockerfile is not None:
            raise ValueError("Pass image or dockerfile, not both")
        self.image, self.dockerfile, self.framework = image, dockerfile, framework
        self.app_name = app_name
        self.packages = list(packages or [])
        self.lifetime, self.max_output_bytes, self.uid = lifetime, max_output_bytes, uid
        self._sessions: dict[str, ExecutionSession] = {}

    def _to_config(self) -> ModalExecutorConfig:
        if self.image is not None and not isinstance(self.image, str):
            raise TypeError("Modal image must be a registry string to serialize")
        return ModalExecutorConfig(
            image=self.image,
            dockerfile=self.dockerfile,
            framework=self.framework,
            packages=self.packages,
            app_name=self.app_name,
            network=self.network,
            allow_list=self.allow_list,
            lifetime=self.lifetime,
            max_output_bytes=self.max_output_bytes,
            uid=self.uid,
        )

    def describe_environment(self) -> str:
        """Where commands run, for the system prompt."""
        ours = self.image is None and self.dockerfile is None
        text = (
            "Commands run in an isolated Linux sandbox (Modal) as a non-root user. "
            + self._network_description("pip, uv or npm" if ours else "pip")
        )
        if ours:
            text += " Python 3.11, uv, Node.js/npm, git and ripgrep are available."
        if self.packages:
            text += f" Preinstalled: {', '.join(self.packages)}."
        return text

    def _build_image(self, modal: Any) -> Any:
        """The image, with max_ai in /opt/maxai, the agent user and /workspaces.
        Modal caches every layer, so this builds once per change."""
        if self.image is None and self.dockerfile is None:
            image = modal.Image.from_dockerfile(
                _RUNTIME_DOCKERFILE,
                build_args={
                    "MAXAI": self.framework,
                    "AGENT_UID": str(self.uid),
                    "AGENT_GID": str(self.uid),
                },
            )
        else:
            if self.dockerfile is not None:
                image = modal.Image.from_dockerfile(self.dockerfile)
            elif isinstance(self.image, str):
                image = modal.Image.from_registry(self.image)
            else:
                image = self.image
            image = image.run_commands(
                f'python3 -m venv /opt/maxai && /opt/maxai/bin/pip install "{self.framework}"',
                f"id -u {self.uid} >/dev/null 2>&1 || useradd --uid {self.uid} --create-home agent",
                f"mkdir -p /workspaces && chown {self.uid}:{self.uid} /workspaces",
                # Modal puts /root on sys.path; pip scans it and fails as non-root.
                "chmod 755 /root",
            ).env({"HOME": "/tmp"})  # so `pip install --user` works as the agent user
        if self.packages:
            image = image.pip_install(*self.packages)
        return image

    @classmethod
    def _from_config(cls, config: ModalExecutorConfig) -> ModalExecutor:
        return cls(**config.model_dump())

    def _as_user(self, *argv: str) -> tuple[str, ...]:
        """Modal starts every sandbox process as root (the image's USER is
        ignored), so each one drops to ``uid`` before running."""
        return (
            "setpriv",
            f"--reuid={self.uid}",
            f"--regid={self.uid}",
            "--clear-groups",
            "--",
            *argv,
        )

    def _check(self, session: ExecutionSession) -> None:
        if self._sessions.get(session.id) is not session:
            raise ValueError("Session does not belong to this executor")

    async def connect(
        self, workspace: WorkspaceBase, user_id: str, conversation_id: str
    ) -> ExecutionSession:
        """Create a sandbox for this conversation."""
        try:
            import modal
        except ImportError as error:
            raise RuntimeError("Install Modal with: uv sync --extra runtime-modal") from error
        workspace.materialize(user_id, conversation_id)
        app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
        image = self._build_image(modal)
        root = f"/workspaces/{user_id}"
        sandbox = await modal.Sandbox.create.aio(
            "sleep",
            "infinity",
            app=app,
            image=image,
            timeout=self.lifetime,
            workdir="/workspaces",
            env={"WORKSPACE": root},
            block_network=self.network == "none",
            outbound_domain_allowlist=self._allowed_domains(),
        )
        session = ExecutionSession(
            short_id(), user_id, conversation_id, workspace, root, _Sandbox(sandbox)
        )
        self._sessions[session.id] = session
        try:
            check = await sandbox.exec.aio(
                *self._as_user(
                    "sh",
                    "-c",
                    'test "$(id -u)" -ne 0 && mkdir -p "$WORKSPACE" && test -w "$WORKSPACE"',
                ),
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

    async def _command(
        self,
        session: ExecutionSession,
        argv: list[str],
        *,
        stdin: str | None = None,
        timeout: float = 60,
        limit: int | None = None,
    ) -> ExecutionResult:
        """Run ``argv`` through max_ai's command supervisor in the sandbox."""
        self._check(session)
        handle = session.handle
        if handle.closed:
            raise RuntimeError("Modal sandbox is closed; rebuild it before executing")
        if not argv or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Command and finite positive timeout required")
        invocation = short_id()
        process = await handle.sandbox.exec.aio(
            *self._as_user(
                FRAMEWORK_PYTHON,
                "-m",
                "max_ai.capabilities.executor.modal.modal_command",
            ),
            workdir=session.workspace_path,
            timeout=math.ceil(timeout + 30),
        )
        payload = {
            "id": invocation,
            "argv": argv,
            "stdin": stdin,
            "timeout": timeout,
            "max_output_bytes": limit or self.max_output_bytes,
        }

        async def communicate() -> ExecutionResult:
            process.stdin.write(json.dumps(payload).encode())
            process.stdin.write_eof()
            await process.stdin.drain.aio()
            stdout, stderr, code = await asyncio.gather(
                process.stdout.read.aio(),
                process.stderr.read.aio(),
                process.wait.aio(),
            )
            if code != 0:
                raise RuntimeError(stderr or "Modal command supervisor failed")
            return ExecutionResult(**json.loads(stdout))

        task = asyncio.create_task(communicate())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:

            async def cancel_remote() -> None:
                await handle.sandbox.filesystem.write_text.aio(
                    "cancel",
                    f"/tmp/maxai-cancel-{invocation}",
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

    async def execute_argv(
        self,
        session: ExecutionSession,
        argv: list[str],
        *,
        stdin: str | None = None,
        timeout: float = 60,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        """Run ``argv`` through the command supervisor, one command at a time."""
        self._check(session)
        if cancellation_token is not None and cancellation_token.is_cancelled():
            raise asyncio.CancelledError
        async with session.handle.lock:
            task = asyncio.create_task(self._command(session, argv, stdin=stdin, timeout=timeout))
            if cancellation_token is not None:
                cancellation_token.link_future(task)
            return await task

    async def execute(
        self,
        session: ExecutionSession,
        command: str,
        *,
        timeout: float,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        """Run ``command`` with bash in the sandbox; see ``supervised``."""
        if not command.strip():
            raise ValueError("Command cannot be empty")
        self._check(session)
        handle = session.handle
        if handle.closed:
            raise RuntimeError("Modal sandbox is closed; rebuild it before executing")
        run_id = short_id()
        async with handle.lock:
            start = time.monotonic()
            process = await handle.sandbox.exec.aio(
                *self._as_user(*supervised(command, timeout, run_id)),
                workdir=session.workspace_path,
                timeout=math.ceil(timeout + 30),
            )

            async def wait() -> ExecutionResult:
                stdout, stderr, code = await asyncio.gather(
                    process.stdout.read.aio(),
                    process.stderr.read.aio(),
                    process.wait.aio(),
                )
                return ExecutionResult(stdout, stderr, code)

            task = asyncio.create_task(wait())
            if cancellation_token is not None:
                cancellation_token.link_future(task)
            try:
                result = await task
            except asyncio.CancelledError:
                kill = await handle.sandbox.exec.aio(
                    *self._as_user(*kill_argv(run_id)), timeout=10
                )
                await kill.wait.aio()
                raise
        timed_out = result.exit_code != 0 and time.monotonic() - start >= timeout
        return replace(result, timed_out=timed_out)

    async def sync(self, session: ExecutionSession, direction: SyncDirection) -> None:
        """Copy the workspace into the sandbox ("to_environment") or back
        ("to_workspace"), only what changed since the last sync."""
        self._check(session)
        if direction not in {"to_environment", "to_workspace"}:
            raise ValueError("Unknown synchronization direction")
        async with session.handle.lock:
            root = session.workspace.materialize(session.user_id, session.conversation_id).root
            local = snapshot(root)
            remote_result = await self._command(
                session,
                [
                    FRAMEWORK_PYTHON,
                    "-m",
                    "max_ai.capabilities.executor.modal.sync",
                    "snapshot",
                    session.workspace_path,
                ],
                timeout=120,
                limit=24 << 20,
            )
            if remote_result.exit_code != 0 or remote_result.truncated or remote_result.timed_out:
                raise RuntimeError(remote_result.stderr or "Workspace snapshot failed")
            remote = json.loads(remote_result.stdout)
            if not isinstance(remote, dict):
                raise ValueError("Invalid remote workspace snapshot")
            source, destination = (
                (local, remote) if direction == "to_environment" else (remote, local)
            )
            baseline = session.handle.baseline
            desired = dict(destination)
            changes = set(source) | set(baseline)
            for name in changes:
                before, incoming, current = (
                    baseline.get(name),
                    source.get(name),
                    destination.get(name),
                )
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
                        session,
                        [
                            FRAMEWORK_PYTHON,
                            "-m",
                            "max_ai.capabilities.executor.modal.sync",
                            "apply",
                            session.workspace_path,
                        ],
                        stdin=json.dumps({"desired": desired, "expected": destination}),
                        timeout=120,
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

    async def disconnect(self, session: ExecutionSession) -> None:
        self._check(session)
        # No detach before clean: Modal invalidates the detached handle.

    async def clean(self, session: ExecutionSession) -> None:
        """Terminate the sandbox; calling it twice is fine."""
        self._check(session)
        handle = session.handle
        if not handle.closed:
            await handle.sandbox.terminate.aio(wait=True)
            await handle.sandbox.detach.aio()
            handle.closed = True

    async def rebuild(self, session: ExecutionSession) -> ExecutionSession:
        """Save the workspace, then a fresh sandbox for the same conversation."""
        self._check(session)
        await self.sync(session, "to_workspace")
        await self.clean(session)
        return await self.connect(session.workspace, session.user_id, session.conversation_id)
