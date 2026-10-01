"""Docker runtime executor using the Docker CLI."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ....base.executor import ExecutionResult, ExecutionSession, SyncDirection
from ....core.executor.process import run_process
from ....core.executor.remote import RemoteExecutor, kill_argv, supervised
from ....core.executor.sync import sync_session
from ....core.ids import short_id
from ....errors.executor import SandboxLost
from ._model import DockerExecutorConfig

if TYPE_CHECKING:
    from ....base.workspace import WorkspaceBase
    from ....core.termination import CancellationToken

_RUNTIME_DOCKERFILE = Path(__file__).with_name("Dockerfile")
_UID = 1000
# Added on top of a developer's image: the agent user and /workspaces.
_AGENT_LAYER = """FROM {base}
USER root
RUN (id -u {uid} >/dev/null 2>&1 || useradd --uid {uid} --create-home agent) \\
 && mkdir -p /workspaces && chown {uid}:{uid} /workspaces && chmod 755 /root
ENV HOME=/tmp PATH=/tmp/.local/bin:$PATH
WORKDIR /workspaces
"""
# What any image needs: the bash wrapper and the sync use only these.
IMAGE_NEEDS = "bash and GNU coreutils, findutils and tar (any Debian or Ubuntu base has them)"


@dataclass
class _Container:
    """One session's container and its last synced workspace state."""

    name: str
    started: bool = False
    may_exist: bool = False
    baseline: dict[str, str] = field(default_factory=dict)  # last synced state


class DockerExecutor(RemoteExecutor):
    """Runs commands in a local Docker container.

    Network is ``"none"`` (default) or ``"internet"``. Docker cannot filter
    by domain, so ``"packages"`` and ``allow_list`` are not supported here;
    use ModalExecutor for that.

    The workspace is copied into the container and back (no bind mount), so
    it also works when Docker runs elsewhere (Docker-outside-of-Docker, a
    remote daemon). The image is built once and cached by Docker: max_ai's runtime
    Dockerfile by default, or your ``image``/``dockerfile`` with the agent user
    and /workspaces added on top. Nothing of max_ai goes inside.
    """

    supports_allow_list = False
    component_schema = DockerExecutorConfig
    component_provider_override = "max_ai.capabilities.executor.docker.DockerExecutor"

    def __init__(
        self,
        *,
        image: str | None = None,
        dockerfile: str | None = None,
        network: str = "none",
        max_output_bytes: int = 1 << 20,
    ) -> None:
        """Initialize ``DockerExecutor``.

        Parameters
        ----------
        image : str | None
            Your image (``"python:3.12-slim"``). It needs bash and GNU coreutils,
            findutils and tar.
        dockerfile : str | None
            Path to your own Dockerfile, built with its folder as context.
        network : str
            "none" (default) or "internet". ``allow_list`` does not apply to Docker.
        max_output_bytes : int
            Most bytes kept from a command's stdout and from its stderr.
        """
        self._init_network(network, None)
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if image is not None and dockerfile is not None:
            raise ValueError("Pass image or dockerfile, not both")
        self.image, self.dockerfile = image, dockerfile
        self.max_output_bytes = max_output_bytes
        self._runtime_image: str | None = None
        self._image_lock = asyncio.Lock()
        super().__init__()
        self._sessions: dict[str, ExecutionSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closed: dict[str, ExecutionSession] = {}

    def describe_environment(self) -> str:
        """Where commands run, for the system prompt."""
        ours = self.image is None and self.dockerfile is None
        text = (
            "Commands run in an isolated Docker container as a non-root user. "
            + self._network_description("pip, uv or npm" if ours else "pip")
            + " Only the workspace and /tmp are writable."
        )
        if ours:
            text += " Python 3.12, uv, Node.js 22/npm, git and ripgrep are available."
        return text

    async def prepare(self) -> None:
        await self._ensure_image()

    async def _ensure_image(self) -> str:
        """Build the runtime image once (Docker caches it by tag) and return its tag."""
        async with self._image_lock:
            if self._runtime_image is None:
                self._runtime_image = await self._build_image()
            return self._runtime_image

    async def _build_image(self) -> str:
        """Our runtime image, or the developer's with the agent layer on top."""
        if self.image is None and self.dockerfile is None:
            recipe = _RUNTIME_DOCKERFILE.read_text()
            tag = _tag("maxai-runtime", recipe)
            await self._docker_build(tag, ["-"], stdin=recipe)
            return tag
        base = self.image
        if self.dockerfile is not None:
            path = Path(self.dockerfile).resolve()
            base = _tag("maxai-base", path.read_text())
            await self._docker_build(base, ["-f", str(path), str(path.parent)])
        layer = _AGENT_LAYER.format(base=base, uid=_UID)
        tag = _tag("maxai-runtime", layer)
        await self._docker_build(tag, ["-"], stdin=layer)
        return tag

    async def _docker_build(self, tag: str, args: list[str], stdin: str | None = None) -> None:
        """``docker build`` unless ``tag`` already exists locally."""
        exists = await run_process(["docker", "image", "inspect", tag], timeout=30)
        if exists.exit_code == 0:
            return
        result = await run_process(
            ["docker", "build", "-t", tag, *args],
            stdin=stdin,
            timeout=3600,
            max_output_bytes=self.max_output_bytes,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Building the Docker image {tag} failed. Your image needs "
                f"{IMAGE_NEEDS}.\n{result.stderr[-2000:]}"
            )

    def _to_config(self) -> DockerExecutorConfig:
        return DockerExecutorConfig(
            image=self.image,
            dockerfile=self.dockerfile,
            network=self.network,
            max_output_bytes=self.max_output_bytes,
        )

    @classmethod
    def _from_config(cls, config: DockerExecutorConfig) -> DockerExecutor:
        return cls(**config.model_dump())

    def _check(self, session: ExecutionSession) -> None:
        if self._sessions.get(session.id) is not session:
            raise ValueError("session does not belong to this executor")

    async def connect(
        self, workspace: WorkspaceBase, user_id: str, conversation_id: str
    ) -> ExecutionSession:
        """Start a container for this conversation."""
        await self._ensure_image()
        directory = workspace.materialize(user_id, conversation_id)
        handle = _Container(f"maxai-runtime-{uuid4().hex}")
        session = ExecutionSession(
            uuid4().hex,
            user_id,
            conversation_id,
            workspace,
            f"/workspaces/{user_id}",
            handle,
        )
        self._sessions[session.id] = session
        self._locks[session.id] = asyncio.Lock()
        try:
            await self._start(session, directory.root)
            return session
        except BaseException:
            await self.clean(session)
            raise

    async def _start(self, session: ExecutionSession, root: Path | None) -> None:
        """Run the container and check its user can write the workspace."""
        h = session.handle
        args = [
            "run",
            "--detach",
            "--pull=never",
            "--name",
            h.name,
            "--user",
            f"{_UID}:{_UID}",
            "--network=none" if self.network == "none" else "--network=bridge",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--init",
            "--pids-limit=128",
            "--memory=512m",
            "--cpus=1",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
            # An anonymous volume: no host path, removed with the container.
            "--mount",
            "type=volume,dst=/workspaces",
            "--workdir",
            "/workspaces",
            "--env",
            f"WORKSPACE={session.workspace_path}",
            await self._ensure_image(),
            "sleep",
            "infinity",
        ]
        h.may_exist = True
        try:
            result = await run_process(
                ["docker", *args], timeout=30, max_output_bytes=self.max_output_bytes
            )
            if result.exit_code != 0:
                raise RuntimeError(result.stderr or "Docker startup failed")
            h.started = True
            h.baseline = {}
            verify = await self._docker_exec(
                session,
                [
                    "sh",
                    "-c",
                    'test "$(id -u)" -ne 0 && mkdir -p "$WORKSPACE" && test -w "$WORKSPACE"',
                ],
                workdir="/workspaces",
            )
            if verify.exit_code != 0:
                raise RuntimeError("Docker image user must be non-root and workspace-writable")
        except BaseException:
            await self._destroy_container(session)
            raise

    async def _docker_exec(
        self,
        session: ExecutionSession,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float = 60,
        workdir: str | None = None,
        max_output_bytes: int | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        """``docker exec`` in the session's container, from the workspace by default."""
        interactive = ["-i"] if stdin is not None else []
        result = await run_process(
            [
                "docker",
                "exec",
                *interactive,
                "--workdir",
                workdir or session.workspace_path,
                session.handle.name,
                *argv,
            ],
            stdin=stdin,
            timeout=timeout,
            max_output_bytes=max_output_bytes or self.max_output_bytes,
            cancellation_token=cancellation_token,
        )
        # The daemon answered instead of the command, or it was killed (137):
        # the container itself may be gone.
        suspect = "Error response from daemon" in result.stderr or "No such container" in result.stderr
        if result.exit_code not in (0, None) and (suspect or result.exit_code == 137):
            if not await self._running(session.handle.name):
                raise SandboxLost(result.stderr.strip().splitlines()[-1] if result.stderr.strip()
                                  else "the container stopped")
        return result

    async def _running(self, name: str) -> bool:
        """``docker inspect`` says the container is up; True when Docker can't
        tell, so a hiccup of the daemon doesn't throw a session away."""
        result = await run_process(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            timeout=15, max_output_bytes=4096,
        )
        if result.exit_code != 0:
            return "No such" not in result.stderr
        return result.stdout.strip() == "true"

    async def execute(
        self,
        session: ExecutionSession,
        command: str,
        *,
        timeout: float,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        """Run ``command`` with bash in the container; see ``supervised``."""
        if not command.strip():
            raise ValueError("command cannot be empty")
        self._check(session)
        run_id = short_id()
        async with self._locks[session.id]:
            if not session.handle.started:
                await self._start(session, None)
            start = time.monotonic()
            try:
                result = await self._docker_exec(
                    session,
                    supervised(command, timeout, run_id),
                    timeout=timeout + 30,
                    cancellation_token=cancellation_token,
                )
            except asyncio.CancelledError:
                # Stopping `docker exec` leaves the command running inside.
                await asyncio.shield(self._docker_exec(session, kill_argv(run_id), timeout=10))
                raise
            if result.timed_out:
                # Not even timeout could stop it: the container goes.
                await self._destroy_container(session)
                return result
        timed_out = result.exit_code != 0 and time.monotonic() - start >= timeout
        return replace(result, timed_out=timed_out)

    async def sync(self, session: ExecutionSession, direction: SyncDirection) -> None:
        """Copy workspace/ and skills/ into the container ("to_environment"), or
        workspace/ back ("to_workspace"); only what changed since the last sync."""
        self._check(session)
        async with self._locks[session.id]:
            h = session.handle
            if not h.started:
                if direction == "to_workspace":
                    return  # nothing ran: the container holds no changes
                await self._start(session, None)
            directory = session.workspace.materialize(session.user_id, session.conversation_id)

            async def run(script: str, args: list[str], *, stdin: str | None = None,
                          limit: int = 1 << 20) -> ExecutionResult:
                return await self._docker_exec(
                    session, ["bash", "-c", script, "maxai", *args],
                    stdin=stdin, timeout=120, workdir="/workspaces", max_output_bytes=limit,
                )

            await sync_session(run, directory, session.workspace_path, h.baseline, direction)

    async def disconnect(self, session: ExecutionSession) -> None:
        self._check(session)

    async def _remove_owned(self, handle: _Container) -> None:
        """``docker rm`` the container and its anonymous workspace volume."""
        if not handle.may_exist:
            return
        result = await run_process(
            ["docker", "rm", "--force", "--volumes", handle.name],
            timeout=15,
            max_output_bytes=self.max_output_bytes,
        )
        if result.exit_code != 0 and "No such container" not in result.stderr:
            raise RuntimeError(result.stderr or "Docker cleanup failed")
        handle.may_exist = False

    async def _destroy_container(self, session: ExecutionSession) -> None:
        """Remove the container even if the caller is cancelled meanwhile."""
        handle = session.handle
        if handle.may_exist:
            cleanup = asyncio.create_task(self._remove_owned(handle))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            handle.started = False

    async def clean(self, session: ExecutionSession) -> None:
        """Remove the container; calling it twice is fine."""
        if self._sessions.get(session.id) is not session:
            if self._closed.get(session.id) is session:
                return
            raise ValueError("session does not belong to this executor")
        await self._destroy_container(session)
        self._sessions.pop(session.id, None)
        self._locks.pop(session.id, None)
        self._closed[session.id] = session


def _tag(name: str, *parts: str) -> str:
    """A local tag that changes whenever the recipe changes."""
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:12]
    return f"{name}:{digest}"
