"""Docker runtime executor using the Docker CLI."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ....base.executor import ExecutionSession
from ....core.executor.process import run_process
from ....core.executor.remote import DEFAULT_FRAMEWORK, RemoteExecutor
from ._model import DockerExecutorConfig

_RUNTIME_DOCKERFILE = Path(__file__).with_name("Dockerfile")
_UID = 1000
# Added on top of a developer's image: max_ai, the agent user, /workspaces.
_FRAMEWORK_LAYER = """FROM {base}
USER root
RUN python3 -m venv /opt/maxai && /opt/maxai/bin/pip install --no-cache-dir "{framework}" \\
 && (id -u {uid} >/dev/null 2>&1 || useradd --uid {uid} --create-home agent) \\
 && mkdir -p /workspaces && chown {uid}:{uid} /workspaces && chmod 755 /root
ENV HOME=/tmp
WORKDIR /workspaces
"""


@dataclass
class _Container:
    """_Container represents structured data used by the capability system."""
    name: str
    started: bool = False
    may_exist: bool = False


class DockerExecutor(RemoteExecutor):
    """Runs commands in a local Docker container.

    Network is ``"none"`` (default) or ``"internet"``. Docker cannot filter
    by domain, so ``"packages"`` and ``allow_list`` are not supported here;
    use ModalExecutor for that.

    The image is built on first use and cached by Docker: max_ai's runtime
    Dockerfile by default, or your ``image``/``dockerfile`` with max_ai, the
    agent user and /workspaces added on top, so you never install max_ai.
    """

    supports_allow_list = False
    component_schema = DockerExecutorConfig
    component_provider_override = "max_ai.capabilities.executor.docker.DockerExecutor"

    def __init__(self, *, image: str | None = None, dockerfile: str | None = None,
                 framework: str = DEFAULT_FRAMEWORK, network: str = "none",
                 max_output_bytes: int = 1 << 20):
        """Initialize ``DockerExecutor``.

Parameters
----------
image : str | None
    Your image (``"python:3.12-slim"``). It only needs Linux, bash, git and
    python3 with pip and venv.
dockerfile : str | None
    Path to your own Dockerfile, built with its folder as context.
framework : str
    pip requirement that installs max_ai inside the container.
network : str
    "none" (default) or "internet". ``allow_list`` does not apply to Docker.
max_output_bytes : int
    Value supplied for ``max_output_bytes``."""
        self._init_network(network, None)
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if image is not None and dockerfile is not None:
            raise ValueError("Pass image or dockerfile, not both")
        self.image, self.dockerfile, self.framework = image, dockerfile, framework
        self.max_output_bytes = max_output_bytes
        self._runtime_image: str | None = None
        self._image_lock = asyncio.Lock()
        super().__init__()
        self._sessions = {}
        self._locks = {}
        self._closed = {}

    def describe_environment(self) -> str:
        """Describe the environment exposed by ``DockerExecutor``."""
        ours = self.image is None and self.dockerfile is None
        text = ("Commands run in an isolated Docker container as a non-root user. "
                + self._network_description("pip, uv or npm" if ours else "pip")
                + " Only the workspace and /tmp are writable.")
        if ours:
            text += " Python 3.11, uv, Node.js/npm, git and ripgrep are available."
        return text

    async def _ensure_image(self) -> str:
        """Build the runtime image once (Docker caches it by tag) and return its tag."""
        async with self._image_lock:
            if self._runtime_image is None:
                self._runtime_image = await self._build_image()
            return self._runtime_image

    async def _build_image(self) -> str:
        """Perform the internal ``build image`` operation for ``DockerExecutor``."""
        if self.image is None and self.dockerfile is None:
            tag = _tag("maxai-runtime", _RUNTIME_DOCKERFILE.read_text(), self.framework)
            await self._docker_build(tag, ["--build-arg", f"MAXAI={self.framework}", "-"],
                                     stdin=_RUNTIME_DOCKERFILE.read_text())
            return tag
        base = self.image
        if self.dockerfile is not None:
            path = Path(self.dockerfile).resolve()
            base = _tag("maxai-base", path.read_text())
            await self._docker_build(base, ["-f", str(path), str(path.parent)])
        layer = _FRAMEWORK_LAYER.format(base=base, framework=self.framework, uid=_UID)
        tag = _tag("maxai-runtime", layer)
        await self._docker_build(tag, ["-"], stdin=layer)
        return tag

    async def _docker_build(self, tag: str, args: list[str], stdin: str | None = None) -> None:
        """``docker build`` unless ``tag`` already exists locally."""
        exists = await run_process(["docker", "image", "inspect", tag], timeout=30)
        if exists.exit_code == 0:
            return
        result = await run_process(["docker", "build", "-t", tag, *args], stdin=stdin,
                                   timeout=3600, max_output_bytes=self.max_output_bytes)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Building the Docker image {tag} failed. Your image needs Linux, bash, "
                f"git and python3 with pip and venv.\n{result.stderr[-2000:]}"
            )

    def _to_config(self) -> DockerExecutorConfig:
        """Build the serializable configuration for ``DockerExecutor``."""
        return DockerExecutorConfig(
            image=self.image, dockerfile=self.dockerfile, framework=self.framework,
            network=self.network,
            max_output_bytes=self.max_output_bytes,
        )

    @classmethod
    def _from_config(cls, config: DockerExecutorConfig) -> "DockerExecutor":
        """Create an instance from its configuration for ``DockerExecutor``.

Parameters
----------
config : DockerExecutorConfig
    Value supplied for ``config``."""
        return cls(**config.model_dump())

    def _check(self, session):
        """Perform the internal ``check`` operation for ``DockerExecutor``.

Parameters
----------
session
    Value supplied for ``session``."""
        if self._sessions.get(session.id) is not session:
            raise ValueError("session does not belong to this executor")

    async def connect(self, workspace, user_id, conversation_id):
        """Open required resources for ``DockerExecutor``.

Parameters
----------
workspace
    Value supplied for ``workspace``.
user_id
    Value supplied for ``user_id``.
conversation_id
    Value supplied for ``conversation_id``."""
        await self._ensure_image()
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
        """Perform the internal ``start`` operation for ``DockerExecutor``.

Parameters
----------
session
    Value supplied for ``session``.
root
    Value supplied for ``root``."""
        h = session.handle
        if "," in str(root):
            raise ValueError("Docker workspace paths cannot contain commas")
        args = ["run", "--detach", "--pull=never", "--name", h.name, "--user", f"{_UID}:{_UID}",
                "--network=none" if self.network == "none" else "--network=bridge",
                "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--init", "--pids-limit=128", "--memory=512m", "--cpus=1",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m", "--mount",
                f"type=bind,src={root},dst={session.workspace_path}",
                "--workdir", session.workspace_path, "--env", f"WORKSPACE={session.workspace_path}",
                await self._ensure_image(), "sleep", "infinity"]
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
        """Perform the internal ``docker exec`` operation for ``DockerExecutor``.

Parameters
----------
session
    Value supplied for ``session``.
argv
    Value supplied for ``argv``.
kwargs
    Value supplied for ``kwargs``."""
        stdin = kwargs.get("stdin")
        interactive = ["-i"] if stdin is not None else []
        return await run_process(["docker", "exec", *interactive, "--workdir", session.workspace_path,
                                  session.handle.name, *argv], max_output_bytes=self.max_output_bytes, **kwargs)

    async def execute_argv(self, session, argv, *, stdin=None, timeout=60, cancellation_token=None):
        """Run an argument vector for ``DockerExecutor``.

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
        """Execute the requested operation for ``DockerExecutor``.

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
            raise ValueError("command cannot be empty")
        return await self.execute_argv(session, ["bash", "--noprofile", "--norc", "-c", command],
                                       timeout=timeout, cancellation_token=cancellation_token)

    async def sync(self, session, direction):
        """Perform the ``sync`` operation for ``DockerExecutor``.

Parameters
----------
session
    Value supplied for ``session``.
direction
    Value supplied for ``direction``."""
        self._check(session)

    async def disconnect(self, session):
        """Release resources held for ``DockerExecutor``.

Parameters
----------
session
    Value supplied for ``session``."""
        self._check(session)

    async def _remove_owned(self, handle):
        """Perform the internal ``remove owned`` operation for ``DockerExecutor``.

Parameters
----------
handle
    Value supplied for ``handle``."""
        if not handle.may_exist:
            return
        result = await run_process(["docker", "rm", "--force", handle.name], timeout=15,
                                   max_output_bytes=self.max_output_bytes)
        if result.exit_code != 0 and "No such container" not in result.stderr:
            raise RuntimeError(result.stderr or "Docker cleanup failed")
        handle.may_exist = False

    async def _destroy_container(self, session):
        """Perform the internal ``destroy container`` operation for ``DockerExecutor``.

Parameters
----------
session
    Value supplied for ``session``."""
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
        """Remove temporary resources owned for ``DockerExecutor``.

Parameters
----------
session
    Value supplied for ``session``."""
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
        """Recreate the runtime environment for ``DockerExecutor``.

Parameters
----------
session
    Value supplied for ``session``."""
        self._check(session)
        workspace = session.workspace
        user_id = session.user_id
        conversation_id = session.conversation_id
        await self.clean(session)
        return await self.connect(workspace, user_id, conversation_id)


def _tag(name: str, *parts: str) -> str:
    """A local tag that changes whenever the recipe changes."""
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:12]
    return f"{name}:{digest}"
