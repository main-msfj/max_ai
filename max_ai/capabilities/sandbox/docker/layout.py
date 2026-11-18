"""DockerSandbox: one container per user."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, Field

from ....base.sandbox import CommandResult, SandboxCore
from ....base.session_store import validate_id
from ....config import setting
from ....core.ids import short_id
from ....errors.sandbox import SandboxError

HOME = "/home/agent"  # .agent/ goes here, on the user's volume
TIMED_OUT = (124, 137)  # how `timeout` ends a command it had to stop
# Read once from the package, then run in the proxy's container.
PROXY = Path(__file__).with_name("proxy.py").read_text()
KILL = Path(__file__).with_name("kill.sh").read_text()  # stops a cancelled command
IDLE = Path(__file__).with_name("idle.sh").read_text()  # the container's main process
AGENT = "1000:1000"  # who the model's commands run as: no privileges
ROOT = "0:0"  # who the harness's own work runs as: owns setting.json and the skills
ALIVE = "USR1"  # sent before each command: the idle clock starts again
# Nothing in the container has more power than its user has on its own files.
LOCKED = ("--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only")
PROXY_PORT = 3128
DOCKERFILE = str(Path(__file__).with_name("Dockerfile"))  # ours: python + pypdf


class DockerSandboxConfig(BaseModel):
    image: str | None = None
    dockerfile: str | None = None
    networks: list[str] = Field(default_factory=list)
    memory: str = "2g"
    cpus: float = 2
    processes: int = 512
    extra_env: list[str] = Field(default_factory=list)


class DockerSandbox(SandboxCore):
    """A container named ``docker-sandbox-<identifier>`` with a volume of the
    same name, so skills and files survive the container.

    With nothing given, the image is built from our ``Dockerfile`` (python and
    pypdf, which read_file uses for PDFs). Your ``image`` needs bash and
    python3. With ``dockerfile`` (a path), the image is built from it
    instead, once per content of the file.

    ``networks`` are the domains the container may reach, subdomains
    included (``["pypi.org", "files.pythonhosted.org"]``); empty means no
    network at all. They are reached through a proxy, so they work for what
    honours ``HTTPS_PROXY``: pip, uv, npm, curl, git over https. A domain
    where anyone can write (github.com) is also a way out for the user's files.

    The model's commands run as a user without privileges: they write the
    workspace, the memory and tmp, never ``setting.json`` or the skills,
    and nothing outside the volume (``pip install`` goes to the user's
    folder; system packages must come in the image). ``memory``, ``cpus``
    and ``processes`` are the most a container may use.

    The container stops by itself after ``harness_sandbox_seconds`` without
    a command and is created again by the next one; the volume stays
    until ``remove()``.

    ``extra_env`` names variables of this machine that the commands also
    see; nothing else of this machine's environment enters the container.

    Use it bound: ``sandbox.bind(identifier)``.
    """

    component_schema = DockerSandboxConfig
    isolated = True
    component_provider_override = "max_ai.capabilities.sandbox.docker.DockerSandbox"

    def __init__(
        self,
        image: str | None = None,
        dockerfile: str | None = None,
        networks: list[str] | None = None,
        memory: str = "2g",
        cpus: float = 2,
        processes: int = 512,
        extra_env: list[str] | None = None,
    ) -> None:
        super().__init__(HOME, extra_env)
        self.image = image
        self.dockerfile = dockerfile
        self.networks = list(networks or [])
        self.memory = memory
        self.cpus = cpus
        self.processes = processes

    def _to_config(self) -> DockerSandboxConfig:
        return DockerSandboxConfig(
            image=self.image, dockerfile=self.dockerfile, networks=self.networks,
            memory=self.memory, cpus=self.cpus, processes=self.processes,
            extra_env=self.extra_env,
        )

    @classmethod
    def _from_config(cls, config: DockerSandboxConfig) -> DockerSandbox:
        return cls(**config.model_dump())

    @property
    def name(self) -> str:
        """The container and its volume: ``docker-sandbox-<identifier>``."""
        if self.identifier is None:
            raise SandboxError("DockerSandbox has no user: call bind(identifier) first")
        return f"docker-sandbox-{validate_id('identifier', self.identifier)}"

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float = 60,
        stdin: bytes | None = None,
        harness: bool = False,
    ) -> CommandResult:
        """Run ``command`` with bash in the user's container, starting it if
        needed. `timeout` runs inside: stopping `docker exec` from outside
        would leave the command running. The model's commands run without
        privileges; only ``harness`` work runs as the owner of the settings."""
        await self._start()
        # "-e NAME" takes the value from this call's environment, so the
        # values never show in the process list.
        shown = self.environment()
        variables = [arg for name in shown for arg in ("-e", name)]
        # Its home is a folder it may write: pip and git keep their files there.
        user = ["--user", ROOT] if harness else ["--user", AGENT, "-e", f"HOME={self.layout.tmp}"]
        mark = f"run-{short_id()}"  # an extra argument, only to find the command later
        try:
            code, stdout, stderr = await self._docker(
                "exec", "-i", *user, *variables, self.name,
                "timeout", "-k", "2", str(timeout), "bash", "-c", command, mark,
                stdin=stdin, env={**os.environ, **shown},
            )
        except asyncio.CancelledError:
            # Stopped from outside: the command would go on inside the container.
            # Killed by its own user: nobody here may kill another user's process.
            await asyncio.shield(
                self._docker("exec", *user[:2], self.name, "bash", "-c", KILL, "_", mark)
            )
            raise
        timed_out = code in TIMED_OUT
        return CommandResult(stdout, stderr, None if timed_out else code, timed_out=timed_out)

    async def prepare(self) -> None:
        """Create ``.agent/`` as the harness's user; the model's user may
        write only the workspace, the memory and tmp."""
        layout = self.layout
        # 1. The volume must be root's: else the model could swap ``.agent/`` whole.
        owner = await self.execute_command(f"stat -c %u {HOME}", harness=True)
        if owner.stdout.strip() != "0":
            raise SandboxError(
                f"{HOME} belongs to user {owner.stdout.strip() or '?'} in this image; it must "
                "not exist in the image, so the volume mounted there belongs to root"
            )
        # 2. The folders, and the three the model's user writes.
        await self._run(
            "create", **layout.skeleton(),
            open=[str(layout.workspace), str(layout.tmp), str(layout.memory)],
        )

    async def close(self) -> None:
        """Remove the user's container, its proxy and their network. The
        volume, with ``.agent/``, stays."""
        await self._docker("rm", "-f", self.name, f"{self.name}-proxy")
        await self._docker("network", "rm", f"{self.name}-net")

    async def remove(self) -> None:
        """Delete everything of the user: the container, its proxy, their
        network and the volume, with ``.agent/`` in it."""
        await self.close()
        code, _, error = await self._docker("volume", "rm", "-f", self.name)
        if code != 0:
            raise SandboxError(f"could not remove the volume of {self.name}: {error.strip()}")

    async def _start(self) -> None:
        """Make sure the container runs and tell it a command comes: it stops
        by itself after ``harness_sandbox_seconds`` without one."""
        proxy = f"{self.name}-proxy"
        # 1. Running: the signal starts its idle clock again.
        if await self._alive(self.name):
            if self.networks and not await self._alive(proxy):
                await self._start_proxy(proxy, await self._image())
            return
        # 2. Not running: create it, with its proxy when it has networks.
        image = await self._image()
        network = "none"
        settings: list[str] = []
        if self.networks:
            network = await self._start_proxy(proxy, image)
            # By its alias: a long container name is not a valid host name.
            address = f"http://proxy:{PROXY_PORT}"
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                settings += ["-e", f"{name}={address}"]
        created = [
            "run", "-d", "--rm", "--init", "--name", self.name, "--network", network, *settings,
            *self._limits(self.memory, self.cpus, self.processes),
            # /tmp is the only place outside the volume where anything is written.
            "--tmpfs", "/tmp:exec,mode=1777",
            "-v", f"{self.name}:{HOME}", "-w", HOME, image,
            "sh", "-c", IDLE, "_", str(int(setting.harness_sandbox_seconds)),
        ]
        code, _, error = await self._docker(*created)
        if code != 0 and not await self._alive(self.name):
            # Not made by another call at the same moment: one that stopped
            # holds the name. It goes, and this one is created.
            await self._docker("rm", "-f", self.name)
            code, _, error = await self._docker(*created)
            if code != 0:
                raise SandboxError(f"could not start {self.name}: {error.strip()}")

    async def _alive(self, container: str) -> bool:
        """Whether ``container`` runs; asking also starts its idle clock again."""
        code, _, _ = await self._docker("kill", "--signal", ALIVE, container)
        return code == 0

    def _limits(self, memory: str, cpus: float, processes: int) -> list[str]:
        """What every container of ours is created with: no privileges, a
        root file system nobody writes, and the most it may use."""
        return [
            "--user", AGENT, *LOCKED, "--label", f"max_ai.sandbox={self.identifier}",
            "--memory", memory, "--cpus", str(cpus), "--pids-limit", str(processes),
        ]

    async def _start_proxy(self, proxy: str, image: str) -> str:
        """Start the proxy with today's ``networks`` and return the network
        the user's container joins: one with no way out except the proxy."""
        network = f"{self.name}-net"
        await self._docker("network", "create", "--internal", network)
        await self._docker("rm", "-f", proxy)
        code, _, error = await self._docker(
            "run", "-d", "--rm", "--init", "--name", proxy, *self._limits("256m", 1, 128),
            image, "python3", "-c", PROXY, str(int(setting.harness_sandbox_seconds)), *self.networks,
        )
        if code == 0:
            code, _, error = await self._docker("network", "connect", "--alias", "proxy", network, proxy)
        if code != 0:
            raise SandboxError(f"could not start the proxy of {self.name}: {error.strip()}")
        # The first command must not arrive before the proxy listens.
        listening = f"import socket; socket.create_connection(('localhost', {PROXY_PORT}), 1)"
        for _ in range(50):
            code, _, _ = await self._docker("exec", proxy, "python3", "-c", listening)
            if code == 0:
                break
            await asyncio.sleep(0.1)
        return network

    async def _image(self) -> str:
        """The image to run: ``image``, or the one built from ``dockerfile``.
        Its tag is the hash of the file, so it is rebuilt only when it changes."""
        # 1. Their image is used as it is; with nothing given, ours is built.
        if self.dockerfile is None and self.image is not None:
            return self.image
        path = Path(self.dockerfile or DOCKERFILE).expanduser().resolve()

        # 2. Build only when no image has this file's hash as its tag.
        tag = f"docker-sandbox-image:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
        code, _, _ = await self._docker("image", "inspect", tag)
        if code != 0:
            code, _, error = await self._docker(
                "build", "-t", tag, "-f", str(path), str(path.parent)
            )
            if code != 0:
                raise SandboxError(f"could not build {path}: {error.strip()[-500:]}")
        return tag

    @staticmethod
    async def _docker(
        *args: str, stdin: bytes | None = None, env: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        """Run the ``docker`` command: its exit code and what it printed."""
        process = await asyncio.create_subprocess_exec(
            "docker", *args, env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate(stdin)
        except asyncio.CancelledError:
            process.kill()  # do not leave the docker command waiting
            raise
        # Output is not always text: never fail on it.
        return process.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


__all__ = ["DockerSandbox", "DockerSandboxConfig"]
