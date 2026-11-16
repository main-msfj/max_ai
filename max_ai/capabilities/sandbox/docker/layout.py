"""DockerSandbox: one container per user."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, Field

from ....base.sandbox import CommandResult, SandboxCore, SandboxError
from ....base.session_store import validate_id

HOME = "/home/agent"  # .agent/ goes here, on the user's volume
TIMED_OUT = (124, 137)  # how `timeout` ends a command it had to stop
# Read once from the package, then run in the proxy's container.
PROXY = Path(__file__).with_name("proxy.py").read_text()
PROXY_PORT = 3128


class DockerSandboxConfig(BaseModel):
    image: str = "python:3.12-slim"
    dockerfile: str | None = None
    networks: list[str] = Field(default_factory=list)


class DockerSandbox(SandboxCore):
    """A container named ``docker-sandbox-<identifier>`` with a volume of the
    same name, so skills and files survive the container.

    ``image`` needs bash and python3. With ``dockerfile`` (a path), the image
    is built from it instead, once per content of the file.

    ``networks`` are the domains the container may reach, subdomains
    included (``["pypi.org", "files.pythonhosted.org"]``); empty means no
    network at all. They are reached through a proxy, so they work for what
    honours ``HTTPS_PROXY``: pip, uv, npm, apt, curl, git over https.

    Use it bound: ``sandbox.bind(identifier)``.
    """

    component_schema = DockerSandboxConfig
    isolated = True
    component_provider_override = "max_ai.capabilities.sandbox.docker.DockerSandbox"

    def __init__(
        self,
        image: str = "python:3.12-slim",
        dockerfile: str | None = None,
        networks: list[str] | None = None,
    ) -> None:
        super().__init__(HOME)
        self.image = image
        self.dockerfile = dockerfile
        self.networks = list(networks or [])

    def _to_config(self) -> DockerSandboxConfig:
        return DockerSandboxConfig(
            image=self.image, dockerfile=self.dockerfile, networks=self.networks
        )

    @classmethod
    def _from_config(cls, config: DockerSandboxConfig) -> DockerSandbox:
        return cls(image=config.image, dockerfile=config.dockerfile, networks=config.networks)

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
    ) -> CommandResult:
        """Run ``command`` with bash in the user's container, starting it if
        needed. `timeout` runs inside: stopping `docker exec` from outside
        would leave the command running."""
        await self._start()
        # "-e NAME" takes the value from this call's environment, so the
        # values never show in the process list.
        variables = [arg for name in self.variables for arg in ("-e", name)]
        code, stdout, stderr = await self._docker(
            "exec", "-i", *variables, self.name,
            "timeout", "-k", "2", str(timeout), "bash", "-c", command,
            stdin=stdin, env={**os.environ, **self.variables},
        )
        timed_out = code in TIMED_OUT
        return CommandResult(stdout, stderr, None if timed_out else code, timed_out=timed_out)

    async def close(self) -> None:
        """Remove the user's container, its proxy and their network. The
        volume, with ``.agent/``, stays."""
        await self._docker("rm", "-f", self.name, f"{self.name}-proxy")
        await self._docker("network", "rm", f"{self.name}-net")

    async def _start(self) -> None:
        """Make sure the container runs: start it, or create it the first time."""
        proxy = f"{self.name}-proxy"
        code, _, _ = await self._docker("start", self.name)
        if code == 0:
            if self.networks:
                await self._docker("start", proxy)
            return
        image = await self._image()
        network, settings = "none", []
        if self.networks:
            network = await self._start_proxy(proxy, image)
            # By its alias: a long container name is not a valid host name.
            address = f"http://proxy:{PROXY_PORT}"
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                settings += ["-e", f"{name}={address}"]
        code, _, error = await self._docker(
            "run", "-d", "--init", "--name", self.name, "--network", network, *settings,
            "-v", f"{self.name}:{HOME}", "-w", HOME, image, "sleep", "infinity",
        )
        if code != 0:
            # Another call created it at the same moment: that one is fine too.
            code, _, _ = await self._docker("start", self.name)
            if code != 0:
                raise SandboxError(f"could not start {self.name}: {error.strip()}")

    async def _start_proxy(self, proxy: str, image: str) -> str:
        """Start the proxy with today's ``networks`` and return the network
        the user's container joins: one with no way out except the proxy."""
        network = f"{self.name}-net"
        await self._docker("network", "create", "--internal", network)
        await self._docker("rm", "-f", proxy)
        code, _, error = await self._docker(
            "run", "-d", "--init", "--name", proxy, image, "python3", "-c", PROXY, *self.networks,
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
        if self.dockerfile is None:
            return self.image
        path = Path(self.dockerfile).expanduser().resolve()
        tag = f"docker-sandbox-image:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
        code, _, _ = await self._docker("image", "inspect", tag)
        if code != 0:
            code, _, error = await self._docker(
                "build", "-t", tag, "-f", str(path), str(path.parent)
            )
            if code != 0:
                raise SandboxError(f"could not build {self.dockerfile}: {error.strip()[-500:]}")
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
        stdout, stderr = await process.communicate(stdin)
        # Output is not always text: never fail on it.
        return process.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


__all__ = ["DockerSandbox", "DockerSandboxConfig"]
