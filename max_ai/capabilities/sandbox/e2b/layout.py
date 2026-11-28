"""E2BSandbox: one machine in the E2B cloud per user."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shlex
import typing as t
from pathlib import Path

from pydantic import BaseModel, Field

from ....base.sandbox import CommandResult, SandboxCore
from ....base.session_store import validate_id
from ....config import setting
from ....core.ids import short_id
from ....errors.sandbox import SandboxError

if t.TYPE_CHECKING:
    from e2b import AsyncSandbox

HOME = "/home/agent"  # .agent/ goes here; the folder is root's
TIMED_OUT = (124, 137)  # how `timeout` ends a command it had to stop
KILL = Path(__file__).with_name("kill.sh").read_text()  # stops a cancelled command
AGENT = "agent"  # who the model's commands run as: no sudo
ROOT = "root"  # who the harness's own work runs as: owns setting.json and the skills
OWNER = "max_ai"  # the metadata key that says whose machine it is
SPARE = 15  # seconds E2B waits after our own `timeout` should have ended the command
# Run once, as root, when the machine is created: E2B's own user has sudo.
SETUP = f"id {AGENT} >/dev/null 2>&1 || useradd --no-create-home --shell /bin/bash {AGENT}; mkdir -p {HOME}"


class E2BSandboxConfig(BaseModel):
    template: str | None = None
    networks: list[str] = Field(default_factory=list)
    api_key_env: str = "E2B_API_KEY"
    extra_env: list[str] = Field(default_factory=list)


class E2BSandbox(SandboxCore):
    """A machine in E2B (https://e2b.dev) found by its metadata
    ``max_ai=<identifier>``. Needs the ``e2b`` extra.

    With nothing given, the machine is E2B's base template. Your
    ``template`` needs bash, python3 and useradd; add pypdf to it for
    read_file to read PDFs. Memory and CPU are the template's.

    ``networks`` are the domains the machine may reach, subdomains
    included; empty means no network at all. E2B filters by name only on
    ports 80 and 443. A domain where anyone can write (github.com) is also
    a way out for the user's files. They are set when the machine is
    created: ``remove()`` it to change them.

    The model's commands run as a user without sudo: they write the
    workspace, the memory and tmp, never ``setting.json`` or the skills.

    The machine is paused after ``harness_sandbox_seconds`` without a
    command, and by ``close()``; the next command wakes it with its files.
    E2B keeps a paused machine until ``remove()``.

    ``api_key_env`` is the name of the variable with the E2B key: the key
    is read on this machine and never stored or sent into the sandbox.
    ``extra_env`` names variables of this machine that the commands also see.

    Use it bound: ``sandbox.bind(identifier)``.
    """

    component_schema = E2BSandboxConfig
    component_provider_override = "max_ai.capabilities.sandbox.e2b.E2BSandbox"

    def __init__(
        self,
        template: str | None = None,
        networks: list[str] | None = None,
        api_key_env: str = "E2B_API_KEY",
        extra_env: list[str] | None = None,
    ) -> None:
        super().__init__(HOME, extra_env)
        self.template = template
        self.networks = list(networks or [])
        self.api_key_env = api_key_env
        self._box: AsyncSandbox | None = None  # the user's machine, once found
        self._starting = asyncio.Lock()  # one search or creation at a time

    def _to_config(self) -> E2BSandboxConfig:
        return E2BSandboxConfig(
            template=self.template, networks=self.networks, api_key_env=self.api_key_env,
            extra_env=self.extra_env,
        )

    @classmethod
    def _from_config(cls, config: E2BSandboxConfig) -> E2BSandbox:
        return cls(**config.model_dump())

    def bind(self, identifier: str, variables: dict[str, str] | None = None) -> t.Self:
        """A copy for one user, with its own machine and lock."""
        bound = super().bind(identifier, variables)
        bound._box = None
        bound._starting = asyncio.Lock()
        return bound

    @property
    def name(self) -> str:
        """What the machine's metadata carries: the identifier."""
        if self.identifier is None:
            raise SandboxError("E2BSandbox has no user: call bind(identifier) first")
        return validate_id("identifier", self.identifier)

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float = 60,
        stdin: bytes | None = None,
        harness: bool = False,
    ) -> CommandResult:
        """Run ``command`` with bash in the user's machine, waking or
        creating it if needed. `timeout` runs inside, so nothing is left
        running. The model's commands run without sudo; only ``harness``
        work runs as root."""
        self._ready()
        from e2b import CommandExitException, SandboxException

        box = await self._start()
        user = ROOT if harness else AGENT
        # Its home is a folder it may write: pip and git keep their files there.
        variables = self.environment()
        if not harness:
            variables["HOME"] = str(self.layout.tmp)
        mark = f"run-{short_id()}"  # an extra argument, only to find the command later
        inside = f"timeout -k 2 {timeout} bash -c {shlex.quote(command)} {mark}"
        try:
            # 1. Start it; the bytes of stdin go as they are, then stdin closes.
            running = await box.commands.run(
                inside, background=True, stdin=True, user=user, cwd=HOME,
                envs=variables, timeout=timeout + SPARE,
            )
            try:
                if stdin:
                    await running.send_stdin(stdin)
                await running.close_stdin()
            except SandboxException:
                pass  # it ended without reading its stdin: its exit code tells
            # 2. Wait for its end: a failing command is an exit code, not an error.
            try:
                ended = await running.wait()
                code, stdout, stderr = ended.exit_code, ended.stdout, ended.stderr
            except CommandExitException as failed:
                code, stdout, stderr = failed.exit_code, failed.stdout, failed.stderr
        except asyncio.CancelledError:
            # Stopped from outside: the command would go on inside the machine.
            await asyncio.shield(self._kill(box, user, mark))
            raise
        except SandboxException as error:
            raise SandboxError(f"E2B could not run the command: {error}") from error
        timed_out = code in TIMED_OUT
        return CommandResult(stdout, stderr, None if timed_out else code, timed_out=timed_out)

    async def prepare(self) -> None:
        """Create ``.agent/`` as root; the model's user may write only the
        workspace, the memory, tmp and the tasks."""
        layout = self.layout
        await self._run(
            "create", **layout.skeleton(),
            open=[
                str(layout.workspace), str(layout.tmp), str(layout.memory), str(layout.tasks),
            ],
        )

    async def close(self) -> None:
        """Pause the user's machine: it costs no running time and keeps
        ``.agent/``. A machine that cannot pause now pauses at its timeout."""
        self._ready()
        from e2b import AsyncSandbox, SandboxException, SandboxState

        try:
            found = await self._find([SandboxState.RUNNING])
            if found is not None:
                await AsyncSandbox.pause(found, **self._key())
        except SandboxException:
            pass
        self._box = None

    async def remove(self) -> None:
        """Delete the user's machine and everything in it, ``.agent/`` too."""
        self._ready()
        from e2b import AsyncSandbox, SandboxException, SandboxState

        try:
            found = await self._find([SandboxState.RUNNING, SandboxState.PAUSED])
            if found is not None:
                await AsyncSandbox.kill(found, **self._key())
        except SandboxException as error:
            raise SandboxError(f"E2B could not remove the sandbox of {self.name}: {error}") from error
        self._box = None

    async def _start(self) -> AsyncSandbox:
        """The user's machine, running, with its idle clock started again:
        E2B pauses it after ``harness_sandbox_seconds`` without a command."""
        from e2b import ALL_TRAFFIC, AsyncSandbox, SandboxException, SandboxState

        name, idle = self.name, int(setting.harness_sandbox_seconds)
        async with self._starting:
            try:
                # 1. Known or found: connecting wakes it and moves its timeout.
                found = await self._find([SandboxState.RUNNING, SandboxState.PAUSED])
                if found is not None:
                    try:
                        self._box = await AsyncSandbox.connect(found, timeout=idle, **self._key())
                        return self._box
                    except SandboxException:
                        self._box = None  # removed meanwhile: a new one is created

                # 2. Not there: create it, closed to every domain not listed.
                network: dict[str, t.Any] = {"allow_internet_access": False}
                if self.networks:
                    allowed = [name for domain in self.networks for name in (domain, f"*.{domain}")]
                    network = {"network": {"allow_out": allowed, "deny_out": [ALL_TRAFFIC]}}
                box = await AsyncSandbox.create(
                    template=self.template, timeout=idle, metadata={OWNER: name},
                    lifecycle={"on_timeout": "pause"}, **network, **self._key(),
                )
                # 3. The model's user, before any command of the model.
                made = await box.commands.run(f"{SETUP}; id -u {AGENT}", user=ROOT)
                if not made.stdout.strip().isdigit():
                    await box.kill()
                    raise SandboxError(f"could not create the user {AGENT}: {made.stderr.strip()}")
            except SandboxException as error:
                raise SandboxError(f"E2B could not start the sandbox of {name}: {error}") from error
            self._box = box
            return box

    async def _find(self, states: list[t.Any]) -> str | None:
        """The id of the user's machine: the one in use, or the one the
        account has with this user's metadata. It is not woken."""
        from e2b import AsyncSandbox, SandboxQuery

        if self._box is not None:
            return self._box.sandbox_id
        query = SandboxQuery(metadata={OWNER: self.name}, state=states)
        found = await AsyncSandbox.list(query=query, limit=1, **self._key()).next_items()
        return found[0].sandbox_id if found else None

    def _ready(self) -> None:
        """Say what is missing in words, before anything of E2B is used:
        the user, the key or the ``e2b`` package."""
        _, _ = self.name, self._key()
        if importlib.util.find_spec("e2b") is None:
            raise SandboxError("E2BSandbox needs the e2b package: install max_ai[e2b]")

    def _key(self) -> dict[str, t.Any]:
        """The E2B key, read from this machine's environment when it is used."""
        key = os.environ.get(self.api_key_env)
        if not key:
            raise SandboxError(f"E2BSandbox needs the E2B key in the variable {self.api_key_env}")
        return {"api_key": key}

    @staticmethod
    async def _kill(box: AsyncSandbox, user: str, mark: str) -> None:
        """Stop the command that carries ``mark``, as its own user."""
        from e2b import SandboxException

        try:
            await box.commands.run(f"bash -c {shlex.quote(KILL)} _ {mark}", user=user)
        except SandboxException:
            pass


__all__ = ["E2BSandbox", "E2BSandboxConfig"]
