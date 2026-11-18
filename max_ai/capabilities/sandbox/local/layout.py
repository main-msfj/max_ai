"""LocalSandbox: this computer is the sandbox."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from pydantic import BaseModel, Field

from ....base.sandbox import CommandResult, SandboxCore

# What bash and its programs need from this machine; nothing else passes.
SYSTEM = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TERM", "TZ", "TMPDIR")


class LocalSandboxConfig(BaseModel):
    root: str = "."
    extra_env: list[str] = Field(default_factory=list)


class LocalSandbox(SandboxCore):
    """This computer is the sandbox, without isolation. ``root`` is where
    ``.agent/`` goes; ``.`` is the folder the program runs in.

    A command sees the system's variables (PATH, HOME, the locale), the ones
    named in ``extra_env`` and the user's; never the rest of this process's
    environment, where the API keys are. It still runs as you: it can read
    any file you can, ``.env`` included. For content you do not trust, use
    DockerSandbox or E2BSandbox.
    """

    component_schema = LocalSandboxConfig
    component_provider_override = "max_ai.capabilities.sandbox.local.LocalSandbox"

    def __init__(self, root: str = ".", extra_env: list[str] | None = None) -> None:
        self.project = root
        super().__init__(str(Path(root).expanduser().resolve()), extra_env)

    def _to_config(self) -> LocalSandboxConfig:
        return LocalSandboxConfig(root=self.project, extra_env=self.extra_env)

    @classmethod
    def _from_config(cls, config: LocalSandboxConfig) -> LocalSandbox:
        return cls(root=config.root, extra_env=config.extra_env)

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float = 60,
        stdin: bytes | None = None,
        harness: bool = False,
    ) -> CommandResult:
        """Run ``command`` with bash on this computer, starting in ``root``.
        Here the harness and the model are the same user: ``harness``
        changes nothing."""
        # 1. Only the system's variables of this machine, never its secrets.
        system = {
            name: value for name, value in os.environ.items()
            if name in SYSTEM or name.startswith("LC_")
        }
        # 2. git never takes a repository above the session's folder: the
        #    project's own would get the model's commits.
        ceiling = f"{self.layout.workspace}{os.pathsep}{self.layout.tmp}"
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            cwd=self.layout.root.parent,
            env={**system, **self.environment(), "GIT_CEILING_DIRECTORIES": ceiling},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # its own group, so a timeout kills every child
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout)
        except asyncio.TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = await process.communicate()
            timed_out = True
        except asyncio.CancelledError:
            os.killpg(process.pid, signal.SIGKILL)  # stopped from outside: nothing stays running
            raise
        # Output is not always text: never fail on it.
        return CommandResult(
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            None if timed_out else process.returncode,
            timed_out=timed_out,
        )
