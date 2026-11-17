"""LocalSandbox: this computer is the sandbox."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from pydantic import BaseModel

from ....base.sandbox import CommandResult, SandboxCore


class LocalSandboxConfig(BaseModel):
    root: str = "."


class LocalSandbox(SandboxCore):
    """This computer is the sandbox, without isolation. ``root`` is where
    ``.agent/`` goes; ``.`` is the folder the program runs in."""

    component_schema = LocalSandboxConfig
    component_provider_override = "max_ai.capabilities.sandbox.local.LocalSandbox"

    def __init__(self, root: str = ".") -> None:
        self.project = root
        super().__init__(str(Path(root).expanduser().resolve()))

    def _to_config(self) -> LocalSandboxConfig:
        return LocalSandboxConfig(root=self.project)

    @classmethod
    def _from_config(cls, config: LocalSandboxConfig) -> LocalSandbox:
        return cls(root=config.root)

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float = 60,
        stdin: bytes | None = None,
    ) -> CommandResult:
        """Run ``command`` with bash on this computer, starting in ``root``.
        It sees this process's environment plus ``self.variables``."""
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            cwd=self.layout.root.parent,
            env={**os.environ, **self.variables},
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
