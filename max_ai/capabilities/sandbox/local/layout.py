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
    """This computer is the sandbox. ``root`` is where ``.agent/`` goes;
    ``.`` is the folder the program runs in."""

    component_schema = LocalSandboxConfig
    component_provider_override = "max_ai.capabilities.sandbox.local.LocalSandbox"

    def __init__(self, root: str = ".") -> None:
        super().__init__(project=Path(root))
        self.create_project()

    def _to_config(self) -> LocalSandboxConfig:
        return LocalSandboxConfig(root=str(self.root.parent))

    @classmethod
    def _from_config(cls, config: LocalSandboxConfig) -> LocalSandbox:
        return cls(root=config.root)

    async def synchronize_environment(self, skill_var: dict[str, str], identifier: str) -> None:
        raise NotImplementedError

    async def upload_file(self, path: str, content: bytes) -> None:
        raise NotImplementedError

    async def get_or_update_skill(self, name: str) -> None:
        raise NotImplementedError

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float = 60,
        stdin: bytes | None = None,
    ) -> CommandResult:
        """Run ``command`` with bash on this computer, starting in the workspace."""
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            cwd=self.workspace,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # its own group, so a timeout kills every child
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout)
        except asyncio.TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = await process.communicate()
            return CommandResult(stdout.decode(), stderr.decode(), None, timed_out=True)
        return CommandResult(stdout.decode(), stderr.decode(), process.returncode)

