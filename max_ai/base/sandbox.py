"""The machine where the agent works: one per user.

It owns the project's ``.agent/`` (through Layout): skills go in, the
files the agent creates come out, and commands run there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from ..core.harness.root import Layout
from .component import ComponentBase


@dataclass(frozen=True)
class CommandResult:
    """What a command printed and how it ended."""

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False


class SandboxCore(ComponentBase[BaseModel], Layout, ABC):
    """Implement once per provider: Local, Docker, ACA, Firecracker.

    Each provider defines its own config model and overrides
    ``_to_config`` / ``_from_config``."""

    component_type = "sandbox"

    @abstractmethod
    async def synchronize_environment(self, skill_var: dict[str, str], identifier: str) -> None:
        """Bring the sandbox and ``.agent/`` in line: skills in, the files
        the agent created out.

        ``skill_var`` maps each skill's name to its value, as in
        ``setting.global.json``; ``identifier`` names whose machine it is
        (the user_id)."""

    @abstractmethod
    async def upload_file(self, path: str, content: bytes) -> None:
        """Put ``content`` at ``path`` inside the sandbox."""

    @abstractmethod
    async def get_or_update_skill(self, name: str) -> None:
        """Copy skill ``name`` into the sandbox, or refresh it if it changed."""

    # Not yet: commands come back when bash moves to the sandbox.
    # @abstractmethod
    # async def execute_command(
    #     self,
    #     command: str,
    #     *,
    #     timeout: float = 60,
    #     stdin: bytes | None = None,
    # ) -> CommandResult:
    #     """Run ``command`` with bash in the sandbox. A failing command is an
    #     exit code; only infrastructure problems raise."""
