"""Execution + session-lifecycle contract, one implementation per provider.

Pluggable per provider (Local, Docker, Modal) — that's why this lives in
base/, not core/: implementations vary, the contract doesn't.
``core/environment/environment_manager.py`` is the one fixed orchestrator
that decides *when* to call these methods (cache, lock, idle expiry); it
never varies per provider, which is why it lives in core/ instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from ..core.termination import CancellationToken
from .component import ComponentBase
from .workspace import WorkspaceBase

SyncDirection = Literal["to_environment", "to_workspace"]


@dataclass(frozen=True)
class ExecutionResult:
    """
    Describe the result of running a command in an execution session.
    """

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class ExecutionSession:
    """Provider handle plus trusted workspace identity; never model arguments."""

    id: str
    user_id: str
    conversation_id: str
    workspace: WorkspaceBase
    workspace_path: str
    handle: Any = field(default=None, repr=False, compare=False)


class ExecutorBase(ComponentBase[BaseModel], ABC):
    """Implement once per provider: Local, Docker, Modal.

    Commands start in the user's workspace, including skills and conversations.
    Local implementations do not imply sandbox isolation. Implementations must
    clean partially created resources if connect fails. No method deletes the
    persistent workspace. Provider instances must reject foreign sessions.
    Each concrete provider defines its own config model (like
    LocalSkillRegistryConfig) and overrides _to_config/_from_config.
    """

    component_type = "executor"
    # Commands run away from the host (a container, a VM). When they don't,
    # the user approves every command.
    isolated: ClassVar[bool] = False

    @property
    def runs_commands(self) -> bool:
        """Whether the Agent gives the model the bash tool."""
        return True

    async def prepare(self) -> None:
        """Slow one-time setup (building an image), run by the Agent before
        any tool so it never counts against a tool's timeout."""

    def describe_environment(self) -> str | None:
        """What the model should know about where its commands run (network,
        preinstalled packages), or ``None``. The Agent reads it once, when it
        is built, and puts it in the system prompt."""
        return None

    # -------- RUN CODE -----------------------------------------------------------
    @abstractmethod
    async def execute(
        self,
        session: ExecutionSession,
        command: str,
        *,
        timeout: float = 60,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        """Run ``command`` with bash and return its output and exit code.

        This is the only way a command reaches the environment: the harness
        builds the script, the provider only runs it. Timeout and cancellation
        must stop every process the command started. A failing command is an
        exit code; only infrastructure problems raise.
        """

    # -------- SESSION LIFECYCLE -----------------------------------------------------------
    @abstractmethod
    async def connect(
        self,
        workspace: WorkspaceBase,
        user_id: str,
        conversation_id: str,
    ) -> ExecutionSession:
        """Create/attach a session and mount or expose the user's workspace."""

    async def sync(self, session: ExecutionSession, direction: SyncDirection) -> None:
        """Transfer changed files; shared mounts/no remote copy: no-op by default.

        Never silently overwrite concurrent edits or discard unsynced changes.
        A remote provider (e.g. Modal) must override this and define its
        conflict policy before use.
        """
        return None

    @abstractmethod
    async def disconnect(self, session: ExecutionSession) -> None:
        """Release connections/resources after sync, preserving workspace files."""

    @abstractmethod
    async def clean(self, session: ExecutionSession) -> None:
        """Idempotently destroy temporary runtime resources, not workspace files."""
