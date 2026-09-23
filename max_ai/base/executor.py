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
from typing import Any, Literal

from pydantic import BaseModel

from ..core.termination import CancellationToken
from .component import ComponentBase
from .workspace import WorkspaceBase

SyncDirection = Literal["to_environment", "to_workspace"]


@dataclass(frozen=True)
class ExecutionResult:
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

    # -------- RUN CODE -----------------------------------------------------------
    @abstractmethod
    async def run_tool(
        self,
        session: ExecutionSession,
        tool,
        record,
        context,
        cancellation_token: CancellationToken | None = None,
    ):
        """Invoke a native CoreTool in this provider; never fall back to host."""

    @abstractmethod
    async def execute(
        self,
        session: ExecutionSession,
        command: str,
        *,
        timeout: float = 60,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        """Run Bash; timeout/cancellation must terminate its owned processes.

        Return command failures as exit codes. Raise infrastructure errors.
        Cancellation propagates CancelledError after process cleanup.
        """

    @abstractmethod
    async def execute_argv(
        self,
        session: ExecutionSession,
        argv: list[str],
        *,
        stdin: str | None = None,
        timeout: float = 60,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        """Execute an argument vector without shell interpolation, with bounded output."""

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

    @abstractmethod
    async def rebuild(self, session: ExecutionSession) -> ExecutionSession:
        """Recreate the runtime with the same workspace identity, not its image.

        On failure, retain a handle that clean(old_session) can safely clean.
        """
