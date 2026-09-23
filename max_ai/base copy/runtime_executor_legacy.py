"""Execution-provider contract for the new runtime, independent of tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from ..termination import CancellationToken
from .environment import ExecutionResult
from .workspace import Workspace

SyncDirection = Literal["to_environment", "to_workspace"]


@dataclass(frozen=True)
class ExecutionSession:
    """Provider handle plus trusted workspace identity; never model arguments."""

    id: str
    user_id: str
    conversation_id: str
    workspace: Workspace
    workspace_path: str
    handle: Any = field(default=None, repr=False, compare=False)


class Executor(ABC):
    """Implement once per provider: Docker, Modal, E2B or Local.

    Commands start in the user's workspace, including skills and conversations.
    Local implementations do not imply sandbox isolation. Implementations must
    clean partially created resources if connect fails. No method deletes the
    persistent workspace. Provider instances must reject foreign sessions.
    """

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

    @abstractmethod
    async def connect(
        self,
        workspace: Workspace,
        user_id: str,
        conversation_id: str,
    ) -> ExecutionSession:
        """Create/attach a session and mount or expose the user's workspace."""

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
    async def sync(self, session: ExecutionSession, direction: SyncDirection) -> None:
        """Transfer changed files; shared mounts may implement this as a no-op.

        Never silently overwrite concurrent edits or discard unsynced changes.
        A remote provider must define its conflict policy before use.
        """

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
