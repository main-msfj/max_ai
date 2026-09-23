"""Common lifecycle and command contract for execution environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..termination import CancellationToken
from .workspace import WorkspaceBase


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    truncated: bool = False


class Environment(ABC):
    """One environment is bound to one user's conversation.

    Implementations expose the user's workspace, start commands at the workspace
    root, and return separate output streams. Cancelling execute must
    terminate its processes, then propagate CancelledError. Infrastructure
    failures raise exceptions; command failures are ExecutionResult values.
    Command approval belongs to the caller, not to this execution interface.
    """

    def __init__(self, workspace: WorkspaceBase, user_id: str, conversation_id: str):
        self.workspace = workspace
        self.user_id = user_id
        self.conversation_id = conversation_id

    @property
    @abstractmethod
    def variables(self) -> dict[str, str]:
        """WORKSPACE path inside this environment."""

    @abstractmethod
    async def start(self) -> None:
        """Expose workspace and start the environment; safe to call twice."""

    @abstractmethod
    async def execute(
        self, command: str, *, timeout: float = 60,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        """Execute after start; bound runtime and captured output."""

    async def release(self) -> None:
        """Release after a turn; backends may retain resources briefly for reuse."""
        await self.stop()

    @abstractmethod
    async def stop(self) -> None:
        """Terminate owned processes and release resources; preserve files."""

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.stop()
