"""The native workspace shared by every agent for a given user."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..config import setting
from ..types.workspace import WorkspaceDirectory
from .capability import CoreAgentCapabilities


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: str | None = None


class Workspace(CoreAgentCapabilities[WorkspaceConfig]):
    """Own ``.agents/<user>/skills`` and ``.agents/<user>/<conversation>``.

    User and conversation identifiers come from the run context. Files remain
    on disk between runs; agents using the same root share the same workspace.
    """

    component_schema = WorkspaceConfig
    component_type = "workspace"

    def __init__(self, root: str | Path | None = None) -> None:
        super().__init__()
        self.base_root = Path(root if root is not None else setting.root_dir / ".agents").expanduser().resolve()
        self.base_root.mkdir(parents=True, exist_ok=True)
        self._filesystem = None

    def _to_config(self) -> WorkspaceConfig:
        return WorkspaceConfig(root=str(self.base_root))

    @classmethod
    def _from_config(cls, config: WorkspaceConfig) -> Workspace:
        return cls(root=config.root)

    def get_filesystem(self):
        from ..workspace.filesystem import UserFileSystem

        if self._filesystem is None:
            self._filesystem = UserFileSystem(self.base_root)
        return self._filesystem

    def materialize(self, user_id: str, conversation_id: str | None = None) -> WorkspaceDirectory:
        """Create a user's directories without clearing existing files."""
        filesystem = self.get_filesystem()
        filesystem._safe_id(user_id, "user_id")
        if conversation_id is not None:
            filesystem._safe_session_id(conversation_id)
        root = filesystem.user_root(user_id)
        user_fd = filesystem._open_user_fd(user_id)
        try:
            skills_fd = filesystem._open_dir_at(user_fd, "skills", create=True)
            os.close(skills_fd)
        finally:
            os.close(user_fd)
        conversation = (
            filesystem.conversation_root(user_id, conversation_id)
            if conversation_id is not None else None
        )
        scratch = (
            filesystem.scratchpad_root(user_id, conversation_id)
            if conversation_id is not None else None
        )
        return WorkspaceDirectory(
            root=root,
            skill_dir=root / "skills",
            conversation_dir=conversation,
            scratch_dir=scratch,
            artifacts_dir=conversation or root / "artifacts",
        )


# Import compatibility only: there is one concrete workspace implementation.
WorkSpaceRegistry = Workspace
