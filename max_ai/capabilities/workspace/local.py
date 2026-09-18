"""Local-disk WorkspaceBase implementation."""

from __future__ import annotations

from ...base.workspace import WorkspaceBase, WorkspaceConfig


class LocalWorkspace(WorkspaceBase):
    """Local disk is the source of truth — nothing to move."""

    async def download(self, user_id: str, conversation_id: str | None = None) -> None:
        return None

    async def upload(self, user_id: str, conversation_id: str | None = None) -> None:
        return None


# Compat aliases for older call sites.
Workspace = LocalWorkspace
WorkspaceLocal = LocalWorkspace
WorkspaceLocalConfig = WorkspaceConfig

__all__ = [
    "LocalWorkspace",
    "Workspace",
    "WorkspaceLocal",
    "WorkspaceConfig",
    "WorkspaceLocalConfig",
]
