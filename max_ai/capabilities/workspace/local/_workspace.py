"""Local-disk WorkspaceBase implementation."""

from __future__ import annotations

from ....base.workspace import WorkspaceBase
from ._model import LocalWorkspaceConfig


class LocalWorkspace(WorkspaceBase):
    """Local disk is the source of truth — nothing to move."""

    component_provider_override = "max_ai.capabilities.workspace.local.LocalWorkspace"
    component_schema = LocalWorkspaceConfig

    def _to_config(self) -> LocalWorkspaceConfig:
        return LocalWorkspaceConfig(root=str(self.base_root))

    async def download(self, user_id: str, conversation_id: str | None = None) -> None:
        return None

    async def upload(self, user_id: str, conversation_id: str | None = None) -> None:
        return None


Workspace = LocalWorkspace
WorkspaceLocal = LocalWorkspace
