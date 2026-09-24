"""Local-disk WorkspaceBase implementation."""

from __future__ import annotations

from ....base.workspace import WorkspaceBase
from ._model import LocalWorkspaceConfig


class LocalWorkspace(WorkspaceBase):
    """Local disk is the source of truth — nothing to move."""

    component_provider_override = "max_ai.capabilities.workspace.local.LocalWorkspace"
    component_schema = LocalWorkspaceConfig

    def _to_config(self) -> LocalWorkspaceConfig:
        """Build the serializable configuration for ``LocalWorkspace``."""
        return LocalWorkspaceConfig(root=str(self.base_root))

    async def download(self, user_id: str, conversation_id: str | None = None) -> None:
        """Perform the ``download`` operation for ``LocalWorkspace``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
conversation_id : str | None
    Value supplied for ``conversation_id``."""
        return None

    async def upload(self, user_id: str, conversation_id: str | None = None) -> None:
        """Perform the ``upload`` operation for ``LocalWorkspace``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
conversation_id : str | None
    Value supplied for ``conversation_id``."""
        return None


Workspace = LocalWorkspace
WorkspaceLocal = LocalWorkspace
