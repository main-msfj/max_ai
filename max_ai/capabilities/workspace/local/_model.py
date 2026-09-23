"""Serializable configuration for the local workspace."""

from ....base.workspace import WorkspaceConfig


class LocalWorkspaceConfig(WorkspaceConfig):
    """Local filesystem root inherited from the workspace contract."""


WorkspaceLocalConfig = LocalWorkspaceConfig
