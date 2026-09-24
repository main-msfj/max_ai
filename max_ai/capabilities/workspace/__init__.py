from ._remote import RemoteWorkspace
from .azure_blob import AzureBlobWorkspace, AzureBlobWorkspaceConfig
from .local import (
    LocalWorkspace,
    LocalWorkspaceConfig,
    Workspace,
    WorkspaceConfig,
    WorkspaceLocal,
    WorkspaceLocalConfig,
)
from .minio import MinIOWorkspace, MinIOWorkspaceConfig

__all__ = [
    "AzureBlobWorkspace",
    "AzureBlobWorkspaceConfig",
    "LocalWorkspace",
    "LocalWorkspaceConfig",
    "MinIOWorkspace",
    "MinIOWorkspaceConfig",
    "RemoteWorkspace",
    "Workspace",
    "WorkspaceConfig",
    "WorkspaceLocal",
    "WorkspaceLocalConfig",
]
