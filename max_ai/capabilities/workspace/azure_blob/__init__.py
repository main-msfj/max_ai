"""Workspace files in Azure Blob Storage (or Azurite)."""

from ._model import AzureBlobWorkspaceConfig
from ._workspace import AzureBlobWorkspace

__all__ = ["AzureBlobWorkspace", "AzureBlobWorkspaceConfig"]
