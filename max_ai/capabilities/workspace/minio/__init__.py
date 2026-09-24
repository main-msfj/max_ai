"""Workspace files in MinIO or any S3-compatible storage."""

from ._model import MinIOWorkspaceConfig
from ._workspace import MinIOWorkspace

__all__ = ["MinIOWorkspace", "MinIOWorkspaceConfig"]
