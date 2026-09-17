"""Compatibility imports for the native Workspace capability."""

from ...base.workspace import Workspace, WorkspaceConfig

WorkspaceLocal = Workspace
WorkspaceLocalConfig = WorkspaceConfig

__all__ = ["Workspace", "WorkspaceConfig", "WorkspaceLocal", "WorkspaceLocalConfig"]
