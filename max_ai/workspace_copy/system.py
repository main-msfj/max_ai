"""Compatibility imports for the native Workspace implementation."""

from ..capabilities.workspace.local import LocalWorkspace as Workspace

LocalWorkSpace = Workspace

__all__ = ["Workspace", "LocalWorkSpace"]
