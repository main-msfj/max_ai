"""Configuration and toolset construction for filesystem access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ....workspace_copy.filesystem import UserFileSystem
from ._toolset import READ_ONLY_TOOL_NAMES, FileSystemTools


@dataclass
class FileSystem:
    """Expose user-scoped file tools using the workspace's existing path rules.

    When workspace is omitted, tools use the workspace from ToolContext.
    """

    workspace: str | Path | UserFileSystem | None = None
    read_only: bool = False

    def get_toolset(self) -> FileSystemTools:
        toolset = FileSystemTools(self.workspace)
        if self.read_only:
            toolset.tools = [
                tool for tool in toolset.tools if tool.name in READ_ONLY_TOOL_NAMES
            ]
        return toolset
