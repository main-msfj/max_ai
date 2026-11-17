"""What the file tools share: where a path starts and how a seen file is named."""

from __future__ import annotations

import posixpath

from ....base.tools import ToolContext


class FilesSeen:
    """The files the model saw live in ``ctx.files_seen``: path -> sha256."""

    @staticmethod
    def start(tool_ctx: ToolContext, workspace: str) -> str:
        """Where a relative path starts: where bash ended, or the workspace."""
        return tool_ctx.run_context.cwd or workspace

    @staticmethod
    def key(start: str, path: str) -> str:
        """The name of ``path`` in ``ctx.files_seen``: absolute, without ``..``.
        Links are not resolved: a file reached through a link counts as not seen."""
        return posixpath.normpath(posixpath.join(start, path))
