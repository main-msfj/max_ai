"""write_file."""

from __future__ import annotations

import typing as t

from ......base.tools import ToolContext
from ......core.termination.cancellation import CancellationToken
from ......types.tool_call import ToolCallRecord, ToolResult
from ......types.tools import ToolApprovalMode
from .._base import FileSystem


class WriteFile(FileSystem):
    TOOL_NAME: t.ClassVar[str] = "write_file"
    DESCRIPTION = "Create a file in the workspace, or replace all of its content."

    def __init__(self) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=self.DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            policy_subject="path",
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the workspace."},
                "content": {"type": "string", "description": "The whole new content."},
            },
            "required": ["path", "content"],
        }

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        raise NotImplementedError
