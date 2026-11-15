"""edit_file."""

from __future__ import annotations

import typing as t

from .....base.tools import ToolContext
from .....core.termination.cancellation import CancellationToken
from .....types.tool_call import ToolCallRecord, ToolResult
from .....types.tools import ToolApprovalMode
from .._base import FileSystem


class EditFile(FileSystem):
    TOOL_NAME: t.ClassVar[str] = "edit_file"
    DESCRIPTION = "Replace an exact piece of text in a file with new text."

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
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Text to put in its place."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence, not only a unique one.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        raise NotImplementedError
