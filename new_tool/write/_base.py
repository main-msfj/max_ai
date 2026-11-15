"""write_file."""

from __future__ import annotations

import typing as t

from max_ai.types.tools import ToolApprovalMode

from .._base import PATH_DESCRIPTION, FileTool, script_of


class WriteFile(FileTool):
    TOOL_NAME: t.ClassVar[str] = "write_file"
    DESCRIPTION = (
        "Create a file, or replace all of its content. Read an existing file "
        "before replacing it."
    )
    SCRIPT = script_of(__file__)

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
                "path": {"type": "string", "description": PATH_DESCRIPTION},
                "content": {"type": "string", "description": "The whole new content."},
            },
            "required": ["path", "content"],
        }
