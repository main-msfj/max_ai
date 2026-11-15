"""read_file."""

from __future__ import annotations

import typing as t

from max_ai.types.tools import ToolApprovalMode

from .._base import PATH_DESCRIPTION, FileTool, script_of


class ReadFile(FileTool):
    TOOL_NAME: t.ClassVar[str] = "read_file"
    DESCRIPTION = "Read a text file, optionally a range of its lines. Lines come numbered."
    SCRIPT = script_of(__file__)

    def __init__(self) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=self.DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            read_only=True,
            policy_subject="path",
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": PATH_DESCRIPTION},
                "offset": {"type": "integer", "minimum": 1, "description": "First line, from 1."},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "How many lines. Default 2000.",
                },
            },
            "required": ["path"],
        }
