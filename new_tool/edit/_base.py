"""edit_file."""

from __future__ import annotations

import typing as t

from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import CoreToolParameters, ToolApprovalMode

from .._base import PATH_DESCRIPTION, FileTool, script_of


class EditFile(FileTool):
    TOOL_NAME: t.ClassVar[str] = "edit_file"
    DESCRIPTION = (
        "Replace an exact piece of text in a file with new text. Read the "
        "file first; old_string must appear once, or set replace_all."
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
                "old_string": {
                    "type": "string",
                    "description": "Text to replace, copied from the file without line numbers.",
                },
                "new_string": {"type": "string", "description": "Text to put in its place."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence, not only a unique one.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        """The schema, plus the two mistakes that need no file to be seen."""
        validation = super().validate_parameters(tool_request)
        if not validation.is_tool_valid:
            return validation
        old = tool_request.parameters["old_string"]
        new = tool_request.parameters["new_string"]
        if not old:
            return CoreToolParameters(
                is_tool_valid=False,
                msg_error="old_string is empty; to create a file use write_file",
            )
        if old == new:
            return CoreToolParameters(
                is_tool_valid=False,
                msg_error="old_string and new_string are the same: nothing would change",
            )
        return validation
