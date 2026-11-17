"""edit_file: replace an exact text inside a file of the session's folders."""

from __future__ import annotations

import json
import shlex
import typing as t
from pathlib import Path

from .....base.tools import CoreTool, ToolContext
from .....types.tool_call import ToolCallRecord, ToolResult
from .....types.tools import ToolApprovalMode
from .._seen import FilesSeen

SCRIPT = Path(__file__).with_name("script.py").read_text()  # sent to the sandbox


class EditFileTool(CoreTool):
    """Replaces a text with another inside a file of the workspace or tmp."""

    TOOL_NAME: t.ClassVar[str] = "edit_file"
    DESCRIPTION: t.ClassVar[str] = (
        "Change part of a text file: replace old_text with new_text. old_text "
        "must be copied exactly from the file, spaces and line breaks included, "
        "and appear only once; add lines around it to make it unique, or set "
        "replace_all to change every place. Read the file first."
    )

    sandbox = True  # the files are there, not on the host

    def __init__(self) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=self.DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file, relative to your working directory or absolute.",
                },
                "old_text": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The text to replace, exactly as it is in the file.",
                },
                "new_text": {
                    "type": "string",
                    "description": "The text that takes its place. Empty deletes it.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every place old_text appears. Default false.",
                },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        path = tool_request.parameters["path"]
        old, new = tool_request.parameters["old_text"], tool_request.parameters["new_text"]

        # 1. Nothing to do when both texts are the same.
        if old == new:
            return ToolResult.execution_error(
                tool_request.id, f"{path}: old_text and new_text are the same."
            )

        # 2. Where this session may write: its workspace, its tmp and the memory; not the skills.
        sandbox = tool_ctx.sandbox
        paths = sandbox.session_paths(tool_ctx.session_id)
        allowed = [paths["workspace"], paths["tmp"], str(sandbox.layout.memory)]
        start = FilesSeen.write_start(tool_ctx, paths["workspace"], allowed)
        request = {
            "path": path, "cwd": start, "old": old, "new": new,
            "all": tool_request.parameters.get("replace_all", False),
            "allowed": allowed,
            "memory": str(sandbox.layout.memory), "memory_max": sandbox.layout.MEMORY_FILE_BYTES,
        }

        # 3. The request goes on stdin: the texts can be longer than an argument allows.
        ran = await sandbox.execute_command(
            f"python3 -c {shlex.quote(SCRIPT)}", stdin=json.dumps(request).encode("utf-8")
        )
        if ran.exit_code != 0:
            return ToolResult.execution_error(tool_request.id, f"{path} could not be edited.")
        answer = json.loads(ran.stdout)

        # 4. The script explains in words what went wrong.
        if not answer["ok"]:
            return ToolResult.execution_error(tool_request.id, f"{path}: {answer['error']}.")

        # 5. The model knows what the file holds now: it edited it.
        tool_ctx.run_context.files_seen[FilesSeen.key(start, path)] = answer["sha256"]

        # 6. Say how many places changed.
        places = "1 place" if answer["replaced"] == 1 else f"{answer['replaced']} places"
        return ToolResult.success_result(tool_request.id, f"Edited {path}: {places} changed.")
