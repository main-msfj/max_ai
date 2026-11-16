"""read_file: read code, text, a notebook or a PDF from the session's folders."""

from __future__ import annotations

import json
import shlex
import typing as t
from pathlib import Path

from .....base.tools import CoreTool, ToolContext
from .....types.tool_call import ToolCallRecord, ToolResult
from .....types.tools import ToolApprovalMode

SCRIPT = Path(__file__).with_name("script.py").read_text()  # sent to the sandbox


class ReadFileTool(CoreTool):
    """Reads lines of a file inside the session's workspace, tmp or skills."""

    TOOL_NAME: t.ClassVar[str] = "read_file"
    DESCRIPTION: t.ClassVar[str] = (
        "Read a file: code, text, a Jupyter notebook or a PDF. Returns its lines "
        "with their numbers; use offset and limit to read a long file in parts."
    )
    MAX_LINES: t.ClassVar[int] = 2000  # lines one call returns at most
    MAX_LINE_CHARS: t.ClassVar[int] = 2000  # a longer line is cut

    sandbox = True  # the files are there, not on the host

    def __init__(self) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=self.DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            read_only=True,
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
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "First line to read. Default 1.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": f"How many lines to read. Default and maximum {self.MAX_LINES}.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        path = tool_request.parameters["path"]
        first = tool_request.parameters.get("offset", 1)
        limit = min(tool_request.parameters.get("limit", self.MAX_LINES), self.MAX_LINES)

        # 1. Where this session may read: its workspace, its tmp and the skills.
        sandbox = tool_ctx.sandbox
        paths = sandbox.session_paths(tool_ctx.session_id)
        request = {
            "path": path, "cwd": paths["workspace"], "first": first, "limit": limit,
            "width": self.MAX_LINE_CHARS,
            "allowed": [paths["workspace"], paths["tmp"], str(sandbox.layout.skills)],
        }

        # 2. The script runs inside the sandbox, where the files and their links are.
        ran = await sandbox.execute_command(
            f"python3 -c {shlex.quote(SCRIPT)} {shlex.quote(json.dumps(request))}"
        )
        if ran.exit_code != 0:
            return ToolResult.execution_error(tool_request.id, f"{path} could not be read.")
        answer = json.loads(ran.stdout)

        # 3. The script explains in words what went wrong.
        if not answer["ok"]:
            return ToolResult.execution_error(tool_request.id, f"{path}: {answer['error']}.")

        # 4. Tell the model when there is more, and where to continue.
        total, lines = answer["total"], answer["lines"]
        last = first + limit - 1
        if not lines:
            return ToolResult.success_result(tool_request.id, f"{path} has {total} lines.")
        if last < total:
            lines += f"\n[lines {first}-{last} of {total}; continue with offset={last + 1}]"
        return ToolResult.success_result(tool_request.id, lines)
