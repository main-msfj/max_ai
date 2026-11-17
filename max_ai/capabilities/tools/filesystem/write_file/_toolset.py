"""write_file: create a text file, or replace one, in the session's folders."""

from __future__ import annotations

import hashlib
import json
import shlex
import typing as t
from pathlib import Path

from .....base.tools import CoreTool, ToolContext
from .....types.tool_call import ToolCallRecord, ToolResult
from .....types.tools import ToolApprovalMode
from .._seen import FilesSeen

SCRIPT = Path(__file__).with_name("script.py").read_text()  # sent to the sandbox


class WriteFileTool(CoreTool):
    """Writes a whole text file inside the session's workspace or tmp."""

    TOOL_NAME: t.ClassVar[str] = "write_file"
    DESCRIPTION: t.ClassVar[str] = (
        "Create a text file, or replace an existing one, with the full content "
        "given. It does not append: what was in the file is lost. Read an "
        "existing file before replacing it. Missing folders are created. "
        "For text only (code, markdown, JSON, CSV)."
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
                "content": {
                    "type": "string",
                    "description": "Everything the file will contain.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        path = tool_request.parameters["path"]
        content = tool_request.parameters["content"].encode("utf-8")

        # 1. Where this session may write: its workspace and its tmp, not the skills.
        sandbox = tool_ctx.sandbox
        paths = sandbox.session_paths(tool_ctx.session_id)
        seen = tool_ctx.run_context.files_seen
        start = FilesSeen.start(tool_ctx, paths["workspace"])
        key = FilesSeen.key(start, path)
        request = {
            "path": path, "cwd": start,
            "allowed": [paths["workspace"], paths["tmp"]],
            "sha256": hashlib.sha256(content).hexdigest(), "seen": seen.get(key),
        }

        # 2. The content goes on stdin: an argument has a size limit, stdin does not.
        ran = await sandbox.execute_command(
            f"python3 -c {shlex.quote(SCRIPT)} {shlex.quote(json.dumps(request))}",
            stdin=content,
        )
        if ran.exit_code != 0:
            return ToolResult.execution_error(tool_request.id, f"{path} could not be written.")
        answer = json.loads(ran.stdout)

        # 3. The script explains in words what went wrong.
        if not answer["ok"]:
            return ToolResult.execution_error(tool_request.id, f"{path}: {answer['error']}.")

        # 4. The model knows what the file holds now: it wrote it.
        seen[key] = request["sha256"]

        # 5. Say whether it is new, so the model notices a replace it did not expect.
        did = "Replaced" if answer["existed"] else "Created"
        lines = len(content.splitlines())
        return ToolResult.success_result(tool_request.id, f"{did} {path} ({lines} lines).")
