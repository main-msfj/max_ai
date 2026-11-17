"""read_file."""

from __future__ import annotations

import json
import shlex
import typing as t
from pathlib import Path

from ......base.tools import ToolContext
from ......config import setting
from ......core.termination.cancellation import CancellationToken
from ......types.tool_call import ToolCallRecord, ToolResult
from ......types.tools import ToolApprovalMode
from .._base import FileSystem

# Read once from the package, then sent to the environment with every call
SCRIPT = (Path(__file__).parent / "script.py").read_text()


class ReadFile(FileSystem):
    TOOL_NAME: t.ClassVar[str] = "read_file"
    DESCRIPTION = "Read a text file from the workspace, optionally a range of its lines."

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
                "path": {
                    "type": "string",
                    "description": (
                        "Starts with $SESSION/ for this session, $WORKSPACE/ for all sessions"
                        " or $TMP/ for scratch files."
                    ),
                },
                "offset": {"type": "integer", "minimum": 1, "description": "First line, from 1."},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "How many lines. Default 2000.",
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        call_id = tool_request.id
        sandbox = (tool_context.deps or {}).get("sandbox") if tool_context else None
        if tool_context is None or sandbox is None:
            return ToolResult.execution_error(call_id, "read_file needs a sandbox")
        try:
            root, path = self.locate(tool_request.parameters["path"], tool_context.session_id)
        except ValueError as error:
            return ToolResult.execution_error(call_id, str(error))

        request = {**tool_request.parameters, "root": root, "path": path}
        timeout = setting.tool_timeout_seconds
        result = await sandbox.execute_command(
            f"python3 -c {shlex.quote(SCRIPT)}",
            timeout=timeout,
            stdin=json.dumps(request).encode(),
        )

        if result.timed_out:
            return ToolResult.timeout(call_id, timeout)
        if result.exit_code != 0:
            return ToolResult.execution_error(call_id, result.stderr)
        answer = json.loads(result.stdout)
        if not answer.pop("ok"):
            return ToolResult.execution_error(call_id, answer["error"])
        return ToolResult.success_result(call_id, answer)
