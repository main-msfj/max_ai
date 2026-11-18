"""save_memory: keep one fact in the user's memory, as a file with its header."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import typing as t
from pathlib import Path

from ....base.tools import CoreTool, CoreToolParameters, ToolContext
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode

SCRIPT = Path(__file__).with_name("script.py").read_text()  # sent to the sandbox


class SaveMemoryTool(CoreTool):
    """Writes ``<memory>/<name>.md`` with its header: the model gives the
    fact, never a path or the header's format."""

    TOOL_NAME: t.ClassVar[str] = "save_memory"
    DESCRIPTION: t.ClassVar[str] = (
        "Remember a fact for future conversations: who the user is, how they want you "
        "to work, a decision that will still hold. One fact per call: two facts are two "
        "calls. A name that is already in your memory updates that fact."
    )
    TYPES: t.ClassVar[tuple[str, ...]] = ("user", "feedback", "project", "reference")
    NAME: t.ClassVar[str] = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
    MAX_DESCRIPTION: t.ClassVar[int] = 150  # characters: what the index shows of it

    sandbox = True  # the memory is kept there, not on the host

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
                "name": {
                    "type": "string",
                    "description": "Short name of the fact, with hyphens: prefers-pytest.",
                },
                "type": {
                    "type": "string",
                    "enum": list(self.TYPES),
                    "description": (
                        "user: who they are. feedback: how they want you to work. "
                        "project: a decision or goal. reference: where something is."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "The fact itself in one line, because this line is all you see of it "
                        "in later conversations: 'Installs packages with uv, never pip'."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The fact with its details, and the why when it has one.",
                },
            },
            "required": ["name", "type", "description", "content"],
            "additionalProperties": False,
        }

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        # 1. The schema first: after it, the four fields are there.
        checked = super().validate_parameters(tool_request)
        if not checked.is_tool_valid:
            return checked
        given = tool_request.parameters
        description = str(given["description"]).strip()

        # 2. A name that is a file name, and a description that is one line.
        error = ""
        if not re.fullmatch(self.NAME, str(given["name"])):
            error = "name takes letters, digits and hyphens only, at most 64: prefers-pytest"
        elif not description or "\n" in description or len(description) > self.MAX_DESCRIPTION:
            error = f"description is one line of at most {self.MAX_DESCRIPTION} characters"
        elif not str(given["content"]).strip():
            error = "content is empty: say the fact"
        if error:
            return CoreToolParameters(is_tool_valid=False, msg_error=error)
        return checked

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        given = tool_request.parameters
        name, file = str(given["name"]), f"{given['name']}.md"
        sandbox = tool_ctx.sandbox
        if sandbox is None:
            return ToolResult.execution_error(tool_request.id, f"{self.TOOL_NAME} needs a sandbox.")

        # 1. The file the index reads: the header, then the fact.
        text = (
            f"---\nname: {name}\ndescription: {str(given['description']).strip()}\n"
            f"type: {given['type']}\n---\n{str(given['content']).strip()}\n"
        )
        content = text.encode("utf-8")
        most = sandbox.layout.MEMORY_FILE_BYTES
        if len(content) > most:
            return ToolResult.execution_error(
                tool_request.id,
                f"A memory file holds at most {most // 1000} kB; save it as several smaller facts.",
            )

        # 2. The script writes it inside the sandbox; the content goes on stdin.
        digest = hashlib.sha256(content).hexdigest()
        request = {"folder": str(sandbox.layout.memory), "file": file, "sha256": digest}
        ran = await sandbox.execute_command(
            f"python3 -c {shlex.quote(SCRIPT)} {shlex.quote(json.dumps(request))}", stdin=content
        )
        if ran.exit_code != 0:
            return ToolResult.execution_error(tool_request.id, f"{file} could not be saved.")
        answer = json.loads(ran.stdout)
        if not answer["ok"]:
            return ToolResult.execution_error(tool_request.id, f"{file}: {answer['error']}.")

        # 3. The model knows what the file holds now: it can edit it later.
        tool_ctx.run_context.files_seen[str(sandbox.layout.memory / file)] = digest
        did = "Updated" if answer["existed"] else "Saved"
        return ToolResult.success_result(tool_request.id, f"{did} {file} in your memory.")
