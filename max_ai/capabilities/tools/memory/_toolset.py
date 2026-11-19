"""save_memory: keep one fact in the user's memory, as a file with its header."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import typing as t
from pathlib import Path

from ....base.sandbox import SandboxCore
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
        "to work, a decision that will still hold. One fact per call: 'I am a backend "
        "developer and I use uv' is two calls, one for who they are and one for what "
        "they use. A name that is already in your memory updates that fact."
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

        # 2. A name that is a file name, and a description that is one line.
        error = self.problem(str(given["name"]), str(given["description"]), str(given["content"]))
        if error:
            return CoreToolParameters(is_tool_valid=False, msg_error=error)
        return checked

    @classmethod
    def problem(cls, name: str, description: str, content: str) -> str:
        """Why these cannot be saved as a fact; empty when they can."""
        description = description.strip()
        if not re.fullmatch(cls.NAME, name):
            return "name takes letters, digits and hyphens only, at most 64: prefers-pytest"
        if not description or "\n" in description or len(description) > cls.MAX_DESCRIPTION:
            return f"description is one line of at most {cls.MAX_DESCRIPTION} characters"
        if not content.strip():
            return "content is empty: say the fact"
        return ""

    @classmethod
    async def write(
        cls, sandbox: SandboxCore, name: str, kind: str, description: str, content: str
    ) -> tuple[str, bool]:
        """Write the fact as ``<memory>/<name>.md`` and return its hash and
        whether the file was already there. A ValueError says why it could not."""
        # 1. The same checks the tool makes on what the model gives.
        error = cls.problem(name, description, content)
        if not error and kind not in cls.TYPES:
            error = f"type is one of: {', '.join(cls.TYPES)}"
        if error:
            raise ValueError(error)

        # 2. The file the index reads: the header, then the fact.
        file = f"{name}.md"
        text = (
            f"---\nname: {name}\ndescription: {description.strip()}\n"
            f"type: {kind}\n---\n{content.strip()}\n"
        )
        data = text.encode("utf-8")
        most = sandbox.layout.MEMORY_FILE_BYTES
        if len(data) > most:
            raise ValueError(
                f"A memory file holds at most {most // 1000} kB; save it as several smaller facts."
            )

        # 3. The script writes it inside the sandbox; the content goes on stdin.
        digest = hashlib.sha256(data).hexdigest()
        request = {"folder": str(sandbox.layout.memory), "file": file, "sha256": digest}
        ran = await sandbox.execute_command(
            f"python3 -c {shlex.quote(SCRIPT)} {shlex.quote(json.dumps(request))}", stdin=data
        )
        if ran.exit_code != 0:
            raise ValueError(f"{file} could not be saved.")
        answer = json.loads(ran.stdout)
        if not answer["ok"]:
            raise ValueError(f"{file}: {answer['error']}.")
        return digest, bool(answer["existed"])

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        given = tool_request.parameters
        file = f"{given['name']}.md"
        sandbox = tool_ctx.sandbox
        if sandbox is None:
            return ToolResult.execution_error(tool_request.id, f"{self.TOOL_NAME} needs a sandbox.")

        # 1. Write the fact; what went wrong goes to the model in words.
        try:
            digest, existed = await self.write(
                sandbox, str(given["name"]), str(given["type"]),
                str(given["description"]), str(given["content"]),
            )
        except ValueError as error:
            return ToolResult.execution_error(tool_request.id, str(error))

        # 2. The model knows what the file holds now: it can edit it later.
        tool_ctx.run_context.files_seen[str(sandbox.layout.memory / file)] = digest
        did = "Updated" if existed else "Saved"
        return ToolResult.success_result(tool_request.id, f"{did} {file} in your memory.")
