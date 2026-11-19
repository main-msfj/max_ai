"""read_skill: read the instructions of an installed skill."""

from __future__ import annotations

import json
import re
import shlex
import typing as t
from pathlib import Path

from ....base.tools import CoreTool, ToolContext
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode

SCRIPT = Path(__file__).with_name("script.py").read_text()  # sent to the sandbox


class ReadSkillTool(CoreTool):
    """Reads one skill's SKILL.md from the sandbox and returns it."""

    TOOL_NAME: t.ClassVar[str] = "read_skill"
    DESCRIPTION: t.ClassVar[str] = "Read the instructions of a skill. Call it before you use a skill."
    FILE: t.ClassVar[str] = "SKILL.md"
    MAX_FILES: t.ClassVar[int] = 100  # files of the skill listed at most
    MISSING: t.ClassVar[dict[str, str]] = {  # what the script found missing → what the model reads
        "missing": "There is no skill named {!r}.",
        "no_instructions": "The skill {!r} has no SKILL.md.",
    }
    NOT_LOADED: t.ClassVar[str] = "The skill {!r} is not loaded for this agent."
    SKILL_NAME: t.ClassVar[str] = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"

    sandbox = True  # the skills are installed there, not on the host

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
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill, exactly as it is listed.",
                },
            },
            "required": ["skill_name"],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        name = tool_request.parameters["skill_name"]

        # 1. A name, never a path: nothing outside the skills folder can be read.
        if not re.fullmatch(self.SKILL_NAME, name):
            return ToolResult.execution_error(tool_request.id, f"{name!r} is not a skill name.")

        # 2. Only a skill the agent was given: a folder someone left there is not one.
        if name not in tool_ctx.skills:
            return ToolResult.execution_error(tool_request.id, self.NOT_LOADED.format(name))

        # 3. The skills live in the sandbox: with sandbox = True the dispatcher
        #    never gets here without one.
        sandbox = tool_ctx.sandbox
        if sandbox is None:
            return ToolResult.execution_error(tool_request.id, "read_skill needs a sandbox.")

        # 4. The script runs inside the sandbox, on .agent/skills/<name>.
        folder = sandbox.layout.skills / name
        request: dict[str, t.Any] = {"folder": str(folder), "file": self.FILE, "max_files": self.MAX_FILES}
        ran = await sandbox.execute_command(
            f"python3 -c {shlex.quote(SCRIPT)} {shlex.quote(json.dumps(request))}"
        )
        if ran.exit_code != 0:
            return ToolResult.execution_error(tool_request.id, f"The skill {name!r} could not be read.")
        answer = json.loads(ran.stdout)

        # 5. The script says what is missing: the skill or its instructions.
        if not answer["ok"]:
            return ToolResult.execution_error(tool_request.id, self.MISSING[answer["error"]].format(name))

        # 6. First the folder and its files, so the model can open them; then the SKILL.md.
        listed = "".join(f"  {file}\n" for file in answer["files"])
        header = f"Skill folder: {folder}\n{listed}"
        return ToolResult.success_result(tool_request.id, f"{header}\n{answer['instructions']}")
