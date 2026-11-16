"""read_skill: read the instructions of an installed skill."""

from __future__ import annotations

import re
import shlex
import typing as t

from ....base.tools import CoreTool, ToolContext
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode


class ReadSkillTool(CoreTool):
    """Reads one skill's SKILL.md from the sandbox and returns it."""

    TOOL_NAME: t.ClassVar[str] = "read_skill"
    DESCRIPTION: t.ClassVar[str] = "Read the instructions of a skill. Call it before you use a skill."
    FILE: t.ClassVar[str] = "SKILL.md"
    MAX_FILES: t.ClassVar[int] = 100  # files of the skill listed at most
    MARK: t.ClassVar[str] = "__skill_md__"  # between the list of files and the SKILL.md
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

        # 2. The skills live in the sandbox: with sandbox = True the dispatcher
        #    never gets here without one.
        sandbox = tool_ctx.sandbox

        # 3. Where the layout puts it: .agent/skills/<name>, with its SKILL.md
        folder = shlex.quote(str(sandbox.layout.skills / name))
        read = await sandbox.execute_command(
            f"cd {folder} && find . -type f | sed 's|^\\./||' | sort | head -n {self.MAX_FILES}"
            f" && echo {self.MARK} && cat {self.FILE}"
        )

        # 4. cd or cat fail when it is missing: the skill is not installed.
        if read.exit_code != 0:
            return ToolResult.execution_error(
                tool_request.id, f"There is no skill named {name!r}."
            )

        # 5. First the folder and its files, so the model can open them; then the SKILL.md.
        files, _, instructions = read.stdout.partition(f"{self.MARK}\n")
        listed = "".join(f"  {file}\n" for file in files.splitlines())
        header = f"Skill folder: {sandbox.layout.skills / name}\n{listed}"
        return ToolResult.success_result(tool_request.id, f"{header}\n{instructions}")
