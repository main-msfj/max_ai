"""read_skill: read the instructions of an installed skill."""

from __future__ import annotations

import re
import shlex
import typing as t

from ....base.tools import CoreTool, ToolContext
from ....core.termination.cancellation import CancellationToken
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode


class ReadSkillTool(CoreTool):
    """Reads one skill's SKILL.md from the sandbox and returns it."""

    TOOL_NAME: t.ClassVar[str] = "read_skill"
    DESCRIPTION: t.ClassVar[str] = (
        "Read the instructions of a skill. Call it before you use a skill."
    )
    FILE: t.ClassVar[str] = "SKILL.md"
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

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        name = tool_request.parameters["skill_name"]
        # A name, never a path: nothing outside the skills folder can be read.
        if not re.fullmatch(self.SKILL_NAME, name):
            return ToolResult.execution_error(tool_request.id, f"{name!r} is not a skill name.")
        if tool_context is None or tool_context.sandbox is None:
            return ToolResult.execution_error(tool_request.id, "Skills need a sandbox.")
        sandbox = tool_context.sandbox
        path = sandbox.layout.skills / name / self.FILE
        read = await sandbox.execute_command(f"cat {shlex.quote(str(path))}")
        if read.exit_code != 0:
            return ToolResult.execution_error(
                tool_request.id, f"There is no skill named {name!r}."
            )
        return ToolResult.success_result(tool_request.id, read.stdout)
