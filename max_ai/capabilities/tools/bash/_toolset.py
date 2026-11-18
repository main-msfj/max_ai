"""bash: run a command inside the sandbox, starting where the last one ended."""

from __future__ import annotations

import shlex
import typing as t
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from ....base.tools import CoreTool, CoreToolParameters, ToolContext
from ....core.ids import short_id
from ....types.tool_call import ToolCallRecord, ToolResult

SCRIPT = Path(__file__).with_name("script.sh").read_text()  # sent to the sandbox


class BashToolConfig(BaseModel):
    """bash has no settings: stored empty, so an agent that has it can be stored."""


class BashTool(CoreTool):
    """Runs a command with bash inside the sandbox."""

    component_schema = BashToolConfig
    component_provider_override = "max_ai.capabilities.tools.bash.BashTool"

    TOOL_NAME: t.ClassVar[str] = "bash"
    DESCRIPTION: t.ClassVar[str] = (
        "Run a command with bash. The working directory is kept between calls; "
        "variables are not, every call is a new shell. The command cannot ask "
        "for input. Long output is cut in the middle."
    )
    MAX_OUTPUT_BYTES: t.ClassVar[int] = 30_000  # more is cut in the middle
    DEFAULT_TIMEOUT: t.ClassVar[int] = 120  # seconds, when the model gives none
    MAX_TIMEOUT: t.ClassVar[int] = 600  # seconds the model can ask for

    sandbox = True  # commands never run on the host
    timeout = MAX_TIMEOUT + 10  # the dispatcher waits a bit longer than the command

    def __init__(self) -> None:
        super().__init__(name=self.TOOL_NAME, description=self.DESCRIPTION)

    def _to_config(self) -> BashToolConfig:
        return BashToolConfig()

    @classmethod
    def _from_config(cls, config: BaseModel) -> t.Self:
        return cls()

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        # A blank command is checked here, not with a schema pattern: providers
        # that force the schema while the model writes read it as the whole text.
        checked = super().validate_parameters(tool_request)
        if checked.is_tool_valid and not tool_request.parameters["command"].strip():
            return CoreToolParameters(is_tool_valid=False, msg_error="The command is empty.")
        return checked

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to run.",
                },
                "description": {
                    "type": "string",
                    "description": "What the command does, in a few words.",
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self.MAX_TIMEOUT,
                    "description": f"Seconds to wait. Default {self.DEFAULT_TIMEOUT}.",
                },
            },
            "required": ["command", "description"],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        command = tool_request.parameters["command"]
        limit = min(tool_request.parameters.get("timeout", self.DEFAULT_TIMEOUT), self.MAX_TIMEOUT)

        # 1. The folders of this session, and where the last command ended.
        sandbox, ctx = tool_ctx.sandbox, tool_ctx.run_context
        if sandbox is None:
            return ToolResult.execution_error(tool_request.id, "bash needs a sandbox.")
        paths = sandbox.session_paths(tool_ctx.session_id)
        layout = sandbox.layout
        allowed = [paths["workspace"], paths["tmp"], str(layout.skills), str(layout.memory)]
        mark = f"__cwd_{short_id()}__"
        log = f"{paths['tmp']}/tool_results/{short_id()}.log"
        values = {
            "__start": ctx.cwd or paths["workspace"], "__workspace": paths["workspace"],
            "__log": log, "__max": str(self.MAX_OUTPUT_BYTES), "__mark": mark,
            "__command": command,
        }

        # 2. The script with its values on top, each one quoted.
        script = "".join(f"{name}={shlex.quote(value)}\n" for name, value in values.items())
        ran = await sandbox.execute_command(script + SCRIPT, timeout=limit)
        output, found, cwd = ran.stdout.rpartition(mark)
        code, notes = ran.exit_code, []

        # 3. Killed by the time limit: the script printed nothing, take the end of its log.
        if ran.timed_out:
            rest = await sandbox.execute_command(
                f"tail -c {self.MAX_OUTPUT_BYTES} {shlex.quote(log)}"
            )
            output, code = rest.stdout, 124
            notes.append(f"[stopped after {limit} seconds]")

        # 4. Keep the folder for the next call only if it is an allowed one.
        elif found and self.within(cwd, allowed):
            ctx.cwd = None if cwd == paths["workspace"] else cwd
        elif found:
            ctx.cwd = None
            notes.append(
                f"[you left the allowed folders ({cwd}); the next command starts in the workspace]"
            )

        # 5. Exit code first, then the output, then the notes.
        parts = [f"exit code: {code}", output.strip("\n"), *notes]
        return ToolResult.success_result(tool_request.id, "\n".join(p for p in parts if p))

    @staticmethod
    def within(folder: str, allowed: list[str]) -> bool:
        """``folder`` is one of ``allowed`` or inside one."""
        return any(PurePosixPath(folder).is_relative_to(root) for root in allowed)
