"""bash: run a command inside the sandbox, starting where the last one ended."""

from __future__ import annotations

import json
import shlex
import typing as t
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from ....base.tools import CoreTool, CoreToolParameters, ToolContext
from ....core.ids import short_id
from ....errors.sandbox import SandboxError
from ....types.tool_call import ToolCallRecord, ToolResult

SCRIPT = Path(__file__).with_name("script.sh").read_text()  # sent to the sandbox
CHECK = Path(__file__).with_name("check.py").read_text()  # run by the script when the command ends


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
        "for input. Long output is cut in the middle. A command you expect to "
        "take a minute or more (a server, a build, a long test run, a sleep) "
        "must use background=true: it starts and the call returns at once with "
        "a task id. Do not wait for it with a long timeout."
    )
    STARTED: t.ClassVar[str] = (
        "Started in the background as task {id}. You will be told when it ends; "
        "task_output shows what it has printed so far, stop_task stops it."
    )
    MAX_OUTPUT_BYTES: t.ClassVar[int] = 30_000  # more is cut in the middle
    DEFAULT_TIMEOUT: t.ClassVar[int] = 120  # seconds, when the model gives none
    MAX_TIMEOUT: t.ClassVar[int] = 600  # seconds the model can ask for
    MAX_SEEN: t.ClassVar[int] = 200  # files checked after a command; the rest, when written
    MAX_NAMED: t.ClassVar[int] = 10  # files a note names; the rest are counted
    CHANGED: t.ClassVar[str] = (
        "[this command changed {count} file(s) you had read: {names}. "
        "Read them again before you edit them.]"
    )
    DELETED: t.ClassVar[str] = "[this command deleted {count} file(s) you had read: {names}.]"

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
                "background": {
                    "type": "boolean",
                    "description": (
                        "True for a command expected to take a minute or more: it "
                        "starts and the call returns at once with a task id. The "
                        "timeout does not apply."
                    ),
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
        # A background command: the sandbox starts it and the call returns at once.
        if tool_request.parameters.get("background"):
            try:
                task_id = await sandbox.start_task(
                    tool_ctx.session_id, command, ctx.cwd or paths["workspace"]
                )
            except SandboxError as error:
                return ToolResult.execution_error(tool_request.id, str(error))
            ctx.tasks[task_id] = command
            return ToolResult.success_result(
                tool_request.id, self.STARTED.format(id=task_id), metadata={"task_id": task_id}
            )
        layout = sandbox.layout
        allowed = [paths["workspace"], paths["tmp"], str(layout.skills), str(layout.memory)]
        mark = f"__cwd_{short_id()}__"
        log = f"{paths['tmp']}/tool_results/{short_id()}.log"
        # The last files the model read, with the hash it saw of each.
        seen = dict(list(ctx.files_seen.items())[-self.MAX_SEEN:])
        values = {
            "__start": ctx.cwd or paths["workspace"], "__workspace": paths["workspace"],
            "__log": log, "__max": str(self.MAX_OUTPUT_BYTES), "__mark": mark,
            "__check": CHECK, "__seen": json.dumps(seen), "__command": command,
        }

        # 2. The script with its values on top, each one quoted.
        script = "".join(f"{name}={shlex.quote(value)}\n" for name, value in values.items())
        ran = await sandbox.execute_command(script + SCRIPT, timeout=limit)
        rest, _, checked = ran.stdout.rpartition(mark)
        output, found, cwd = rest.rpartition(mark)
        code = ran.exit_code
        notes: list[str] = []

        # 3. Killed by the time limit: the script printed nothing, take the end of its log.
        if ran.timed_out:
            rest = await sandbox.execute_command(
                f"tail -c {self.MAX_OUTPUT_BYTES} {shlex.quote(log)}"
            )
            output, code = rest.stdout, 124
            notes.append(f"[stopped after {limit} seconds]")
            # The script was killed before its check: ask for it now.
            again = await sandbox.execute_command(
                f"python3 -c {shlex.quote(CHECK)} {shlex.quote(json.dumps(seen))}"
            )
            checked = again.stdout

        # 4. Keep the folder for the next call only if it is an allowed one.
        elif found and self.within(cwd, allowed):
            ctx.cwd = None if cwd == paths["workspace"] else cwd
        elif found:
            ctx.cwd = None
            notes.append(
                f"[you left the allowed folders ({cwd}); the next command starts in the workspace]"
            )

        # 5. Files the model had read and the command changed: it is told now,
        #    and they count as not seen, so an edit asks to read them first.
        notes += self.forget(ctx.files_seen, checked)

        # 6. Exit code first, then the output, then the notes.
        parts = [f"exit code: {code}", output.strip("\n"), *notes]
        return ToolResult.success_result(tool_request.id, "\n".join(p for p in parts if p))

    @classmethod
    def forget(cls, files_seen: dict[str, str], checked: str) -> list[str]:
        """Take out of ``files_seen`` what the check found changed or gone,
        and return the notes that tell the model."""
        # 1. No answer from the check: nothing is said, a write still notices.
        try:
            answer: dict[str, list[str]] = json.loads(checked)
        except ValueError:
            return []
        notes: list[str] = []
        for kind, text in (("changed", cls.CHANGED), ("deleted", cls.DELETED)):
            paths = [path for path in answer.get(kind, []) if files_seen.pop(path, None) is not None]
            if not paths:
                continue
            # 2. A few by name, the rest counted.
            names = ", ".join(paths[: cls.MAX_NAMED])
            if len(paths) > cls.MAX_NAMED:
                names += f" and {len(paths) - cls.MAX_NAMED} more"
            notes.append(text.format(count=len(paths), names=names))
        return notes

    @staticmethod
    def within(folder: str, allowed: list[str]) -> bool:
        """``folder`` is one of ``allowed`` or inside one."""
        return any(PurePosixPath(folder).is_relative_to(root) for root in allowed)
