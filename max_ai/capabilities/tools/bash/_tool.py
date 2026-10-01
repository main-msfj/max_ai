"""The bash tool: the model sends only a command string.

The harness wraps it in a script (paths, working directory, output capture)
and the executor runs that script with ``bash -c`` wherever it lives: on the
host, in Docker or in Modal. Nothing of max_ai runs inside the environment.
"""

from __future__ import annotations

import shlex
import typing as t
from pathlib import Path, PurePosixPath

from ....base.tools import CoreRuntimeTool, ToolContext
from ....config import setting
from ....core.ids import short_id
from ....core.termination import CancellationToken
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import CoreToolParameters, ToolApprovalMode
from . import constant as c
from ._permissions import command_matches, split_command

if t.TYPE_CHECKING:
    from ....base.executor import ExecutionSession, ExecutorBase

# The script around every command; build_script() puts the values above it.
_WRAPPER = Path(__file__).with_name("wrapper.sh").read_text()


def build_script(
    command: str,
    *,
    cwd: str,
    workspace: str,
    skills: str,
    log: str,
    max_bytes: int,
) -> str:
    """The script the executor runs for one ``command``: ``wrapper.sh`` with
    its values assigned on top, each one shell-quoted.

    It starts in ``cwd``, runs the command (stdin closed, stdout and stderr
    into ``log``), then prints the output, cut in the middle past
    ``max_bytes``, the workspace files the command changed, and the final
    working directory.
    """
    values = {
        "WORKSPACE": workspace,
        "SKILLS": skills,
        "__maxai_cwd": cwd,
        "__maxai_log": log,
        "__maxai_max_bytes": str(max_bytes),
        "__maxai_command": command,
    }
    return "".join(f"{name}={shlex.quote(value)}\n" for name, value in values.items()) + _WRAPPER


def changed_files(listing: str) -> dict[str, list[str]]:
    """Created, modified and deleted workspace files from ``comm -3`` output.

    ``comm`` prints entries only in the before snapshot as they are and entries
    only in the after one behind a tab; a path on both sides changed size or
    mtime. Empty groups are left out.
    """
    before: set[str] = set()
    after: set[str] = set()
    for line in listing.splitlines():
        side, entry = (after, line[1:]) if line.startswith("\t") else (before, line)
        path = entry.rsplit("\t", 2)[0]  # size and mtime are the last two fields
        if path:
            side.add(path)
    groups = {
        "created": sorted(after - before),
        "modified": sorted(after & before),
        "deleted": sorted(before - after),
    }
    return {name: _first(paths) for name, paths in groups.items() if paths}


def _first(paths: list[str]) -> list[str]:
    """``paths`` up to ``MAX_LISTED``, with a count of the rest."""
    if len(paths) <= c.MAX_LISTED:
        return paths
    return [*paths[: c.MAX_LISTED], f"... and {len(paths) - c.MAX_LISTED} more"]


class BashTool(CoreRuntimeTool):
    """Run shell commands in the agent's execution environment.

    The working directory persists between calls of a conversation;
    environment variables do not (every call is a fresh shell). Every call
    stops after ``TOOL_TIMEOUT_SECONDS``.

    Whether a command runs is the agent's Policy: rules like ``Bash(git push:*)``
    match each command in the line (see ``permission_subjects``).
    """

    runs_commands = True

    def __init__(
        self,
        name: str = "bash",
        description: str | None = None,
        max_output_bytes: int = c.MAX_OUTPUT_BYTES,
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
    ) -> None:
        """Initialize ``BashTool``.

        Parameters
        ----------
        name : str
            Tool name the model calls.
        description : str | None
            Replaces the default description.
        max_output_bytes : int
            Output kept for the model; longer output is cut in the middle.
        approval_mode : ToolApprovalMode | str
            Default when no Policy rule matches the command.
        """
        if max_output_bytes < 2:
            raise ValueError("max_output_bytes must be at least 2")
        super().__init__(
            name=name,
            description=description or c.DESCRIPTION,
            approval_mode=approval_mode,
        )
        self.max_output_bytes = max_output_bytes
        # Working directory per (user, conversation), kept by the harness.
        self._cwd: dict[tuple[str, str], str] = {}

    def permission_subjects(self, parameters: dict[str, t.Any]) -> tuple[list[str], bool]:
        """Each command in the line (``ls && rm x`` → ``ls``, ``rm x``), and
        whether the line holds nothing else ($(...), redirections...)."""
        return split_command(parameters.get("command", ""))

    def matches(self, pattern: str, subject: str) -> bool:
        """``git push:*`` matches a command word by word."""
        return command_matches(subject, pattern)

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "minLength": 1,
                    "description": c.COMMAND_DESCRIPTION,
                },
                "description": {
                    "type": "string",
                    "description": c.INTENT_DESCRIPTION,
                },
            },
            "required": ["command", "description"],
            "additionalProperties": False,
        }

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        """Schema and a non-blank command."""
        validation = super().validate_parameters(tool_request)
        if not validation.is_tool_valid:
            return validation
        command = t.cast(str, tool_request.parameters["command"])
        if not command.strip():
            return CoreToolParameters(is_tool_valid=False, msg_error=c.EMPTY_COMMAND)
        return validation

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        """Build the script, run it in the executor, return exit code and output.

        The dispatcher puts the executor and the conversation's session in
        ``tool_context.deps``.
        """
        deps = tool_context.deps if tool_context is not None else {}
        executor: ExecutorBase | None = deps.get("executor")
        session: ExecutionSession | None = deps.get("execution_session")
        if executor is None or session is None:
            return ToolResult.execution_error(tool_request.id, c.NO_ENVIRONMENT)
        validation = self.validate_parameters(tool_request)
        if not validation.is_tool_valid:
            return ToolResult.invalid_parameters(
                tool_request.id, validation.msg_error or c.INVALID_PARAMETERS
            )

        root = PurePosixPath(session.workspace_path)
        workspace = root / "workspace"
        key = (session.user_id, session.conversation_id)
        timeout = setting.tool_timeout_seconds
        script = build_script(
            tool_request.parameters["command"],
            cwd=self._cwd.get(key, str(workspace)),
            workspace=str(workspace),
            skills=str(root / "skills"),
            log=str(PurePosixPath(c.LOG_DIR) / session.conversation_id / f"{short_id()}.log"),
            max_bytes=self.max_output_bytes,
        )
        result = await executor.execute(
            session, script, timeout=timeout, cancellation_token=cancellation_token
        )

        output, mark, cwd = result.stdout.rpartition(c.CWD_MARK)
        if not mark:  # stopped before the trap printed: keep what came out
            output, cwd = result.stdout, ""
        before_files, mark, listing = output.rpartition(c.FILES_MARK)
        if mark:
            output = before_files
        if result.stderr.strip():
            output = f"{output}\n{result.stderr}" if output else result.stderr
        output = output.rstrip("\n")
        exit_code = result.exit_code
        if result.timed_out and exit_code is None:
            exit_code = 124  # killed from outside: report it like `timeout` does
        data: dict[str, t.Any] = {"exit_code": exit_code, "output": output}
        if files := changed_files(listing if mark else ""):
            data["files"] = files
        # Like Claude Code: the shell may wander, but the next command
        # starts back in the workspace unless it ended somewhere allowed.
        here = PurePosixPath(cwd)
        in_tmp = here.is_relative_to("/tmp") and not here.is_relative_to(root)
        if cwd and (in_tmp or here.is_relative_to(workspace) or here.is_relative_to(root / "skills")):
            self._cwd[key] = cwd
        elif cwd:
            self._cwd.pop(key, None)
            data["note"] = c.LEFT_WORKSPACE.format(cwd=cwd)
        if result.timed_out:
            data["note"] = c.TIMED_OUT.format(timeout=timeout)
        return ToolResult.success_result(
            tool_request.id, data, metadata={"name": self.name, "tool_kind": "bash"}
        )
