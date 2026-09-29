"""The bash tool: the model sends only a command string.

The harness wraps it in a script (paths, working directory, output capture)
and the executor runs that script with ``bash -c`` wherever it lives: on the
host, in Docker or in Modal. Nothing of max_ai runs inside the environment.
"""

from __future__ import annotations

import shlex
import typing as t
from pathlib import PurePosixPath

from ....base.tools import CoreRuntimeTool, ToolContext
from ....config import setting
from ....core.ids import short_id
from ....core.termination import CancellationToken
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import CoreToolParameters, ToolApprovalMode
from ._permissions import BashPermission, BashPermissions

if t.TYPE_CHECKING:
    from ....base.executor import ExecutionSession, ExecutorBase

# Printed after the output with the final working directory; the tool
# strips it, so the model never sees it.
_CWD_MARK = "\n__MAXAI_CWD__="
_CWD_MARK_PRINTF = "\\n__MAXAI_CWD__="  # the same, as a printf format


def build_script(
    command: str,
    *,
    cwd: str,
    workspace: str,
    skills: str,
    scratchpad: str,
    log: str,
    max_bytes: int,
) -> str:
    """The script the executor runs for one ``command``.

    It sets WORKSPACE, SKILLS and SCRATCHPAD, starts in ``cwd``, runs the
    command with ``eval`` (stdin closed, stdout and stderr into ``log``),
    then prints the log, cut in the middle past ``max_bytes``, and the final
    working directory. The EXIT trap prints even when the command calls
    ``exit`` or is stopped by the time limit (TERM).
    """
    q = shlex.quote
    half = max_bytes // 2
    return f"""exec 3>&1 4>&2
        export WORKSPACE={q(workspace)} SKILLS={q(skills)} SCRATCHPAD={q(scratchpad)}
        mkdir -p "$WORKSPACE" "$SCRATCHPAD/.bash"
        cd {q(cwd)} 2>/dev/null || cd "$WORKSPACE"
        __maxai_finish() {{
        __maxai_code=$?
        exec 1>&3 2>&4
        __maxai_size=$(wc -c < {q(log)} 2>/dev/null || echo 0)
        if [ "$__maxai_size" -gt {max_bytes} ]; then
            head -c {half} {q(log)}
            printf '\\n\\n[... %s bytes cut; full output in %s ...]\\n\\n' "$((__maxai_size - {2 * half}))" {q(log)}
            tail -c {half} {q(log)}
        else
            cat {q(log)} 2>/dev/null; rm -f {q(log)}
        fi
        printf '{_CWD_MARK_PRINTF}%s' "$(pwd -P)"
        exit $__maxai_code
        }}
        trap __maxai_finish EXIT
        trap 'exit 143' TERM
        {{ eval {q(command)}
        }} < /dev/null > {q(log)} 2>&1
        """


class BashTool(CoreRuntimeTool):
    """Run shell commands in the agent's execution environment.

    The working directory persists between calls of a conversation;
    environment variables do not (every call is a fresh shell). Every call
    stops after ``TOOL_TIMEOUT_SECONDS``.

    BashPermissions classifies commands as allow, ask or deny. Denied commands
    fail validation; per-command approval routing belongs to the dispatcher.
    """

    _DESCRIPTION = (
        "Run a shell command. Commands start in the user's files; the working "
        "directory persists between calls, environment variables do not. "
        "Returns the exit code and the output (stdout and stderr together); "
        "long output is cut in the middle and the full text saved to a file."
    )

    def __init__(
        self,
        name: str = "bash",
        description: str | None = None,
        max_output_bytes: int = 30_000,
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        *,
        allowed_patterns: list[str] | None = None,
        ask_patterns: list[str] | None = None,
        deny_patterns: list[str] | None = None,
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
            Base approval; the permission patterns decide per command.
        allowed_patterns, ask_patterns, deny_patterns : list[str] | None
            Replace the BashPermissions defaults.
        """
        if max_output_bytes < 2:
            raise ValueError("max_output_bytes must be at least 2")
        super().__init__(
            name=name,
            description=description or self._DESCRIPTION,
            approval_mode=approval_mode,
        )
        self.max_output_bytes = max_output_bytes
        self.permissions = BashPermissions(
            **{
                key: value
                for key, value in {
                    "allowed_patterns": allowed_patterns,
                    "ask_patterns": ask_patterns,
                    "deny_patterns": deny_patterns,
                }.items()
                if value is not None
            }
        )
        # Working directory per (user, conversation), kept by the harness.
        self._cwd: dict[tuple[str, str], str] = {}

    def permission_for(self, command: str) -> BashPermission:
        """Return the per-command decision for the approval coordinator."""
        return self.permissions.evaluate(command)

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Shell command to run. Permissions are evaluated across the "
                        "complete compound command; complex shell syntax may require "
                        "approval."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Clear, concise description of what this command does, in plain "
                        "language, so a human deciding whether to approve it understands "
                        "the intent without reading the raw command."
                    ),
                },
            },
            "required": ["command", "description"],
            "additionalProperties": False,
        }

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        """Schema, a non-blank command, and no deny pattern."""
        validation = super().validate_parameters(tool_request)
        if not validation.is_tool_valid:
            return validation
        command = t.cast(str, tool_request.parameters["command"])
        if not command.strip():
            return CoreToolParameters(is_tool_valid=False, msg_error="command cannot be empty.")
        if self.permission_for(command) == "deny":
            return CoreToolParameters(
                is_tool_valid=False,
                msg_error="Command blocked by Bash deny_patterns. This action is not allowed.",
            )
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
            return ToolResult.execution_error(
                tool_request.id, "bash needs an execution environment; run it through an Agent."
            )
        validation = self.validate_parameters(tool_request)
        if not validation.is_tool_valid:
            return ToolResult.invalid_parameters(
                tool_request.id, validation.msg_error or "Invalid bash parameters."
            )

        root = PurePosixPath(session.workspace_path)
        workspace = root / "workspace"
        scratchpad = root / "scratchpad" / session.conversation_id
        key = (session.user_id, session.conversation_id)
        timeout = setting.tool_timeout_seconds
        script = build_script(
            tool_request.parameters["command"],
            cwd=self._cwd.get(key, str(workspace)),
            workspace=str(workspace),
            skills=str(root / "skills"),
            scratchpad=str(scratchpad),
            log=str(scratchpad / ".bash" / f"{short_id()}.log"),
            max_bytes=self.max_output_bytes,
        )
        result = await executor.execute(
            session, script, timeout=timeout, cancellation_token=cancellation_token
        )

        output, mark, cwd = result.stdout.rpartition(_CWD_MARK)
        if not mark:  # stopped before the trap printed: keep what came out
            output, cwd = result.stdout, ""
        if result.stderr.strip():
            output = f"{output}\n{result.stderr}" if output else result.stderr
        output = output.rstrip("\n")
        exit_code = result.exit_code
        if result.timed_out and exit_code is None:
            exit_code = 124  # killed from outside: report it like `timeout` does
        data: dict[str, t.Any] = {"exit_code": exit_code, "output": output}
        if PurePosixPath(cwd).is_relative_to(root):
            self._cwd[key] = cwd
        elif cwd:
            self._cwd.pop(key, None)
            data["note"] = f"{cwd} is outside the user's files; the next command starts in $WORKSPACE."
        if result.timed_out:
            data["note"] = f"Stopped after {timeout:g} seconds, the time limit for a command."
        return ToolResult.success_result(
            tool_request.id, data, metadata={"name": self.name, "tool_kind": "bash"}
        )
