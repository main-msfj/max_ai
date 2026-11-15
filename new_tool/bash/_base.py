"""bash on the sandbox.

Everything the model sees (parameters, Policy matching, output format) is
the BashTool that already exists; only where the command runs changes:
``sandbox.execute_command`` instead of the executor and its session. The
wrapper script is the same ``wrapper.sh``.
"""

from __future__ import annotations

import shlex
import typing as t
from pathlib import PurePosixPath

from max_ai.base.session_store import validate_id
from max_ai.base.tools import ToolContext
from max_ai.capabilities.tools.bash import constant as c
from max_ai.capabilities.tools.bash._tool import BashTool, build_script, changed_files
from max_ai.config import setting
from max_ai.core.ids import short_id
from max_ai.core.termination import CancellationToken
from max_ai.types.tool_call import ToolCallRecord, ToolResult

LEFT_WORKSPACE = "Shell cwd was reset: {cwd} is outside your workspace, so the next command starts in {session}."


class Bash(BashTool):
    """Run shell commands in the sandbox, starting in the session's folder.

    ``tool_context.deps`` must carry ``sandbox`` and ``bash_cwd`` (one dict
    the Agent owns: where each session's shell stands).
    """

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        call_id = tool_request.id
        sandbox = tool_context.deps.get("sandbox") if tool_context else None
        if tool_context is None or sandbox is None:
            return ToolResult.tool_failure(call_id, "bash needs a sandbox")
        try:
            session_id = validate_id("session_id", tool_context.session_id)
        except ValueError as error:
            return ToolResult.tool_failure(call_id, str(error))

        workspace = PurePosixPath(str(sandbox.workspace))
        session = workspace / session_id
        tmp = PurePosixPath(str(sandbox.tmp)) / session_id
        cwds: dict[str, str] = tool_context.deps.get("bash_cwd", {})
        timeout = setting.tool_timeout_seconds
        # TMPDIR: mktemp and most programs put their temporary files there.
        script = f"export TMPDIR={shlex.quote(str(tmp))}\n" + build_script(
            tool_request.parameters["command"],
            cwd=cwds.get(session_id, str(session)),
            workspace=str(session),
            skills=str(sandbox.skills),
            log=str(tmp / f"bash-{short_id()}.log"),
            max_bytes=self.max_output_bytes,
        )
        result = await sandbox.execute_command(script, timeout=timeout)

        # From here on: the same reading of the output as BashTool.
        output, mark, cwd = result.stdout.rpartition(c.CWD_MARK)
        if not mark:  # stopped before the trap printed: keep what came out
            output, cwd = result.stdout, ""
        before_files, mark, listing = output.rpartition(c.FILES_MARK)
        if mark:
            output = before_files
        if result.stderr.strip():
            output = f"{output}\n{result.stderr}" if output else result.stderr
        exit_code = result.exit_code
        if result.timed_out and exit_code is None:
            exit_code = 124  # killed from outside: report it like `timeout` does
        data: dict[str, t.Any] = {"exit_code": exit_code, "output": output.rstrip("\n")}
        if files := changed_files(listing if mark else ""):
            data["files"] = files
        # The shell may stand anywhere in the workspace (other sessions too)
        # or in this session's tmp; anywhere else, the next command starts home.
        here = PurePosixPath(cwd)
        if cwd and (here.is_relative_to(workspace) or here.is_relative_to(tmp)):
            cwds[session_id] = cwd
        elif cwd:
            cwds.pop(session_id, None)
            data["note"] = LEFT_WORKSPACE.format(cwd=cwd, session=session)
        if result.timed_out:
            data["note"] = c.TIMED_OUT.format(timeout=timeout)
        return ToolResult.success_result(
            call_id, data, metadata={"name": self.name, "tool_kind": "bash"}
        )
