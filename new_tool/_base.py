"""FileTool: what read_file, write_file and edit_file share.

The tool runs in the harness; only its script runs inside the environment.
The harness decides where the path points and whether the call may go on,
and keeps the stamp of every file the model read.
"""

from __future__ import annotations

import json
import posixpath
import shlex
import typing as t
from pathlib import Path, PurePosixPath

from max_ai.base.session_store import validate_id
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.config import setting
from max_ai.core.termination.cancellation import CancellationToken
from max_ai.types.tool_call import ToolCallRecord, ToolResult

_COMMON = Path(__file__).with_name("_common.py").read_text()

PATH_DESCRIPTION = (
    "Path of the file. A relative path starts in your working directory; "
    "an absolute path must be inside your workspace."
)


def script_of(tool_file: str) -> str:
    """The code sent to the environment for a tool: ``_common.py`` followed
    by the ``script.py`` next to ``tool_file``. Read once, when imported."""
    return _COMMON + "\n" + Path(tool_file).with_name("script.py").read_text()


class Place(t.NamedTuple):
    """Where a path of the model points."""

    full: str  # absolute, inside the environment
    other_session: bool  # in the workspace, but not this session's folder
    roots: list[str]  # the folders the script may not leave


def locate(sandbox: t.Any, session_id: str, path: str) -> Place:
    """Turn the model's ``path`` into a real one and say where it falls.

    This session's folder and its tmp folder are home. The rest of the
    workspace belongs to other sessions. Anything else is refused.
    """
    # The id becomes a folder name: ".." would leave the workspace.
    validate_id("session_id", session_id)
    workspace = PurePosixPath(str(sandbox.workspace))
    session = workspace / session_id
    tmp = PurePosixPath(str(sandbox.tmp)) / session_id
    # Joining with an absolute path keeps the absolute one.
    full = PurePosixPath(posixpath.normpath(session / path))
    roots = [str(workspace), str(tmp)]
    if full.is_relative_to(session) or full.is_relative_to(tmp):
        return Place(str(full), False, roots)
    if full.is_relative_to(workspace):
        return Place(str(full), True, roots)
    raise ValueError(
        f"{path} is outside your workspace. Work inside {session} (relative "
        f"paths start there); temporary files go in {tmp}."
    )


class FileTool(CoreTool):
    """A file tool: its ``SCRIPT`` does the work inside the environment.

    ``tool_context.deps`` must carry ``sandbox`` and ``file_stamps`` (one
    dict the Agent owns: what each session last read of each file).
    """

    SCRIPT: t.ClassVar[str]

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        call_id = tool_request.id
        parameters = tool_request.parameters
        sandbox = tool_context.deps.get("sandbox") if tool_context else None
        if tool_context is None or sandbox is None:
            return ToolResult.tool_failure(call_id, f"{self.name} needs a sandbox")
        try:
            place = locate(sandbox, tool_context.session_id, parameters["path"])
        except ValueError as error:
            return ToolResult.tool_failure(call_id, str(error))
        # Until the dispatcher can ask the user, another session is read-only.
        if place.other_session and not self.read_only:
            return ToolResult.tool_failure(
                call_id,
                f"{parameters['path']} belongs to another session: you can read it, not change it",
            )

        # Missing "file_stamps" means every edit answers "read it first".
        stamps = tool_context.deps.get("file_stamps", {})
        key = (tool_context.session_id, place.full)
        request = {
            **parameters,
            "path": place.full,
            "roots": place.roots,
            "stamp": stamps.get(key),
        }
        timeout = setting.tool_timeout_seconds
        result = await sandbox.execute_command(
            f"python3 -c {shlex.quote(self.SCRIPT)}",
            timeout=timeout,
            stdin=json.dumps(request).encode(),
        )

        if result.timed_out:
            return ToolResult.timeout(call_id, timeout)
        if result.exit_code != 0:
            return ToolResult.execution_error(call_id, result.stderr)
        answer = json.loads(result.stdout)
        if not answer.pop("ok"):
            return ToolResult.tool_failure(call_id, f"{parameters['path']}: {answer['error']}")
        # The stamp stays in the harness; the model never sees it.
        if "stamp" in answer:
            stamps[key] = answer.pop("stamp")
        return ToolResult.success_result(call_id, answer)
