"""enter_worktree and exit_worktree: a git worktree for the session, inside the sandbox."""

from __future__ import annotations

import shlex
import typing as t

from ....base.sandbox import SandboxCore
from ....base.tools import CoreTool, ToolContext
from ....types.run_context import RunContext
from ....types.tool_call import ToolCallRecord, ToolResult

FOLDER = ".worktrees"  # inside the session's workspace, so it stays an allowed folder
NO_SANDBOX = "This tool needs a sandbox."


class EnterWorktreeTool(CoreTool):
    """Creates a git worktree on a new branch and moves the agent into it."""

    TOOL_NAME: t.ClassVar[str] = "enter_worktree"
    DESCRIPTION: t.ClassVar[str] = (
        "Create a separate copy of the repository on a new branch and work there, "
        "so the files of the main folder are not touched. Commands and relative "
        "paths start in the copy until you call exit_worktree."
    )
    NOT_A_REPOSITORY: t.ClassVar[str] = (
        "{folder} is not a git repository with a commit; a worktree needs one."
    )

    sandbox = True  # git runs where the files are

    def __init__(self) -> None:
        super().__init__(name=self.TOOL_NAME, description=self.DESCRIPTION)

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "pattern": "^[a-z0-9][a-z0-9-]{0,40}$",
                    "description": "Name of the worktree and of its new branch.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        box, ctx = tool_ctx.sandbox, tool_ctx.run_context
        if box is None:
            return ToolResult.tool_failure(tool_request.id, NO_SANDBOX)
        # 1. The repository is the session's workspace; it needs a commit to branch from.
        workspace = self.workspace(box, ctx)
        here = shlex.quote(workspace)
        found = await box.execute_command(f"cd {here} && git rev-parse --verify HEAD")
        if found.exit_code != 0:
            text = self.NOT_A_REPOSITORY.format(folder=workspace)
            return ToolResult.tool_failure(tool_request.id, text)
        # 2. Create the worktree on its new branch; git says why when it cannot.
        name = tool_request.parameters["name"]
        target = f"{workspace}/{FOLDER}/{name}"
        made = await box.execute_command(
            f"cd {here} && git worktree add -b {shlex.quote(name)} {shlex.quote(target)}"
        )
        if made.exit_code != 0:
            return ToolResult.tool_failure(tool_request.id, (made.stderr or made.stdout).strip())
        # 3. The agent is there now: bash and the file tools start from it.
        ctx.cwd = target
        return ToolResult.success_result(
            tool_request.id, f"You are now in {target}, on the new branch {name}."
        )

    @staticmethod
    def workspace(box: SandboxCore, ctx: RunContext) -> str:
        """The session's working directory in the sandbox."""
        return box.session_paths(ctx.session_id or ctx.run_id)["workspace"]


class ExitWorktreeTool(CoreTool):
    """Takes the agent back to the workspace; keeps or removes the worktree."""

    TOOL_NAME: t.ClassVar[str] = "exit_worktree"
    DESCRIPTION: t.ClassVar[str] = (
        "Leave the worktree and go back to the main folder. 'keep' leaves the "
        "worktree and its branch; 'remove' deletes the worktree and fails if it "
        "has changes that are not committed."
    )
    NOT_IN_ONE: t.ClassVar[str] = "You are not in a worktree."

    sandbox = True  # git runs where the files are

    def __init__(self) -> None:
        super().__init__(name=self.TOOL_NAME, description=self.DESCRIPTION)

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["keep", "remove"],
                    "description": "What to do with the worktree you leave.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        box, ctx = tool_ctx.sandbox, tool_ctx.run_context
        if box is None:
            return ToolResult.tool_failure(tool_request.id, NO_SANDBOX)
        # 1. Where the agent is must be a worktree made by enter_worktree.
        workspace = EnterWorktreeTool.workspace(box, ctx)
        home = f"{workspace}/{FOLDER}/"
        target = ctx.cwd or ""
        if not target.startswith(home):
            return ToolResult.tool_failure(tool_request.id, self.NOT_IN_ONE)
        # 2. The worktree is the first folder under it, wherever inside the agent is.
        target = home + target[len(home):].split("/")[0]
        # 3. Remove only when asked; without --force git refuses to lose changes.
        if tool_request.parameters["action"] == "remove":
            gone = await box.execute_command(
                f"cd {shlex.quote(workspace)} && git worktree remove {shlex.quote(target)}"
            )
            if gone.exit_code != 0:
                error = (gone.stderr or gone.stdout).strip()
                return ToolResult.tool_failure(tool_request.id, error)
        # 4. Back to the workspace.
        ctx.cwd = None
        return ToolResult.success_result(tool_request.id, f"You are back in {workspace}.")
