"""task_output and stop_task: the commands bash left running in the background."""

from __future__ import annotations

import typing as t

from ....base.sandbox import TaskState, TaskStatus
from ....base.tools import CoreTool, ToolContext
from ....errors.sandbox import SandboxError
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode


class TaskOutputTool(CoreTool):
    """Shows the background tasks of the session, or one with what it printed."""

    TOOL_NAME: t.ClassVar[str] = "task_output"
    DESCRIPTION: t.ClassVar[str] = (
        "Look at the commands started with bash in the background. With a task "
        "id: how that task is and the last lines it printed. Without one: the "
        "list of tasks. You are told when a task ends, so do not call this in a "
        "loop to wait."
    )
    LINES: t.ClassVar[int] = 100  # of the end of the output, for one task
    COMMAND_CHARS: t.ClassVar[int] = 80  # of a command, in the list
    NONE: t.ClassVar[str] = "There are no background tasks in this session."
    # How a task that did not end by itself is told to the model.
    STATES: t.ClassVar[dict[TaskStatus, str]] = {
        TaskStatus.CAP: "stopped at the time limit",
        TaskStatus.LOST: "lost: it is not running and never said how it ended",
    }

    sandbox = True  # the tasks live in the sandbox

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
                "task_id": {
                    "type": "string",
                    "description": "The task to look at. Leave it out for the list.",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        task_id: str | None = tool_request.parameters.get("task_id")
        sandbox, ctx = tool_ctx.sandbox, tool_ctx.run_context
        if sandbox is None:
            return ToolResult.execution_error(tool_request.id, "task_output needs a sandbox.")

        # 1. One task with its last lines, or all of them without their output.
        try:
            tasks = await sandbox.tasks(
                tool_ctx.session_id, task_id, lines=self.LINES if task_id else 0
            )
        except (SandboxError, ValueError) as error:
            return ToolResult.execution_error(tool_request.id, str(error))

        # 2. The ones that ended leave the context: the model has seen them now.
        for task in tasks:
            if task.state is not TaskStatus.RUNNING:
                ctx.tasks.pop(task.id, None)

        # 3. One task whole, or one line for each.
        if task_id:
            text = f"{self.state(tasks[0])}\n{tasks[0].output or '(it printed nothing)'}"
        else:
            text = "\n".join(self.line(task) for task in tasks) or self.NONE
        return ToolResult.success_result(tool_request.id, text)

    @classmethod
    def state(cls, task: TaskState) -> str:
        """How a task is, in words: ``task 2: done, exit code 0``."""
        said = cls.STATES.get(task.state, task.state.value)
        code = f", exit code {task.code}" if task.code is not None else ""
        return f"task {task.id}: {said}{code}"

    @classmethod
    def line(cls, task: TaskState) -> str:
        """A task in the list: how it is and the start of its command."""
        command = " ".join(task.command.split())
        if len(command) > cls.COMMAND_CHARS:
            command = command[: cls.COMMAND_CHARS] + "..."
        return f"{cls.state(task)} | {command}"


class StopTaskTool(CoreTool):
    """Stops one background task of the session and all it started."""

    TOOL_NAME: t.ClassVar[str] = "stop_task"
    DESCRIPTION: t.ClassVar[str] = (
        "Stop a command started with bash in the background, and everything it "
        "started. What it printed stays readable with task_output."
    )
    ENDED: t.ClassVar[str] = "It had already ended, nothing was stopped. {state}"

    sandbox = True  # the tasks live in the sandbox

    def __init__(self) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=self.DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "The task to stop."},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        task_id: str = tool_request.parameters["task_id"]
        sandbox, ctx = tool_ctx.sandbox, tool_ctx.run_context
        if sandbox is None:
            return ToolResult.execution_error(tool_request.id, "stop_task needs a sandbox.")

        # 1. Only a task that still runs is stopped; then read how it is left.
        try:
            (before,) = await sandbox.tasks(tool_ctx.session_id, task_id, lines=0)
            if before.state is TaskStatus.RUNNING:
                await sandbox.stop_task(tool_ctx.session_id, task_id)
            (task,) = await sandbox.tasks(tool_ctx.session_id, task_id, lines=0)
        except (SandboxError, ValueError) as error:
            return ToolResult.execution_error(tool_request.id, str(error))

        # 2. It no longer runs: the context forgets it.
        ctx.tasks.pop(task_id, None)

        # 3. Stopped now, or it had ended before by itself.
        state = TaskOutputTool.state(task)
        text = state if before.state is TaskStatus.RUNNING else self.ENDED.format(state=state)
        return ToolResult.success_result(tool_request.id, text)
