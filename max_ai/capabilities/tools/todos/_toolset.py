"""write_todos: the agent's list of things to do, replaced whole each time."""

from __future__ import annotations

import typing as t

from ....base.tools import CoreTool, CoreToolParameters, ToolContext
from ....core.event_type import TodosEvent
from ....types.run_context import RunContext
from ....types.todos import Todo
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode


class WriteTodosTool(CoreTool):
    """Keeps the list of things to do in ``ctx.todos`` and shows it outside."""

    TOOL_NAME: t.ClassVar[str] = "write_todos"
    DESCRIPTION: t.ClassVar[str] = (
        "Keep a list of the things to do in a task with several steps, shown to "
        "the user. Always send the whole list: it replaces the one before. Mark "
        "a step in_progress when you start it and done when you finish it."
    )
    # Added to the description when the user accepts the plans.
    APPROVAL: t.ClassVar[str] = (
        " A new plan, or one whose steps changed, waits for the user to accept it. "
        "If the user rejects it, propose a different one. If they reject it again, "
        "ask what they would change. After a third rejection, stop proposing and "
        "let the user say how they want it."
    )
    ACCEPTED: t.ClassVar[str] = "The user accepted the plan. Start with its first step."
    REJECTED: t.ClassVar[str] = "The user did not accept the plan."
    STATUSES: t.ClassVar[tuple[str, ...]] = ("pending", "in_progress", "done")

    def __init__(self, approval: bool = False) -> None:
        self.approval = approval  # the user accepts a plan before it counts
        super().__init__(
            name=self.TOOL_NAME,
            description=self.DESCRIPTION + (self.APPROVAL if approval else ""),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Every thing to do, in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "What to do."},
                            "status": {"type": "string", "enum": list(self.STATUSES)},
                        },
                        "required": ["content", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["todos"],
            "additionalProperties": False,
        }

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        # 1. The schema first: after it, every item has its content and status.
        checked = super().validate_parameters(tool_request)
        if not checked.is_tool_valid:
            return checked
        todos = tool_request.parameters["todos"]

        # 2. Each thing to do needs its own text.
        texts = [todo["content"].strip().casefold() for todo in todos]
        if not all(texts):
            return CoreToolParameters(is_tool_valid=False, msg_error="A thing to do has no text.")
        if len(set(texts)) < len(texts):
            return CoreToolParameters(is_tool_valid=False, msg_error="Two things to do are the same.")

        # 3. One thing at a time.
        if sum(todo["status"] == "in_progress" for todo in todos) > 1:
            return CoreToolParameters(
                is_tool_valid=False, msg_error="Only one thing to do can be in_progress."
            )
        return checked

    def waits(self, record: ToolCallRecord, ctx: RunContext) -> bool:
        """Only a list with other steps: marking progress never asks."""
        steps = [todo["content"] for todo in record.parameters["todos"]]
        return self.approval and bool(steps) and steps != [todo.content for todo in ctx.todos]

    @classmethod
    def answer(
        cls, record: ToolCallRecord, ctx: RunContext, accepted: bool, comment: str | None = None
    ) -> ToolResult:
        """What the user decided about a plan that waited, as this call's
        result. Only an accepted plan becomes the list."""
        # 1. Accepted: now it is the list.
        if accepted:
            ctx.todos = [Todo(**todo) for todo in record.parameters["todos"]]
            return ToolResult.success_result(record.id, cls.ACCEPTED)

        # 2. Rejected, with what the user said when they said something.
        said = f" They said: {comment.strip()}" if comment and comment.strip() else ""
        return ToolResult.success_result(record.id, cls.REJECTED + said)

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        todos = [Todo(**todo) for todo in tool_request.parameters["todos"]]
        ctx = tool_ctx.run_context

        # 1. The same list again changes nothing and shows nothing.
        if todos == ctx.todos:
            return ToolResult.success_result(tool_request.id, "The list is the same as before.")

        # 2. Keep it and show it outside.
        ctx.todos = todos
        if tool_ctx.emit_event is not None:
            tool_ctx.emit_event(
                TodosEvent(source=self.name, tool_call_id=tool_request.id, todos=todos)
            )

        # 3. How many are in each state.
        counts = [
            f"{sum(todo.status == status for todo in todos)} {status.replace('_', ' ')}"
            for status in self.STATUSES
        ]
        return ToolResult.success_result(tool_request.id, f"List updated: {', '.join(counts)}.")
