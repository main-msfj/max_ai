"""
native scratchpad tool for agent task tracking
"""

from __future__ import annotations
import typing as t

from ..base.reasoning import BaseLoopState
from ..base.tools import CoreTool, ToolContext
from ..termination.cancellation import CancellationToken

from ..types.tools import ToolApprovalMode
from ..types.tool_call import ToolCallRecord, ToolResult


class ScratchpadTool(CoreTool):
    """Native framework tool - update the agent's tasks list with pausing"""

    TOOL_NAME: t.ClassVar[str] = "update_todo"

    def __init__(self, loop_state: BaseLoopState) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=(
                "Update your task list. Call this to track progress on "
                "multi-step tasks so the user can see what you are doing."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        self._state = loop_state

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "short nique id for this task, e.g 'search_competitors'.",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable task description.",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "done", "skipped"],
                    "description": "Current status of the task.",
                },
            },
            "required": ["id", "description", "status"],
        }

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        id_ = tool_request.parameters.get("id", "")
        description = tool_request.parameters.get("description", "")
        status = tool_request.parameters.get("status", "pending")
        self._state.scratchpad.upsert(id=id_, description=description, status=status)
        self._state.scratchpad_updated = True
        return ToolResult(
            tool_call_id=tool_request.id,
            success=True,
            result=f"Task '{id_}' updated to '{status}'",
        )
