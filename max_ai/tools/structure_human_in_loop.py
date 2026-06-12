""" 
Human in the Loop Structure Question for User
This is native Tool in the framework which communicate 
the user the assitant questions 
"""
import asyncio
import typing as t

from max_ai.base.tools import ToolContext
from max_ai.termination.cancellation import CancellationToken
from max_ai.types.tool_call import ToolCallRecord, ToolResult 
from ..types.tools import ToolApprovalMode


from ..base.reasoning import BaseLoopState
from ..base.tools import CoreTool

class UserInputTool(CoreTool):
    """Native framework tool - pause the reasoning loop to request user input"""
    TOOL_NAME: t.ClassVar[str] = "structure_human_in_loop"

    def __init__(self, loop_state: BaseLoopState) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=(
                "Pause and ask the user a question before continuing. "
                "Use when you need clarification or a decision from the user "
                "that cannot be inferred from context."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        self._state = loop_state

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to ask the user."},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of choices. Omit for free-text answers.",
                },
            },
            "required": ["question"],
        }
    
    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        question = tool_request.parameters.get("question", "")
        options = tool_request.parameters.get("options")
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._state.pending_user_input = future
        self._state.pending_user_input_question = question
        self._state.pending_user_input_options = options
        self._state.finish_reason = "input_needed"
        answer = await future
        return ToolResult(tool_call_id=tool_request.id, success=True, result=answer)