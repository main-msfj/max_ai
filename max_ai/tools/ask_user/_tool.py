"""
Human-in-the-loop: the native ask-the-user tool.

The tool is a *declaration*, not an implementation: the framework's
``ToolDispatcher`` intercepts calls to it and never invokes ``execute()``.
A fresh call becomes a ``ToolCallRecord`` in ``INPUT_NEEDED`` (carrying
the question and options) plus a ``UserInputRequestEvent``; the turn
ends with ``finish_reason="input_needed"``. The consumer answers via
``ctx.tool_state.apply_user_answer(tool_call_id, answer)`` and resumes —
the executor then completes the call from the stored answer.

Because the pause is record state (like approvals), a pending question
serializes with the ``RunContext`` and survives process death.
"""

import typing as t

from ...base.tools import CoreTool, ToolContext
from ...termination.cancellation import CancellationToken
from ...types.tool_call import ToolCallRecord, ToolResult
from ...types.tools import ToolApprovalMode


class AskUserTool(CoreTool):
    """Native framework tool — pause the run to ask the user a question."""

    TOOL_NAME: t.ClassVar[str] = "ask_user"
    # The tool's name before the rename. The executor still intercepts
    # records persisted under this name so paused runs resume cleanly.
    LEGACY_TOOL_NAMES: t.ClassVar[frozenset[str]] = frozenset(
        {"structure_human_in_loop"}
    )

    def __init__(self) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=(
                "Ask the user a question and wait for their answer. This is "
                "the ONLY way to ask the user something mid-task: it pauses "
                "the run and shows an interactive question card in the UI. "
                "Use it whenever you need clarification, a decision, or a "
                "preference that cannot be inferred from context — never ask "
                "questions in your plain-text reply. If you have SEVERAL "
                "independent questions, call this tool once per question in "
                "the SAME response — the user sees them together as one form "
                "and you get all answers in one round, instead of one slow "
                "back-and-forth per question."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": (
                                    "Short choice label (1-5 words), shown as "
                                    "the button title."
                                ),
                            },
                            "description": {
                                "type": "string",
                                "description": (
                                    "What this choice means or its trade-off, "
                                    "shown as the button subtitle."
                                ),
                            },
                        },
                        "required": ["label", "description"],
                        "additionalProperties": False,
                    },
                    "description": (
                        "Optional list of 2-4 choices rendered as clickable "
                        "buttons, each with a short label and a description "
                        "of its trade-off. Omit for free-text answers."
                    ),
                },
                "multiSelect": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Whether the user may pick more than one option at "
                        "once. Declared for forward compatibility only: the "
                        "executor and UI do not honor it yet, so treat every "
                        "answer as a single value until that follow-up lands."
                    ),
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
        """Fallback for custom executors that bypass the framework's
        short-circuit: complete from a stored answer if one exists,
        otherwise report that the question is still pending. Never blocks.
        """
        if tool_request.user_answer is not None:
            return ToolResult.success_result(tool_request.id, tool_request.user_answer)
        return ToolResult.tool_failure(
            tool_request.id,
            error=(
                "User input is pending. This tool is resolved by the framework "
                "executor via ToolState.apply_user_answer(); it cannot be "
                "executed directly without an answer."
            ),
        )
