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

from ....base.tools import CoreTool, CoreToolParameters, ToolContext
from ....core.termination.cancellation import CancellationToken
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode

_OPTION_SCHEMA = {
    "type": "array",
    "minItems": 2,
    "maxItems": 4,
    "items": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "Short choice label (1-5 words).",
            },
            "description": {
                "type": "string",
                "description": "What this choice means or its trade-off.",
            },
        },
        "required": ["label", "description"],
        "additionalProperties": False,
    },
    "description": (
        "2-4 choices, each a short label plus its trade-off. The user can "
        "always type their own answer instead, so never add an 'Other' "
        "choice. Omit for a purely free-text question."
    ),
}


def _display_options(raw: t.Any) -> list[str] | None:
    """Options as the "label — description" strings records carry."""
    if not isinstance(raw, list):
        return None
    shown: list[str] = []
    for option in raw:
        if isinstance(option, str) and option.strip():
            shown.append(option.strip())
        elif isinstance(option, dict):
            label = option.get("label")
            if isinstance(label, str) and label.strip():
                text = label.strip()
                description = option.get("description")
                if isinstance(description, str) and description.strip():
                    text = f"{text} — {description.strip()}"
                shown.append(text)
    return shown or None


def pending_questions(parameters: dict[str, t.Any]) -> list[dict[str, t.Any]]:
    """Normalize an ask_user call into ``[{question, header, options}]``.

    Accepts the current ``questions`` list and the legacy single
    ``question``/``options`` form (records persisted before the change).
    """
    raw = parameters.get("questions")
    if not isinstance(raw, list):
        raw = [{"question": parameters.get("question"), "options": parameters.get("options")}]
    questions: list[dict[str, t.Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("question") or "").strip()
        if not text:
            continue
        header = item.get("header")
        questions.append({
            "question": text,
            "header": header.strip() if isinstance(header, str) and header.strip() else None,
            "options": _display_options(item.get("options")),
        })
    return questions


class AskUserTool(CoreTool):
    """Native framework tool — pause the run to ask the user questions."""

    TOOL_NAME: t.ClassVar[str] = "ask_user"
    # The tool's name before the rename. The executor still intercepts
    # records persisted under this name so paused runs resume cleanly.
    LEGACY_TOOL_NAMES: t.ClassVar[frozenset[str]] = frozenset(
        {"structure_human_in_loop"}
    )

    def __init__(self) -> None:
        """Initialize ``AskUserTool``."""
        super().__init__(
            name=self.TOOL_NAME,
            description=(
                "Ask the user 1-4 questions and wait for the answers. This is "
                "the ONLY way to ask the user something mid-task: it pauses "
                "the run and shows ONE interactive form with every question. "
                "Use it whenever you need clarification, a decision, or a "
                "preference that cannot be inferred from context — never ask "
                "questions in your plain-text reply. Put ALL the questions "
                "you need right now in a single call (the `questions` list), "
                "never one call per question: the user answers them together "
                "and you get every answer in one round."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        """Perform the ``parameters`` operation for ``AskUserTool``."""
        return {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "description": "Every question to ask now, in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The full question, ending with '?'.",
                            },
                            "header": {
                                "type": "string",
                                "description": (
                                    "Very short tab label (max 12 chars), "
                                    "e.g. 'Name', 'Format', 'Detail'."
                                ),
                            },
                            "options": _OPTION_SCHEMA,
                        },
                        "required": ["question", "header"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["questions"],
        }

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        # Records persisted with the legacy single-question shape are
        # rewritten to the current one so they still validate on resume.
        """Validate parameters for ``AskUserTool``.

Parameters
----------
tool_request : ToolCallRecord
    Value supplied for ``tool_request``."""
        params = tool_request.parameters
        if "questions" not in params and "question" in params:
            legacy: dict[str, t.Any] = {"question": params["question"], "header": "Question"}
            if params.get("options"):
                legacy["options"] = params["options"]
            tool_request.parameters = {"questions": [legacy]}
        return super().validate_parameters(tool_request)

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
        if tool_request.user_answers is not None:
            return ToolResult.success_result(tool_request.id, {"answers": tool_request.user_answers})
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
