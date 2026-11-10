"""
Human-in-the-loop: the native ask-the-user tool.

A tool like any other, except that a person produces its result. The
dispatcher pauses the turn; the consumer answers with
``ctx.tool_state.provide_result(record.id, AskUserTool.answer(record, answers))``
and resumes. The pause is record state: it survives process death.
"""

import typing as t

from ....base.tools import CoreTool, CoreToolParameters, ToolContext
from ....core.harness import messages as harness
from ....core.termination.cancellation import CancellationToken
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode


class AskUserTool(CoreTool):
    """Native framework tool — pause the run to ask the user questions."""

    TOOL_NAME: t.ClassVar[str] = "ask_user"
    waits_for_user = True
    _OPTION_SCHEMA: t.ClassVar[dict[str, t.Any]] = {
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
            "required": ["label"],
        },
        "description": (
            "2-4 choices, each a short label plus its trade-off. The user can "
            "always type their own answer instead, so never add an 'Other' "
            "choice. Omit for a purely free-text question."
        ),
    }

    def __init__(self) -> None:
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

    @staticmethod
    def answer(record: ToolCallRecord, answers: dict[str, str | list[str]]) -> ToolResult:
        """A person's answers, as this call's result: one per question."""
        return ToolResult.success_result(record.id, {"answers": {
            q["question"]: answers.get(q["question"], harness.QUESTION_NOT_ANSWERED)
            for q in record.parameters["questions"]
        }})

    @property
    def parameters(self) -> dict[str, t.Any]:
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
                            "options": self._OPTION_SCHEMA,
                            "multi_select": {
                                "type": "boolean",
                                "description": (
                                    "True when several options can apply together, "
                                    "e.g. 'Which files should I delete?'. Leave it "
                                    "out when only one can be right, e.g. 'Which "
                                    "database should we use?'."
                                ),
                            },
                        },
                        "required": ["question"],
                    },
                },
            },
            "required": ["questions"],
        }

    @staticmethod
    def _empty_or_repeated(kind: str, texts: list[str]) -> str | None:
        """Why ``texts`` can't be used, or None. Spaces and case don't count."""
        cleaned = [text.strip().casefold() for text in texts]
        if not all(cleaned):
            return f"Every {kind} needs text: one of them is empty."
        # Answers are keyed by question and given by label: repeats are ambiguous.
        if len(set(cleaned)) < len(cleaned):
            return f"Two {kind}s are the same: each one must be different."
        return None

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        # The schema runs first, so these fields exist here. On top of it:
        # no empty or repeated questions, nor options within a question.
        checked = super().validate_parameters(tool_request)
        if not checked.is_tool_valid:
            return checked
        questions = tool_request.parameters["questions"]
        error = self._empty_or_repeated("question", [q["question"] for q in questions])
        for question in questions:
            labels = [option["label"] for option in question.get("options") or []]
            error = error or self._empty_or_repeated("option", labels)
        if error:
            return CoreToolParameters(is_tool_valid=False, msg_error=error)
        return checked

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        """Never reached: a person produces this tool's result."""
        return ToolResult.execution_error(
            tool_request.id, f"{self.TOOL_NAME} is answered by a person"
        )


