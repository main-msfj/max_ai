"""
Native self-directed planning tool.

Lets the model build and revise its own execution plan from inside the
ReAct loop — the planning-as-tool counterpart to ``ReActLoopPlanning``,
where the loop owns the plan. The tool writes an ``AgentPlan`` onto the
loop state; ``ReActLoopSelfDirected`` syncs it to ``ctx.plan`` and emits a
``PlanningEvent`` so the UI sees the same plan object both loops produce.
"""

from __future__ import annotations
import typing as t

from ..base.reasoning import BaseLoopState
from ..base.tools import CoreTool, ToolContext
from ..termination.cancellation import CancellationToken

from ..reasoning.plan import AgentPlan, PlanStep
from ..types.tools import ToolApprovalMode
from ..types.tool_call import ToolCallRecord, ToolResult


class UpdatePlanTool(CoreTool):
    """Native framework tool — let the model set or revise its own plan.

    Unlike ``ReActLoopPlanning`` (where the loop generates and advances the
    plan), this tool puts the plan under the model's control: it calls
    ``update_plan`` with the full list of steps whenever it wants to lay out
    or revise its approach. The plan uses the same ``AgentPlan`` model the
    controller loop uses, so the UI renders one plan format everywhere.
    """

    TOOL_NAME: t.ClassVar[str] = "update_plan"

    def __init__(self, loop_state: BaseLoopState) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=(
                "Set or revise your execution plan. Call this with the full "
                "list of steps whenever you want to lay out a multi-step "
                "approach or update it as you make progress, so the user can "
                "see what you are doing and what is left."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        self._state = loop_state

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "The full, current list of plan steps.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "Unique id for this step.",
                            },
                            "description": {
                                "type": "string",
                                "description": "What this step does.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "active", "done", "failed"],
                                "description": "Current status of this step.",
                            },
                        },
                        "required": ["id", "description", "status"],
                    },
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief reasoning behind the plan.",
                },
            },
            "required": ["steps", "rationale"],
        }

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        raw_steps = tool_request.parameters.get("steps", [])
        rationale = tool_request.parameters.get("rationale", "")

        steps = [
            PlanStep(
                id=s["id"],
                description=s["description"],
                status=s.get("status", "pending"),
            )
            for s in raw_steps
        ]
        self._state.plan_draft = AgentPlan(steps=steps, rationale=rationale)
        self._state.plan_updated = True

        return ToolResult(
            tool_call_id=tool_request.id,
            success=True,
            result=f"Plan updated with {len(steps)} step(s)",
        )
