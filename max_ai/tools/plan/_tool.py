"""
Native self-directed planning tool.

Lets the model build and revise its own execution plan from inside the
ReAct loop. The tool writes an ``AgentPlan`` onto the loop state;
``ReActLoopSelfDirected`` syncs it to ``ctx.plan`` and emits a
``PlanningEvent`` so the UI can render live plan progress.
"""

from __future__ import annotations
import typing as t

from ...base.reasoning import BaseLoopState
from ...base.tools import CoreTool, ToolContext
from ...termination.cancellation import CancellationToken

from ._model import AgentPlan, PlanStep
from ...types.tools import ToolApprovalMode
from ...types.tool_call import ToolCallRecord, ToolResult


class UpdatePlanTool(CoreTool):
    """Native framework tool — let the model set or revise its own plan.

    The plan is under the model's control: it calls ``update_plan`` with
    the full list of steps whenever it wants to lay out or revise its
    approach. The plan is the shared ``AgentPlan`` model, so the UI
    renders one plan format everywhere.
    """

    TOOL_NAME: t.ClassVar[str] = "update_plan"

    def __init__(self, loop_state: BaseLoopState) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=(
                "Create or update your todo list / execution plan, shown "
                "live to the user. For ANY multi-step task: call this first "
                "with the full list of steps (first step 'active', rest "
                "'pending'), then keep it current as you progress — you can "
                "change SEVERAL steps' statuses in one call (e.g. mark "
                "steps 1-2 done and step 3 active together). Always pass "
                "the complete list of steps, not a diff. Keep step ids and "
                "descriptions STABLE across calls — change only statuses; "
                "never renumber or rewrite the plan unless the approach "
                "genuinely changed. Do not call this again if nothing "
                "changed. When you finish the LAST step, "
                "call this marking it done and put your complete final "
                "answer in the same message's text."
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
                            "tool_hint": {
                                "type": "string",
                                "description": (
                                    "Optional: name of the tool you expect "
                                    "to use for this step."
                                ),
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": (
                                    "Optional: ids of steps that must finish "
                                    "before this one."
                                ),
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

        # AgentPlan validates structure (unique ids, single active step,
        # resolvable depends_on). A malformed plan comes back as a tool
        # error the model can read and self-correct from — it never
        # silently replaces a good plan.
        try:
            steps = [
                PlanStep(
                    id=s["id"],
                    description=s["description"],
                    status=s.get("status", "pending"),
                    tool_hint=s.get("tool_hint"),
                    depends_on=list(s.get("depends_on") or []),
                )
                for s in raw_steps
            ]
            plan = AgentPlan(steps=steps, rationale=rationale)
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(
                tool_call_id=tool_request.id,
                success=False,
                error=f"Invalid plan: {e}. Fix the steps and call update_plan again.",
            )

        # No-op detection: models nudged to "keep the plan current" often
        # re-send an identical plan (statuses AND descriptions unchanged).
        # Accept the call but skip the update flag — no PlanningEvent spam
        # for the UI — and tell the model to move on instead of re-calling.
        previous = self._state.plan_draft
        if previous is not None and [
            s.model_dump() for s in previous.steps
        ] == [s.model_dump() for s in plan.steps]:
            return ToolResult(
                tool_call_id=tool_request.id,
                success=True,
                result=(
                    "Plan unchanged — identical to the current plan. Do not "
                    "call update_plan again until a step's status actually "
                    "changes; continue executing the active step."
                ),
            )

        self._state.plan_draft = plan
        self._state.plan_updated = True

        # Replacement detection: a weak model sometimes rewrites the whole
        # plan (new ids, rephrased steps) instead of updating statuses.
        # That silently discards which steps were DONE — and the model then
        # re-executes work it already finished (re-calling the same tools).
        # Accept the rewrite but warn it not to redo completed work.
        if previous is not None:
            prev_ids = {s.id for s in previous.steps}
            new_ids = {s.id for s in plan.steps}
            kept = prev_ids & new_ids
            if prev_ids and len(kept) < (len(prev_ids) + 1) // 2:
                done_before = [
                    s.description for s in previous.steps if s.status == "done"
                ]
                done_note = (
                    " Work already completed: " + "; ".join(done_before) + "."
                    if done_before
                    else ""
                )
                return ToolResult(
                    tool_call_id=tool_request.id,
                    success=True,
                    result=(
                        f"Plan updated with {len(steps)} step(s). WARNING: "
                        "you replaced the previous plan's steps instead of "
                        "updating their statuses. Keep step ids stable and "
                        "only change statuses. Do NOT re-execute steps that "
                        f"were already done.{done_note}"
                    ),
                )

        return ToolResult(
            tool_call_id=tool_request.id,
            success=True,
            result=f"Plan updated with {len(steps)} step(s)",
        )
