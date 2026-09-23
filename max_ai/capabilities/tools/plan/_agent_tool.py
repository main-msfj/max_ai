"""Native update_plan tool — writes straight to RunContext.plan.

This tool expects the caller (``Agent``) to put the live ``RunContext`` on
``ToolContext.deps["run_context"]`` before dispatching — it reads the
previous plan from ``ctx.plan`` and writes the new one back there directly,
emitting ``PlanningEvent`` itself via ``ToolContext.emit_event`` instead of
relying on an outer loop to notice the change.
"""

from __future__ import annotations

import typing as t

from ....base.tools import CoreTool, ToolContext
from ....core.event_type import PlanningEvent
from ....core.termination.cancellation import CancellationToken
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode
from ._model import AgentPlan, PlanStep


class AgentUpdatePlanTool(CoreTool):
    """Native Agent tool — set or revise the plan directly on RunContext."""

    TOOL_NAME: t.ClassVar[str] = "update_plan"

    def __init__(self) -> None:
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
        if tool_context is None or "run_context" not in tool_context.deps:
            return ToolResult(
                tool_call_id=tool_request.id,
                success=False,
                error=(
                    "update_plan requires a RunContext at "
                    "ToolContext.deps['run_context'] — the caller must set "
                    "it before dispatching this tool."
                ),
            )
        ctx = tool_context.deps["run_context"]

        raw_steps = tool_request.parameters.get("steps", [])
        rationale = tool_request.parameters.get("rationale", "")

        # AgentPlan validates structure; a malformed plan comes back as a
        # tool error the model can self-correct from, never a silent replace.
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

        # No-op: an identical resent plan skips the update (no event spam)
        # and tells the model to move on instead of re-calling.
        previous = ctx.plan
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

        ctx.plan = plan
        if tool_context.emit_event is not None:
            tool_context.emit_event(
                PlanningEvent(source=self.name, phase="progress", plan=plan)
            )

        # A rewritten plan (new ids) silently drops which steps were done —
        # accept it but warn the model not to redo finished work.
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
