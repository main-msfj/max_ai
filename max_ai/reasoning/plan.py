"""
Implementation of the Plan class,
which represents a sequence of actions to achieve a goal.
"""

import typing as t
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """Respresents a single step in a plan"""

    id: int = Field(..., description="Unique identifier for the plan step")
    description: str = Field(..., description="Description of the action to be taken")
    tool_hint: str | None = Field(
        None, description="Optional hint for which tool to use for this step"
    )
    depends_on: t.List[int] = Field(
        default_factory=list,
        description="List of plan step IDs that this step depends on",
    )
    status: t.Literal["pending", "active", "done", "failed"] = Field(
        default="pending",
        description="Lifecycle state of this step",
    )


class AgentPlan(BaseModel):
    """Represents a plan consisting of multiple step to achieve a goal"""

    steps: t.List[PlanStep] = Field(
        default_factory=list, description="List of steps in the plan"
    )
    rationale: str = Field(..., description="Rationale behind the plan")

    def active_step(self) -> PlanStep | None:
        """Accumulative Step in the run"""
        return next((s for s in self.steps if s.status == "active"), None)
    
    def next_pending(self) -> PlanStep | None:
         return next((s for s in self.steps if s.status == "pending"), None) 
    
    def mark_done(self, step_id: int) -> None:
        """Mark as completed by id"""
        for s in self.steps:
            if s.id == step_id:
                s.status = "done"
                return 
