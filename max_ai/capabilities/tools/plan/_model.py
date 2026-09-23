"""
Implementation of the Plan class,
which represents a sequence of actions to achieve a goal.
"""

import typing as t

from pydantic import BaseModel, Field, model_validator


class PlanStep(BaseModel):
    """Represents a single step in a plan"""

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
    """Represents a plan consisting of multiple steps to achieve a goal.

    Validated on construction so a malformed plan coming from the model
    (via the ``update_plan`` tool or structured output) fails with a
    message the model can self-correct from:

    - step ids must be unique;
    - at most one step may be ``active``;
    - every ``depends_on`` reference must point at an existing step.
    """

    steps: t.List[PlanStep] = Field(
        default_factory=list, description="List of steps in the plan"
    )
    rationale: str = Field(..., description="Rationale behind the plan")

    @model_validator(mode="after")
    def _validate_plan(self) -> "AgentPlan":
        ids = [s.id for s in self.steps]
        duplicate = {i for i in ids if ids.count(i) > 1}
        if duplicate:
            raise ValueError(
                f"Plan step ids must be unique; duplicated: {sorted(duplicate)}"
            )

        active = [s.id for s in self.steps if s.status == "active"]
        if len(active) > 1:
            raise ValueError(
                f"At most one step may be 'active'; got steps {active}. "
                "Mark the others 'pending' or 'done'."
            )

        known = set(ids)
        for s in self.steps:
            missing = [d for d in s.depends_on if d not in known]
            if missing:
                raise ValueError(
                    f"Step {s.id} depends on unknown step ids {missing}."
                )
        return self

    def as_text(self) -> str:
        """One line per step, for prompts: ``[active] 2. Write the scraper``."""
        return "\n".join(f"[{s.status}] {s.id}. {s.description}" for s in self.steps)

    def has_unfinished_steps(self) -> bool:
        """True if any step is still pending or active (work remains)."""
        return any(s.status in ("pending", "active") for s in self.steps)

    def active_step(self) -> PlanStep | None:
        """The step currently in progress, or None."""
        return next((s for s in self.steps if s.status == "active"), None)

    def next_pending(self) -> PlanStep | None:
        """The first pending step (the next to run)."""
        return next((s for s in self.steps if s.status == "pending"), None)

    def mark_done(self, step_id: int) -> None:
        """Mark as completed by id"""
        for s in self.steps:
            if s.id == step_id:
                s.status = "done"
                return

    def mark_failed(self, step_id: int) -> None:
        """Mark a step as failed (could not be completed)"""
        for s in self.steps:
            if s.id == step_id:
                s.status = "failed"
                return
