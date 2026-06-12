"""
This Module Represent the Evaluation of the Reasoning Process in Max AI.
It Provides Tools and Metrics to Assess the Performance and Effectiveness of the Reasoning Engine, Ensuring that it Meets the Desired Standards and Objectives.
"""

import typing as t
from pydantic import BaseModel, Field


class EvalCheck(BaseModel):
    criterion: str = Field(description="The criterion being evaluated")
    passed: bool = Field(description="Whether the criterion was met")
    reason: str = Field(description="Why it passed or failed")


class EvalResult(BaseModel):
    """Represents the result of an evaluation of the reasoning process"""

    checks: list[EvalCheck] = Field(description="One check per criteriont")
    issues: t.List[str] = Field(
        default_factory=list, description="Specific problems found"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Concrete improvements"
    )
    summary: str = Field(description="One sentence summary of the evaluation")
