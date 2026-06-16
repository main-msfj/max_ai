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


class EvalConfig(BaseModel):
    """Configuration for the self-evaluation stage of a planning loop.

    The *presence* of an ``EvalConfig`` is what turns self-evaluation on:
    a loop built with ``eval=EvalConfig()`` evaluates, one built with
    ``eval=None`` does not. ``self_eval`` therefore defaults to ``True`` —
    if you bothered to pass a config, you want eval — while the extra
    ``intermediate`` pass stays opt-in.
    """

    self_eval: bool = Field(
        default=True, description="Run the post-loop self-evaluation step"
    )
    intermediate: bool = Field(
        default=False,
        description="Inject a reconsider-prompt when a tool call fails mid-loop",
    )
    threshold: float = Field(
        default=0.8, description="Minimum passing score (fraction of checks passed)"
    )
    max_retries: int = Field(
        default=2, description="How many times to retry the loop on a failed eval"
    )
    criteria: list[str] | None = Field(
        default=None, description="Custom eval criteria; falls back to defaults"
    )
    max_extra_tokens: int | None = Field(
        default=None, description="Token budget cap for eval retries"
    )
