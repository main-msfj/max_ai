"""Reasoning loops and planning primitives.

The loop classes are lazy-loaded (PEP 562): ``core.event_type`` imports
``reasoning.plan`` at module load, and importing the loop eagerly here
would drag ``base.reasoning`` -> middleware -> ``core.event_type`` back
in while it is still initializing (circular import).
"""

import typing as t

from ..capabilities.tools.plan import AgentPlan, PlanStep

if t.TYPE_CHECKING:
    from .react_self_directed import ReActLoopSelfDirected, ReActLoopState

__all__ = [
    "AgentPlan",
    "PlanStep",
    "ReActLoopSelfDirected",
    "ReActLoopState",
    "GuardContext",
    "LoopGuard",
    "SchemaRetryGuard",
    "RepetitionGuard",
    "BudgetGuard",
    "NoProgressGuard",
    "default_guards",
]

_LAZY_LOOP = {"ReActLoopSelfDirected", "ReActLoopState"}
_LAZY_GUARDS = {
    "GuardContext",
    "LoopGuard",
    "SchemaRetryGuard",
    "RepetitionGuard",
    "BudgetGuard",
    "NoProgressGuard",
    "default_guards",
}


def __getattr__(name: str) -> t.Any:
    if name in _LAZY_LOOP:
        from . import react_self_directed

        return getattr(react_self_directed, name)
    if name in _LAZY_GUARDS:
        from . import guards

        return getattr(guards, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
