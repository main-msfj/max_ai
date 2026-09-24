"""Reasoning loops and planning primitives.

The loop classes are lazy-loaded (PEP 562): ``core.event_type`` imports
``reasoning.plan`` at module load, and importing the loop eagerly here
would drag ``base.reasoning`` -> middleware -> ``core.event_type`` back
in while it is still initializing (circular import).
"""

import typing as t

from ..tools.plan import AgentPlan, PlanStep

if t.TYPE_CHECKING:
    from .react import ReactLoop, ReActLoopState

__all__ = [
    "AgentPlan",
    "PlanStep",
    "ReactLoop",
    "ReActLoopState",
    "GuardContext",
    "LoopGuard",
    "SchemaRetryGuard",
    "RepetitionGuard",
    "BudgetGuard",
    "NoProgressGuard",
    "default_guards",
]

_LAZY_LOOP = {"ReactLoop", "ReActLoopState"}
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
    """Resolve a lazily exported attribute.

Parameters
----------
name : str
    Value supplied for ``name``."""
    if name in _LAZY_LOOP:
        from . import react

        return getattr(react, name)
    if name in _LAZY_GUARDS:
        from . import guards

        return getattr(guards, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
