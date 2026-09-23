"""Framework-level completion blockers — fixed, not pluggable.

Unlike ``CompletionBase`` (the ABC a host subclasses for domain-specific
completion rules), this always runs, the same way, for every agent. It
belongs in ``core/`` rather than ``base/`` for that reason — nothing here
is meant to be overridden.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...base.completion_gate import CompletionDecision

if TYPE_CHECKING:
    from ...types.run_context import RunContext


def runtime_status(
    ctx: RunContext,
    *,
    cancelled: bool = False,
    exhausted_limits: tuple[str, ...] = (),
) -> CompletionDecision | None:
    """Framework-level blockers every gate must respect, checked first.

    Cancellation, exhausted budgets and pending tool calls apply the same
    way regardless of what the agent does — no subclass should have to
    reimplement them. Returns ``None`` when nothing here blocks completion,
    meaning the caller's own domain checks decide from there.
    """
    if cancelled:
        return CompletionDecision(status="cancelled", reasons=("Execution cancelled",))
    if exhausted_limits:
        return CompletionDecision(
            status="incomplete",
            reasons=tuple(f"Limit exhausted: {name}" for name in exhausted_limits),
        )
    pending = tuple(
        f"Pending tool call: {record.id} ({record.status})"
        for record in ctx.tool_state.records.values()
        if not record.is_consumed
    )
    if pending:
        return CompletionDecision(status="waiting", reasons=pending)
    return None
