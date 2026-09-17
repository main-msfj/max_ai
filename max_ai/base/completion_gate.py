"""Harness-owned completion decisions, independent of plans and publication."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from ..types.run_context import RunContext


class CompletionCheck(BaseModel):
    """Evidence returned by trusted application code, never model arguments."""

    model_config = ConfigDict(frozen=True)
    status: Literal["pass", "fail", "unknown"]
    reason: str = ""


class CompletionDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Literal["completed", "incomplete", "waiting", "cancelled"]
    reasons: tuple[str, ...] = ()


class CompletionGate:
    """Evaluate runtime invariants and optional mandatory checks.

    Checks are synchronous, short, read-only callbacks supplied by the host.
    Run tests, network operations or other expensive verification elsewhere;
    callbacks inspect its evidence. Missing/unknown evidence blocks completion.
    Checks must scope evidence to the current task and current artifact version.
    They cannot override cancellation, pending calls or exhausted budgets.

    The gate neither reads TODO status nor treats old tool failures as permanent
    blockers. Without mandatory checks, completion certifies only the runtime
    conditions, not semantic task correctness. It never publishes files or
    retries the model. Cancellation and resource limits are enforced by the
    runtime; this component classifies their effect on completion.
    """

    def __init__(
        self,
        checks: Mapping[str, Callable[[RunContext], CompletionCheck]] | None = None,
    ) -> None:
        self._checks = dict(checks or {})
        if any(not isinstance(name, str) or not name.strip() or not callable(check)
               for name, check in self._checks.items()):
            raise ValueError("Completion checks require non-empty names and callables")

    def evaluate(
        self,
        context: RunContext,
        *,
        final_requested: bool,
        cancelled: bool = False,
        exhausted_limits: tuple[str, ...] = (),
    ) -> CompletionDecision:
        if cancelled:
            return CompletionDecision(status="cancelled", reasons=("Execution cancelled",))
        if exhausted_limits:
            return CompletionDecision(
                status="incomplete",
                reasons=tuple(f"Limit exhausted: {name}" for name in exhausted_limits),
            )
        pending = tuple(
            f"Pending tool call: {record.id} ({record.status})"
            for record in context.tool_state.records.values() if not record.is_consumed
        )
        if pending:
            return CompletionDecision(status="waiting", reasons=pending)
        if not final_requested:
            return CompletionDecision(status="incomplete", reasons=("No final response proposed",))

        reasons = []
        for name, check in self._checks.items():
            try:
                result = check(context)
                if not isinstance(result, CompletionCheck):
                    raise TypeError("Check must return CompletionCheck")
            except Exception as error:
                reasons.append(f"{name}: verification unavailable ({type(error).__name__})")
                continue
            if result.status != "pass":
                reasons.append(f"{name}: {result.status}: {result.reason or 'No passing evidence'}")
        return CompletionDecision(
            status="incomplete" if reasons else "completed", reasons=tuple(reasons),
        )
