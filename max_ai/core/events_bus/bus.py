"""Fixed event routing for one Agent: observers vs. the completion veto.

One EventBus per Agent, owned by the loop. Not pluggable — it never varies
per provider or strategy, only its handlers do (CompletionBase subclasses
the host supplies). See base/completion_gate.py for the handler contract.
"""

from __future__ import annotations

import inspect
import typing as t

from ...base.completion_gate import CompletionDecision
from ..event_type import (
    CoreEvent,
    ModelResponseEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)
from .hooks import EVENT_HOOKS

if t.TYPE_CHECKING:
    from ...types.run_context import RunContext


class CompletionHandler(t.Protocol):
    """Structural shape a completion component must satisfy.

    Duck-typed on purpose: EventBus never imports base/completion.py, so
    core/ stays free of a hard dependency on a specific pluggable subclass.
    """

    def on_model_response(self, event: ModelResponseEvent, ctx: RunContext) -> None: ...
    def on_call_tool(self, event: ToolCallEvent, ctx: RunContext) -> None: ...
    def on_tool_response(
        self, event: ToolCallResponseEvent, ctx: RunContext
    ) -> None: ...
    def on_final_response(
        self, ctx: RunContext
    ) -> CompletionDecision | t.Awaitable[CompletionDecision]: ...


# Severity order for aggregating multiple handlers' decisions — the worst
# status wins, regardless of whether it carries a reason string.
_STATUS_SEVERITY = {"completed": 0, "incomplete": 1, "waiting": 2, "cancelled": 3}


class EventBus:
    """Routes emitted events to registered handlers.

    ``emit`` is fire-and-forget: handlers observe, none can veto. Only
    ``check_final_response`` can force the loop to keep iterating — the
    loop calls it directly and waits, it is not a subscription.

    Which event types are hookable, and which named method on the handler
    they map to, lives in ``hooks.py`` (not every event deserves a hook —
    a partial ``ModelStreamChunkEvent`` mid-stream isn't useful evidence).
    Add a new one by adding a line there, never by adding a branch to
    ``emit``.
    """

    def __init__(self, handlers: list[CompletionHandler] | None = None) -> None:
        self._handlers = list(handlers or [])
        seen: set[str] = set()
        for handler in self._handlers:
            gate_id = getattr(handler, "gate_id", None)
            if gate_id is None:
                continue
            if gate_id in seen:
                raise ValueError(
                    f"Duplicate completion gate_id {gate_id!r} — two gates "
                    "sharing an id would silently overwrite each other's "
                    "evidence in ctx.completion_state. Give one an explicit "
                    "gate_id."
                )
            seen.add(gate_id)

    def emit(self, event: CoreEvent, ctx: RunContext) -> None:
        method_name = EVENT_HOOKS.get(type(event))
        if method_name is None:
            return
        for handler in self._handlers:
            method = getattr(handler, method_name, None)
            if method is not None:
                method(event, ctx)

    async def check_final_response(self, ctx: RunContext) -> CompletionDecision:
        """Both sync and async ``on_final_response`` implementations work —
        a handler can run a cheap local check or await an LLM-judge call."""
        worst = "completed"
        blocking: list[str] = []
        # Reasons on a "completed" decision are notes (e.g. "closed with an
        # unresolved failure") — surfaced only when nothing blocks.
        notes: list[str] = []
        for handler in self._handlers:
            on_final = getattr(handler, "on_final_response", None)
            if on_final is None:
                continue
            decision = on_final(ctx)
            if inspect.isawaitable(decision):
                decision = await decision
            if decision.status != "completed":
                blocking.extend(decision.reasons)
            else:
                notes.extend(decision.reasons)
            if _STATUS_SEVERITY[decision.status] > _STATUS_SEVERITY[worst]:
                worst = decision.status
        reasons = notes if worst == "completed" else blocking
        return CompletionDecision(status=worst, reasons=tuple(reasons))
