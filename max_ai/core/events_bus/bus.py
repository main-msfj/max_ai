"""Fixed event routing for one Agent: observers vs. the completion veto.

One EventBus per Agent, owned by the loop. Not pluggable — it never varies
per provider or strategy, only its handlers do (CompletionBase subclasses
the host supplies). See base/completion.py for the handler contract.
"""

from __future__ import annotations

import typing as t

from ...base.completion_gate import CompletionDecision
from ..event_type import CoreEvent, ModelResponseEvent, ToolCallEvent, ToolCallResponseEvent
from .hooks import EVENT_HOOKS

if t.TYPE_CHECKING:
    from ...types.run_context import RunContext


class CompletionHandler(t.Protocol):
    """Structural shape a completion component must satisfy.

    Duck-typed on purpose: EventBus never imports base/completion.py, so
    core/ stays free of a hard dependency on a specific pluggable subclass.
    """

    def on_call_llm(self, event: ModelResponseEvent) -> None: ...
    def on_call_tool_start(self, event: ToolCallEvent) -> None: ...
    def on_call_tool(self, event: ToolCallResponseEvent) -> None: ...
    def on_final_response(self, ctx: RunContext) -> CompletionDecision: ...


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

    def emit(self, event: CoreEvent) -> None:
        method_name = EVENT_HOOKS.get(type(event))
        if method_name is None:
            return
        for handler in self._handlers:
            method = getattr(handler, method_name, None)
            if method is not None:
                method(event)

    def check_final_response(self, ctx: RunContext) -> CompletionDecision:
        reasons: list[str] = []
        for handler in self._handlers:
            decision = handler.on_final_response(ctx)
            if decision.status != "completed":
                reasons.extend(decision.reasons)
        return CompletionDecision(
            status="incomplete" if reasons else "completed", reasons=tuple(reasons),
        )
