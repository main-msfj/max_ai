"""Which events are hookable, and the named method each maps to on a
CompletionHandler. Curated on purpose — not every event deserves a hook
(e.g. a partial ModelStreamChunkEvent mid-stream isn't useful evidence).

Add a new hook by adding a line here; EventBus.emit never needs a new branch.
"""

from ..event_type import CoreEvent, ModelResponseEvent, ToolCallEvent, ToolCallResponseEvent

EVENT_HOOKS: dict[type[CoreEvent], str] = {
    ToolCallEvent: "on_call_tool_start",
    ToolCallResponseEvent: "on_call_tool",
    ModelResponseEvent: "on_call_llm",
}
