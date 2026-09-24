"""Harness-owned completion decisions, independent of plans and publication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from .component import ComponentBase

if TYPE_CHECKING:
    from ..core.event_type import (
        ModelResponseEvent,
        ToolCallEvent,
        ToolCallResponseEvent,
    )
    from ..types.run_context import RunContext


class CompletionCheck(BaseModel):
    """Evidence returned by trusted application code, never model arguments."""

    model_config = ConfigDict(frozen=True)
    status: Literal["pass", "fail", "unknown"]
    reason: str = ""


class CompletionDecision(BaseModel):
    """
    Represent the result of evaluating whether an agent turn may finish.
    """
    model_config = ConfigDict(frozen=True)
    status: Literal["completed", "incomplete", "waiting", "cancelled"]
    reasons: tuple[str, ...] = ()


class CompletionConfig(BaseModel):
    """
    Define the serializable settings shared by completion gates.
    """
    model_config = ConfigDict(extra="forbid")


class CompletionBase(ComponentBase[CompletionConfig], ABC):
    """Subclass and override what you need — only ``on_final_response`` is
    required.

    Registered as a handler on an Agent's ``EventBus``. The three
    observation hooks (``on_model_response``, ``on_call_tool``,
    ``on_tool_response``) are fire-and-forget: nothing reads their return
    value, and they cannot veto anything. Only ``on_final_response`` can
    force the loop to keep iterating instead of ending the turn.

    Write anything a hook needs to remember into ``ctx`` (never into
    ``self``) — one instance is shared across every turn this Agent runs,
    so private state on ``self`` would leak between turns and wouldn't
    survive a resume after a process restart.

    Serialization is opt-in per subclass, like every other ``ComponentBase``
    (``CoreTool`` doesn't implement it either): implement your own
    ``_to_config``/``_from_config`` if you need it. There is no default
    here on purpose — a default that quietly ignores a subclass's real
    config is worse than no default at all.
    """

    component_type = "completion"
    component_schema = CompletionConfig

    @property
    def gate_id(self) -> str:
        """Key into ``ctx.completion_state``. Override if you register more
        than one instance of the same class — two gates sharing an id would
        silently overwrite each other's evidence."""
        return type(self).__name__

    def on_model_response(self, event: ModelResponseEvent, ctx: RunContext) -> None:
        """Fires once the model responds, before any of its tool calls run."""
        return None

    def on_call_tool(self, event: ToolCallEvent, ctx: RunContext) -> None:
        """Fires right before a tool call executes."""
        return None

    def on_tool_response(self, event: ToolCallResponseEvent, ctx: RunContext) -> None:
        """Fires once a tool call finishes, success or failure."""
        return None

    @abstractmethod
    def on_final_response(
        self, ctx: RunContext
    ) -> CompletionDecision | Awaitable[CompletionDecision]:
        """Decide whether the turn can end. Called only when the model
        proposed a final answer with no tool calls. May be ``def`` or
        ``async def`` — implement it async if verifying needs an LLM-judge
        call or any other I/O; a cheap local check can stay synchronous."""
        ...
