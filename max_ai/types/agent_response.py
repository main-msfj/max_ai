"""
Final result returned from ``Agent.run()``.
"""

from __future__ import annotations

import typing as t
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field

from ..core.messages import CoreMessage, AssistantMessage
from .completions import Usage
from .run_context import RunContext
from .tool_call import ToolCallRecord


# Allowed values for ``finish_reason``. Using ``Literal`` so callers
# get type-checking help; new reasons should be added here as the
# framework grows.
FinishReason = t.Literal[
    "stop",  # LLM emitted no tool calls; conversation done
    "max_iterations",  # ReActLoop hit max_loop_iterations
    "approval_needed",  # paused waiting for user approval on a tool
    "tool_direct_return",  # a tool with return_control_to_llm=False finished
    "no_result",  # client returned without a result (provider error)
    "error",  # uncaught exception in the run
    "cancelled",  # cancellation token was triggered
    "input_needed",  # run paused waiting for additional input from the user
]


class AgentResponse(BaseModel):
    """Final result from ``Agent.run()``.

    Wraps the run's terminal state: the ``RunContext`` (with full chat
    history, tool state, runtime state), why the run stopped, aggregate
    usage, and easy accessors for the most common downstream needs
    (rendering the reply, checking for pending approvals, persisting
    state, resuming).
    """

    model_config = ConfigDict(frozen=True)

    context: RunContext | None = Field(
        default=None,
        description=(
            "The run's terminal state. ``None`` only on catastrophic "
            "failure before a context could be assembled."
        ),
    )
    source: str = Field(
        ...,
        description="Name of the agent that produced this response.",
    )
    usage: Usage = Field(
        ...,
        description="Aggregate usage and timing for the run.",
    )
    finish_reason: FinishReason = Field(
        ...,
        description="Why the agent stopped this run.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this response was assembled.",
    )

    # -------- MESSAGE ACCESS -----------------------------------------------------------
    @property
    def messages(self) -> list[CoreMessage]:
        """All conversational messages from the run (history + new).

        Excludes the synthetic ``SystemMessage`` the client assembles
        from rendered prompt layers — that one is regenerated on every
        turn from the agent's prompt stack and is never persisted as
        part of the conversation transcript.
        """
        if self.context is None:
            return []
        return [
            *self.context.message_history.iter_messages(),
            *self.context.messages,
        ]

    @property
    def final_message(self) -> AssistantMessage | None:
        """The last assistant message produced this run, if any."""
        for msg in reversed(self.messages):
            if isinstance(msg, AssistantMessage):
                return msg
        return None

    @property
    def final_text(self) -> str:
        """Convenience: text of the final assistant message, or ``""``."""
        msg = self.final_message
        return msg.text() if msg else ""

    # -------- APPROVAL FLOW -----------------------------------------------------------
    @property
    def needs_approval(self) -> bool:
        """``True`` if the run paused waiting for user approval on a tool call."""
        if self.context is None:
            return False
        return self.context.tool_state.waiting_for_approval

    @property
    def pending_approvals(self) -> list[ToolCallRecord]:
        """Tool call records currently awaiting user approval."""
        if self.context is None:
            return []
        return self.context.tool_state.pending_approvals

    # -------- DUNDERS -----------------------------------------------------------
    def __str__(self) -> str:
        msg_lines = "\n".join(str(m) for m in self.messages)
        usage = self.usage
        duration_s = usage.duration_ms / 1000

        def fmt(n: int) -> str:
            return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

        tokens = (
            f"in:{fmt(usage.tokens_input)} "
            f"out:{fmt(usage.tokens_output)} "
            f"cached:{fmt(usage.tokens_cached)}"
        )

        if self.needs_approval:
            tail = f" | {len(self.pending_approvals)} approval(s) needed"
        else:
            tail = f" | finish: {self.finish_reason}"

        usage_line = (
            f"[usage] duration: {duration_s:.1f}s, "
            f"tokens: {tokens}, "
            f"llm_calls: {usage.llm_calls}, "
            f"tool_calls: {usage.tool_calls}"
            f"{tail}"
        )

        return f"{msg_lines}\n\n{usage_line}"

    def __repr__(self) -> str:
        approval_info = (
            f", approvals_needed={len(self.pending_approvals)}"
            if self.needs_approval
            else ""
        )
        return (
            f"AgentResponse(source={self.source!r}, "
            f"messages={len(self.messages)}, "
            f"finish_reason={self.finish_reason!r}"
            f"{approval_info})"
        )
