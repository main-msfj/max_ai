"""
Core data models for maxais.

Includes tool execution, agent responses, usage tracking,
stop signals, and orchestration results.
"""

from __future__ import annotations

import typing as t
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from ..capabilities.tools.plan import AgentPlan
from ..types.completions import Usage
from ..types.todos import Todo
from ..types.tool_call import ToolResult
from .ids import short_id
from .messages import CoreMessage, Message


# -------- -----------------------------------------------------------
# Bases
# -------- -----------------------------------------------------------
class CoreEvent(BaseModel):
    """Abstract base class for all events"""

    EVENT_TYPE: t.ClassVar[str]
    __abstract__: t.ClassVar[bool] = True

    model_config = ConfigDict(frozen=True)

    source: str = Field()
    event_type: str = Field(default="")
    event_id: str = Field(default_factory=short_id)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip validation if this subclass marks itself as abstract.
        if cls.__dict__.get("__abstract__", False):
            return
        # Concrete subclasses MUST define their own EVENT_TYPE.
        if "EVENT_TYPE" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must define EVENT_TYPE (or set "
                "__abstract__ = True if it's an abstract base)"
            )

    def __init__(self, **data: t.Any):
        if "event_type" in data:
            raise ValueError("event_type is auto-generated and cannot be set manually")
        data["event_type"] = self.EVENT_TYPE
        super().__init__(**data)

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{self.source}] {time_str} | {self.event_type}"


# -------- -----------------------------------------------------------
# Groups event
# -------- -----------------------------------------------------------


class ModelEvent(CoreEvent):
    """Base class for all Model-related events"""

    __abstract__ = True

    pass


class ReasoningEvent(CoreEvent):
    """Base class for all Reasoning-related events"""

    __abstract__ = True

    pass


class AgentEvent(CoreEvent):
    """Base class for all Agent-related events"""

    __abstract__ = True

    pass


class ToolEvent(CoreEvent):
    """Base class for all Tool-related events"""

    __abstract__ = True

    pass


class ErrorsEvent(CoreEvent):
    """Base class for all Tool-related events"""

    __abstract__ = True

    pass


# -------- -----------------------------------------------------------
#  Model Events
# -------- -----------------------------------------------------------
class ModelCallEvent(ModelEvent):
    """Event emitted when LLM API call is initiated."""

    EVENT_TYPE = "model_call"
    model: str = Field(..., description="Model being called")
    input_messages: t.Sequence[CoreMessage] = Field(
        ..., description="Message sent to model"
    )
    prompt_tokens: int = Field(
        default=0, description="Size of the rendered system prompt for this call"
    )


class ModelResponseEvent(ModelEvent):
    """Event emitted when LLM response is received."""

    EVENT_TYPE = "model_response"
    response: str = Field(..., description="The model's response")
    has_tool_calls: bool = Field(default=False)
    usage: Usage | None = Field(default=None, description="Token usage statistics")


class ModelRetryEvent(ModelEvent):
    """A model call failed for a passing reason; the harness calls again
    after ``delay`` seconds."""

    EVENT_TYPE = "model_retry"
    attempt: int = Field(..., description="The try that failed, from 1")
    max_attempts: int = Field(..., description="Tries allowed in total")
    delay: float = Field(..., description="Seconds before the next try")
    reason: str = Field(..., description="Why it failed, e.g. rate_limit")
    discard_partial: bool = Field(
        default=False,
        description="Part of the answer was already streamed: drop it, the retry streams it again",
    )


class ModelStreamChunkEvent(ModelEvent):
    """Event emitted for each streaming chunk from LLM."""

    EVENT_TYPE = "model_stream_chunk"
    chunk: str = Field(..., description="Incremental text chunk")
    thinking: str | None = Field(default=None, description="Incremental reasoning")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")


# -------- -----------------------------------------------------------
#  Reasoning Events
# -------- -----------------------------------------------------------
class ReasoningIterationEvent(ReasoningEvent):
    """Event emitted at the start of each reasoning loop iteration."""

    EVENT_TYPE = "reasoning_iteration"
    iteration: int = Field(..., description="Current iteration number")
    max_iterations: int = Field(..., description="Maximum allowed iterations")


class ReasoningCompleteEvent(ReasoningEvent):
    """Event emitted when the reasoning loop finishes."""

    EVENT_TYPE = "reasoning_complete"
    finish_reason: str = Field(..., description="Why the loop stopped")
    total_iterations: int = Field(..., description="How many iterations were executed")


class LoopNoticeEvent(ReasoningEvent):
    """A rule of the loop corrected the model or ended the turn."""

    EVENT_TYPE = "loop_notice"
    kind: str = Field(..., description="The rule: empty_answer, repetition, near_limit...")
    message: str = Field(default="", description="What the model was told, if anything")


class PlanningEvent(ReasoningEvent):
    """Event emitted when a plan is generated."""

    EVENT_TYPE = "planning"
    phase: t.Literal["start", "complete", "failed", "skipped", "progress"]
    plan: AgentPlan | None = Field(default=None)


class UserInputRequestEvent(ReasoningEvent):
    """A tool needs a person; the turn pauses until they answer.

    Mirrors ``ToolApprovalEvent``. The consumer answers via
    ``ctx.tool_state.provide_result(tool_call_id, result)`` and resumes.
    """

    EVENT_TYPE = "user_input_request"
    tool_name: str = Field(description="Tool waiting for the person")
    parameters: dict[str, t.Any] = Field(default_factory=dict)
    tool_call_id: str | None = Field(
        default=None,
        description="Record id to answer via tool_state.provide_result().",
    )


# -------- -----------------------------------------------------------
#  Agent Events
# -------- -----------------------------------------------------------


class CompactionEvent(AgentEvent):
    """Emitted when context compaction starts and when it ends.

    ``old_messages`` (end only) are the messages that left the window, for
    hosts that archive the full conversation; the RunContext no longer has them.
    """

    EVENT_TYPE = "compaction"
    phase: t.Literal["start", "end"] = Field(
        ..., description="Whether compaction is starting or finished"
    )
    strategy: str = Field(..., description="Compaction strategy class name")
    changed: bool = Field(default=False, description="Whether the window changed")
    pruned_only: bool = Field(
        default=False, description="The cheap prune pass was enough (no LLM)"
    )
    tokens_before: int = Field(default=0, description="Live tokens before compacting")
    tokens_after: int = Field(default=0, description="Live tokens after compacting")
    kept_message_count: int = Field(
        default=0, description="Messages left in the window"
    )
    usage: Usage | None = Field(default=None, description="What the summary call cost")
    old_messages: list[Message] = Field(  # type: ignore[valid-type]
        default_factory=list, description="Messages that left the window"
    )
    summary: str | None = Field(
        default=None, description="What the model now sees of the past (render())"
    )


# -------- -----------------------------------------------------------
#  Tool Events
# -------- -----------------------------------------------------------


class MCPProgressEvent(ToolEvent):
    """An MCP server reported how far a running call has got."""

    EVENT_TYPE = "mcp_progress"
    tool_call_id: str
    server_id: str
    tool_name: str
    progress: float
    total: float | None = None
    message: str | None = None


class TodosEvent(ToolEvent):
    """The agent changed its list of things to do."""

    EVENT_TYPE = "todos"
    tool_call_id: str
    todos: list[Todo]


class ToolCallEvent(ToolEvent):
    """Emitted when a tool is about to be called."""

    EVENT_TYPE = "tool_call"
    tool_name: str = Field(..., description="Name of the tool being called")
    parameters: dict[str, t.Any] = Field(..., description="Tool parameters")
    tool_call_id: str = Field(..., description="Unique identifier for this tool call")

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"[{self.source}] {time_str} | tool_call: {self.tool_name}({params_str})"


class ToolCallResponseEvent(ToolEvent):
    """Emitted when tool execution completes (success or failure)."""

    EVENT_TYPE = "tool_call_response"
    tool_call_id: str = Field(..., description="Unique identifier for this tool call")
    tool_result: ToolResult | None = Field(
        default=None,
        description="The actual result object (carries success, value, error, etc).",
    )

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        msg = f"[{self.source}] {time_str}"
        if self.tool_result is None:
            return msg + " | tool_response: (no result)"
        status = "[SUCCESS]" if self.tool_result.success else "[FAILED]"
        preview = str(self.tool_result.result or self.tool_result.error or "")
        if len(preview) > 50:
            preview = preview[:50] + "..."
        return msg + f" | tool_response: {status} {preview}"


class ToolApprovalEvent(ToolEvent):
    """Emitted when a tool call requires user approval before executing."""

    EVENT_TYPE = "tool_approval"
    tool_call_id: str = Field(..., description="Tool call ID")
    tool_name: str = Field(..., description="Name of the tool requesting approval")
    parameters: dict[str, t.Any] = Field(..., description="Tool call parameters")
    reason_for_approval: str | None = Field(default=None)

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        reason_part = (
            f" ({self.reason_for_approval})" if self.reason_for_approval else ""
        )
        return (
            f"[{self.source}] {time_str} | "
            f"Approval needed for tool '{self.tool_name}' "
            f"(ID: {self.tool_call_id}){reason_part}"
        )


class ToolAutoApprovalEvent(ToolEvent):
    """Emitted when policy approves a tool call without asking the user."""

    EVENT_TYPE = "tool_auto_approval"
    tool_call_id: str
    tool_name: str


# -------- -----------------------------------------------------------
#  Error Events
# -------- -----------------------------------------------------------
class ErrorEvent(ErrorsEvent):
    """Event emitted for recoverable errors."""

    EVENT_TYPE = "error"
    error_message: str = Field(..., description="Description of the error")
    error_type: str = Field(..., description="Type/category of error")
    is_recoverable: bool = Field(default=True)


class FatalErrorEvent(ErrorsEvent):
    """Event emitted for unrecoverable errors that terminate execution."""

    EVENT_TYPE = "fatal_error"
    error_message: str = Field(..., description="Description of the fatal error")
    error_type: str = Field(..., description="Type/category of error")
    is_recoverable: bool = Field(default=False)
