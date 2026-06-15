"""
Core data models for maxais.

Includes tool execution, agent responses, usage tracking,
stop signals, and orchestration results.
"""

from __future__ import annotations

import uuid
import typing as t
from typing import Annotated
from datetime import datetime, timezone
from pydantic import BaseModel, Discriminator, Field, ConfigDict

from .messages import CoreMessage
from ..types.completions import Usage
from ..types.tool_call import ToolResult

from ..reasoning.plan import AgentPlan
from ..reasoning.eval import EvalResult
from ..base.scratchpad import Scratchpad


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
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
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
class OrchestrationEvent(CoreEvent):
    """Base class for all orchestration-related events"""

    __abstract__ = True

    pass


class TasksEvent(CoreEvent):
    """Base class for all Tasks-related events"""

    __abstract__ = True

    pass


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


class MemoryEvent(CoreEvent):
    """Base class for all Memory-related events"""

    __abstract__ = True

    pass


# -------- -----------------------------------------------------------
#  Orchestration Events For Streaming
# -------- -----------------------------------------------------------
class OrchestrationStartEvent(OrchestrationEvent):
    """Emitted when orchestration begins."""

    EVENT_TYPE = "orchestration_start"
    task: str = Field(..., description="The task being orchestrated")
    pattern: str = Field(..., description="Orchestration pattern being used")


class OrchestrationCompleteEvent(OrchestrationEvent):
    """Emitted when orchestration ends."""

    EVENT_TYPE = "orchestration_complete"
    result: str = Field(..., description="Final orchestration result")
    stop_reason: str = Field(..., description="Why orchestration stopped")


# -------- -----------------------------------------------------------
#  Tasks Events
# -------- -----------------------------------------------------------
class TaskStartEvent(TasksEvent):
    """Emitted when task processing begins."""

    EVENT_TYPE = "task_start"
    task: str = Field(..., description="The task being started")


class TaskCompleteEvent(TasksEvent):
    """Emitted when task processing completes."""

    EVENT_TYPE = "task_complete"
    result: str = Field(..., description="The final task result")


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


class ModelResponseEvent(ModelEvent):
    """Event emitted when LLM response is received."""

    EVENT_TYPE = "model_response"
    response: str = Field(..., description="The model's response")
    has_tool_calls: bool = Field(default=False)
    usage: Usage | None = Field(default=None, description="Token usage statistics")


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


class PlanningEvent(ReasoningEvent):
    """Event emitted when a plan is generated."""

    EVENT_TYPE = "planning"
    phase: t.Literal["start", "complete", "failed", "skipped"]
    plan: AgentPlan | None = Field(default=None)


class EvalEvent(ReasoningEvent):
    """Emmited durin the self-evaluationn step"""

    EVENT_TYPE = "eval"
    phase: t.Literal["start", "complete", "failed", "skipped", "intermediate"]
    score: float | None = Field(default=None)
    passed: bool | None = Field(default=None)
    result: EvalResult | None = Field(default=None)


class UserInputRequestEvent(ReasoningEvent):
    EVENT_TYPE = "user_input_request"
    question: str = Field(description="Questsion ask to user")
    options: list[str] | None

class ScratchpadUpdateEvent(ReasoningEvent):
    EVENT_TYPE = "scratchpad_update"
    scratchpad: Scratchpad


# -------- -----------------------------------------------------------
#  Agent Events
# -------- -----------------------------------------------------------
class AgentSelectionEvent(AgentEvent):
    """Emitted when an agent is selected for execution."""

    EVENT_TYPE = "agent_selection"
    selected_agent: str = Field(..., description="Name of selected agent")
    selection_reason: str | None = Field(default=None)


class AgentExecutionStartEvent(AgentEvent):
    """Emitted when agent execution begins."""

    EVENT_TYPE = "agent_execution_start"
    executing_agent: str = Field(..., description="Name of executing agent")
    context_size: int = Field(..., description="Number of messages in context")


class AgentExecutionCompleteEvent(AgentEvent):
    """Emitted when agent execution completes."""

    EVENT_TYPE = "agent_execution_complete"
    executing_agent: str = Field(..., description="Name of executed agent")
    success: bool = Field(..., description="Whether execution succeeded")
    message_count: int = Field(..., description="Number of messages produced")


class CompactionEvent(AgentEvent):
    """Emitted when context compaction starts or finishes."""

    EVENT_TYPE = "compaction"
    phase: t.Literal["start", "end"] = Field(
        ..., description="Whether compaction is starting or finished"
    )
    strategy: str = Field(..., description="Compaction strategy name")
    changed: bool = Field(
        default=False, description="Whether compaction changed the active transcript"
    )
    old_message_count: int = Field(
        default=0, description="Messages moved out of context"
    )
    recent_message_count: int = Field(default=0, description="Messages kept in context")
    old_token_count: int = Field(
        default=0, description="Token count moved out of context"
    )
    recent_token_count: int = Field(
        default=0, description="Token count kept in context"
    )
    total_token_count: int = Field(
        default=0, description="Token count before compaction"
    )
    live_message_threshold_tokens: int = Field(
        default=0, description="Live message token count that triggered compaction"
    )
    live_message_budget_tokens: int = Field(
        default=0, description="Raw live message budget kept after compaction"
    )
    summary: str | None = Field(
        default=None, description="Updated structured summary payload"
    )
    context_summary_persisted: bool = Field(
        default=False,
        description="Whether the summary was written to the context registry",
    )
    context_summary_session_id: str | None = Field(
        default=None, description="Session id used when persisting the context summary"
    )


# -------- -----------------------------------------------------------
#  Tool Events
# -------- -----------------------------------------------------------
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


class ToolValidationEvent(ToolEvent):
    """Event emitted after parameter validation."""

    EVENT_TYPE = "tool_validation"
    tool_name: str = Field(..., description="Name of the tool being validated")
    is_valid: bool = Field(..., description="Whether parameters are valid")
    errors: str | None = Field(default=None, description="Validation error details")


class ToolProgressEvent(ToolEvent):
    """Event emitted to indicate progress during long-running tool execution."""

    EVENT_TYPE = "tool_progress"
    tool_name: str = Field(..., description="Name of the tool in progress")
    content: str = Field(..., description="Human-readable progress update")
    tool_call_id: str = Field(..., description="Tool call ID")


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


# -------- -----------------------------------------------------------
#  Memory Events
# -------- -----------------------------------------------------------
class MemoryUpdateEvent(MemoryEvent):
    """Event emitted when memory state changes."""

    EVENT_TYPE = "memory_update"
    operation: str = Field(..., description="Type of memory operation")


class MemoryRetrievalEvent(MemoryEvent):
    """Event emitted when memory content is accessed."""

    EVENT_TYPE = "memory_retrieval"
    query: str = Field(..., description="Query used to retrieve memories")
    results_count: int = Field(..., description="Number of memories retrieved")


# -------- -----------------------------------------------------------
# Union types Orchestration
# -------- -----------------------------------------------------------
OrchestrationEvent = Annotated[
    t.Union[
        OrchestrationStartEvent,
        OrchestrationCompleteEvent,
        AgentSelectionEvent,
        AgentExecutionStartEvent,
        AgentExecutionCompleteEvent,
        CompactionEvent,
    ],
    Discriminator("event_type"),
]

# -------- -----------------------------------------------------------
# Union type for all Agent Events
# -------- -----------------------------------------------------------
AgentEvents = Annotated[
    t.Union[
        TaskStartEvent,
        TaskCompleteEvent,
        ModelCallEvent,
        ModelResponseEvent,
        ModelStreamChunkEvent,
        ReasoningIterationEvent,
        ReasoningCompleteEvent,
        PlanningEvent,
        ScratchpadUpdateEvent,
        EvalEvent,
        ToolCallEvent,
        ToolCallResponseEvent,
        ToolApprovalEvent,
        ToolValidationEvent,
        ToolProgressEvent,
        CompactionEvent,
        MemoryUpdateEvent,
        MemoryRetrievalEvent,
        ErrorEvent,
        FatalErrorEvent,
    ],
    Discriminator("event_type"),
]
