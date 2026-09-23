"""
Core data models for maxais.

Includes tool execution, agent responses, usage tracking,
stop signals, and orchestration results.
"""

from __future__ import annotations

import typing as t
from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Discriminator, Field

from ..base.completion_gate import CompletionDecision
from ..capabilities.tools.plan import AgentPlan
from ..ids import short_id
from ..types.completions import Usage
from ..types.tool_call import ToolResult
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
    """Emitted once, only when the completion gate accepts the final
    response. Never emitted by the model, and never on a pause, error,
    or exhausted limit — only on a genuine, gate-approved completion."""

    EVENT_TYPE = "task_complete"
    decision: CompletionDecision = Field(
        ..., description="The aggregate decision that closed this turn"
    )


class CompletionRejectedEvent(TasksEvent):
    """Emitted every time the completion gate rejects a proposed final
    response as incomplete and forces the model to try again. Distinct
    from TaskCompleteEvent so a consumer can show retries live instead
    of only finding out about them after the fact in ctx.messages."""

    EVENT_TYPE = "completion_rejected"
    decision: CompletionDecision = Field(
        ..., description="The aggregate decision that rejected this response"
    )


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


class LastMessageResponseEvent(ReasoningEvent):
    """The model responded with no tool calls — this response is the
    proposed end of the turn, about to be checked by the completion gate."""

    EVENT_TYPE = "last_message_response"
    response: str = Field(..., description="The model's proposed final text")


class PlanningEvent(ReasoningEvent):
    """Event emitted when a plan is generated."""

    EVENT_TYPE = "planning"
    phase: t.Literal["start", "complete", "failed", "skipped", "progress"]
    plan: AgentPlan | None = Field(default=None)


class UserInputRequestEvent(ReasoningEvent):
    """The agent asked the user a question; the turn pauses until answered.

    Mirrors ``ToolApprovalEvent``: emitted by the executor *instead of*
    executing the ask-the-user tool. The consumer answers via
    ``ctx.tool_state.apply_user_answer(tool_call_id, answer)`` and resumes.
    """

    EVENT_TYPE = "user_input_request"
    question: str = Field(description="Question asked to the user")
    options: list[str] | None = Field(default=None)
    tool_call_id: str | None = Field(
        default=None,
        description="Record id to answer via tool_state.apply_user_answer().",
    )


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
    old_messages: list[Message] = Field(  # type: ignore[valid-type]
        default_factory=list, description="Messages that left the window"
    )
    summary: str | None = Field(
        default=None, description="What the model now sees of the past (render())"
    )


# -------- -----------------------------------------------------------
#  Tool Events
# -------- -----------------------------------------------------------
class BashStartedEvent(ToolEvent):
    """Execution requested; declared_action describes model intent only."""

    EVENT_TYPE = "bash_started"
    tool_call_id: str
    command: str
    declared_action: str
    description: str


class BashFinishedEvent(ToolEvent):
    EVENT_TYPE = "bash_finished"
    tool_call_id: str
    exit_code: int | None
    duration_ms: int
    timed_out: bool = False
    truncated: bool = False


class BashFailedEvent(ToolEvent):
    EVENT_TYPE = "bash_failed"
    tool_call_id: str
    error: str


class BashCancelledEvent(ToolEvent):
    EVENT_TYPE = "bash_cancelled"
    tool_call_id: str
    reason: str


class FileReadEvent(ToolEvent):
    """A workspace file was read successfully."""

    EVENT_TYPE = "file_read"
    tool_call_id: str
    path: str
    root_dir: str
    content_hash: str


class DirectoryListedEvent(ToolEvent):
    """A workspace directory was listed successfully."""

    EVENT_TYPE = "directory_listed"
    tool_call_id: str
    path: str
    root_dir: str
    entry_count: int


class FileWrittenEvent(ToolEvent):
    """A workspace file was created or edited successfully."""

    EVENT_TYPE = "file_written"
    tool_call_id: str
    operation: t.Literal["write_file", "edit_file"]
    path: str
    root_dir: str
    content_hash: str


class FilesSearchedEvent(ToolEvent):
    """A workspace search completed successfully."""

    EVENT_TYPE = "files_searched"
    tool_call_id: str
    operation: t.Literal["find_files", "search_text"]
    path: str
    root_dir: str
    match_count: int
    truncated: bool = False


class DirectoryCreatedEvent(ToolEvent):
    EVENT_TYPE = "directory_created"
    tool_call_id: str
    path: str
    root_dir: str


class FileDeletedEvent(ToolEvent):
    EVENT_TYPE = "file_deleted"
    tool_call_id: str
    path: str
    root_dir: str
    content_hash: str


class FileInfoEvent(ToolEvent):
    EVENT_TYPE = "file_info"
    tool_call_id: str
    path: str
    root_dir: str
    file_type: t.Literal["file", "directory"]


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
OrchestrationEvents = Annotated[
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
        CompletionRejectedEvent,
        ModelCallEvent,
        ModelResponseEvent,
        ModelStreamChunkEvent,
        ReasoningIterationEvent,
        ReasoningCompleteEvent,
        LastMessageResponseEvent,
        PlanningEvent,
        UserInputRequestEvent,
        ToolCallEvent,
        ToolCallResponseEvent,
        ToolApprovalEvent,
        ToolAutoApprovalEvent,
        ToolValidationEvent,
        ToolProgressEvent,
        BashStartedEvent,
        BashFinishedEvent,
        BashFailedEvent,
        BashCancelledEvent,
        FileReadEvent,
        DirectoryListedEvent,
        FileWrittenEvent,
        FilesSearchedEvent,
        DirectoryCreatedEvent,
        FileDeletedEvent,
        FileInfoEvent,
        CompactionEvent,
        MemoryUpdateEvent,
        MemoryRetrievalEvent,
        ErrorEvent,
        FatalErrorEvent,
    ],
    Discriminator("event_type"),
]
