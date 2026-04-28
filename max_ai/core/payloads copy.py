import uuid
import typing as t
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, model_validator

from ..context.run_context import RunContext

# -------- ENUMS -----------------------------------------------------------
class ApprovalStatus(str, Enum):
    """Status of a tool call in the approval workflow."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    CONSUMED = "consumed"


class FailureReason(str, Enum):
    """Categorized reason for a ToolResult failure."""

    CANCELLED_BEFORE_START = "cancelled_before_start"
    CANCELLED_DURING_EXECUTION = "cancelled_during_execution"
    INVALID_PARAMETERS = "invalid_parameters"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"


# -------- TOOLS BASE MODEL -----------------------------------------------------------
class ToolApprovalMode(str, Enum):
    """Tool approval requirements"""

    AUTO_APPROVED = "auto_approval"
    ASK_APPROVED = "ask_for_approval"


class CoreToolParameters(BaseModel):
    """Core tool parameters"""

    is_tool_valid: bool = Field(...)
    msg_error: str | None = Field(default=None)


class CoreToolDefinition(BaseModel):
    """Core tool definition"""

    name: str = Field(..., description="Name of the given tool from custom ot MCP")
    description: str = Field(..., description="Description of the tool's purpose")
    parameters: dict[str, t.Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    """Tool Exexcution Request"""

    model_config = ConfigDict(frozen=True)

    # -------- -----------------------------------------------------------
    # TRACKING
    # -------- -----------------------------------------------------------
    session_id: str | None = Field(default=None)
    requester_id: str | None = Field(default=None)
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: dict[str, t.Any] = Field(..., description="Arguments for the tool")

    # -------- -----------------------------------------------------------
    # FIX VALUES
    # -------- -----------------------------------------------------------
    tool_type: t.Literal["mcp", "custom"] = Field(default="custom")
    tool_call_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    requested_by: str = Field(default="user", description="Who requested the tool")
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolResult(BaseModel):
    """Tool execution result to be returned to the LLM."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(...)
    error: str | None = Field(default=None)
    result: t.Any | None = Field(default=None, description="Tool execution output")
    failure_reason: FailureReason | None = Field(default=None)

    # -------- -----------------------------------------------------------
    # TRACKING
    # -------- -----------------------------------------------------------
    completed_at: datetime | None = Field(default=None)
    metadata: dict[str, t.Any] = Field(default_factory=dict)
    tool_request: ToolCallRecord = Field(..., description="Original tool request")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def duration_ms(self) -> int | None:
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds() * 1000)
        return None

    @classmethod
    def success_result(
        cls,
        tool_request: ToolCallRecord,
        result: t.Any,
        metadata: dict[str, t.Any] | None = None,
    ) -> t.Self:
        """Factory for a successful ToolResult."""
        return cls(
            success=True,
            result=result,
            error=None,
            tool_request=tool_request.model_copy(),
            metadata=metadata or {},
        )

    @classmethod
    def tool_failure(
        cls,
        tool_request: ToolCallRecord,
        error: str,
        reason: FailureReason = FailureReason.EXECUTION_ERROR,
        metadata: dict[str, t.Any] | None = None,
    ) -> t.Self:
        """Factory for a failed ToolResult with a categorized reason."""
        return cls(
            success=False,
            result=None,
            error=error,
            failure_reason=reason,
            tool_request=tool_request.model_copy(),
            metadata=metadata or {},
        )

    # -------- CONVENIENCE FACTORIES FOR COMMON FAILURES -----------------------------------------------------------
    @classmethod
    def cancelled_before_start(cls, tool_request: ToolCallRecord) -> t.Self:
        return cls.tool_failure(
            tool_request,
            error="Tool was cancelled before it started.",
            reason=FailureReason.CANCELLED_BEFORE_START,
        )

    @classmethod
    def cancelled_during_execution(cls, tool_request: ToolCallRecord) -> t.Self:
        return cls.tool_failure(
            tool_request,
            error="Tool was cancelled during execution.",
            reason=FailureReason.CANCELLED_DURING_EXECUTION,
        )

    @classmethod
    def invalid_parameters(cls, tool_request: ToolCallRecord, val_err: str) -> t.Self:
        return cls.tool_failure(
            tool_request,
            error=f"Invalid parameters: {val_err}",
            reason=FailureReason.INVALID_PARAMETERS,
        )

    @classmethod
    def execution_error(
        cls, tool_request: ToolCallRecord, err_msg: str | None = None
    ) -> t.Self:
        return cls.tool_failure(
            tool_request,
            error=err_msg or "Sorry, an error occurred while processing the request.",
            reason=FailureReason.EXECUTION_ERROR,
        )

    @classmethod
    def timeout(cls, tool_request: ToolCallRecord, timeout_seconds: float) -> t.Self:
        return cls.tool_failure(
            tool_request,
            error=f"Tool execution exceeded timeout of {timeout_seconds} seconds.",
            reason=FailureReason.TIMEOUT,
        )


# -------- APPROVAL WORKFLOW MODELS -----------------------------------------------------------
class ToolApprovalRequest(BaseModel):
    """Internal record that a tool call is awaiting approval."""

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool_request: ToolCallRecord = Field(..., description="Tool call awaiting approval")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def tool_name(self) -> str:
        return self.tool_request.tool_name

    @property
    def tool_call_id(self) -> str:
        return self.tool_request.tool_call_id

    @property
    def parameters(self) -> dict[str, t.Any]:
        return self.tool_request.parameters


class ToolApprovalResponse(BaseModel):
    """User's approval/rejection decision."""

    approved: bool = Field(...)
    tool_call_id: str = Field(...)
    reason: str | None = Field(default=None)
    parameters: dict[str, t.Any] | None = Field(
        default=None, description="Optional parameter overrides"
    )

    # Tracking fields. `consumed` is flipped to True after the approved tool
    # has been executed successfully, to prevent re-execution of the same
    # tool_call_id within the same run.
    consumed: bool = Field(default=False)
    consumed_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _default_rejection_reason(self):
        if not self.approved:
            self.reason = self.reason or "The tool was rejected from UI."
        return self


