from enum import Enum

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

class ToolCallStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    AUTO_APPROVED = "auto_approved"
    REJECTED = "rejected"
    EXECUTING = "executing"    
    CONSUMED = "consumed"