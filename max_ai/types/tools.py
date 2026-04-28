
import typing as t
from enum import Enum
from pydantic import BaseModel, Field

# -------- TOOLS BASE MODEL -----------------------------------------------------------
class ToolApprovalMode(str, Enum):
    """Tool approval requirements"""

    AUTO_APPROVED = "auto_approval"
    ASK_APPROVED = "ask_for_approval"


class CoreToolParameters(BaseModel):
    """Core tool parameters"""

    is_tool_valid: bool = Field(default=False)
    msg_error: str | None = Field(default=None)


class CoreToolDefinition(BaseModel):
    """Core tool definition"""

    name: str = Field(..., description="Name of the given tool from custom ot MCP")
    description: str = Field(..., description="Description of the tool's purpose")
    parameters: dict[str, t.Any] = Field(default_factory=dict)