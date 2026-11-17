import typing as t
from enum import Enum

from pydantic import BaseModel, Field


# -------- TOOLS BASE MODEL -----------------------------------------------------------
class ToolApprovalMode(str, Enum):
    """Tool approval requirements"""

    AUTO_APPROVED = "auto_approval"
    ASK_APPROVED = "ask_for_approval"


class ToolKind(str, Enum):
    """Where a tool comes from; it decides which rules its name follows."""

    NATIVE = "native"  # written for the framework (bash, read_file)
    FUNCTION = "function"  # a Python function turned into a tool
    MCP = "mcp"  # given by an MCP server, which names it


class CoreToolParameters(BaseModel):
    """Core tool parameters"""

    is_tool_valid: bool = Field(default=False)
    msg_error: str | None = Field(default=None)


class CoreToolDefinition(BaseModel):
    """Core tool definition"""

    name: str = Field(..., description="Name of the given tool from custom ot MCP")
    description: str = Field(..., description="Description of the tool's purpose")
    parameters: dict[str, t.Any] = Field(default_factory=dict)
    sandbox: bool = Field(
        default=False, description="True: it runs in the sandbox. False: it runs on the host."
    )
