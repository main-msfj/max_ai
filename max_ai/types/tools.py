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


class DockerToolRef(BaseModel):
    """Docker Tool Type"""

    kind: t.Literal["function", "class"] = Field(default="function")
    module: str = Field(
        ...,
        description="Importable Python module containing the referenced tool object",
    )
    qualname: str = Field(
        ...,
        description="Qualified name of the function or class inside the module.",
    )
    options: dict[str, t.Any] = Field(
        default_factory=dict,
        description="Runtime tool options used to reconstruct wrapper metadata, such as name, description, version, approval mode, timeout, and retries.",
    )

    config: dict[str, t.Any] = Field(
        default_factory=dict,
        description="JSON-serializable constructor configuration for class-based tools.",
    )
