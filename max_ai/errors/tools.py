import typing as t


class ToolRetry(Exception):
    """Signal the LLM to retry with corrected parameters."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ToolRegistryError(Exception):
    """Raised when a tool cannot be registered."""

    def __init__(self, message: str, kind: str = "generic") -> None:
        super().__init__(message)
        self.kind = kind

    @classmethod
    def not_a_tool(cls, value: t.Any) -> "ToolRegistryError":
        return cls(f"Expected a CoreTool, got {type(value).__name__}; wrap functions with @tool", "not_a_tool")

    @classmethod
    def duplicate_name(cls, name: str) -> "ToolRegistryError":
        return cls(f"Duplicate tool name: {name!r}", "duplicate_name")
