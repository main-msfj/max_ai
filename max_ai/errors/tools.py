import typing as t


class ToolRetry(Exception):
    """Signal the LLM to retry with corrected parameters."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class BaseToolError(Exception):
    """Raised for invalid tool instances."""

    def __init__(self, tool: t.Any):
        super().__init__(
            f"Invalid tool type: {type(tool)}. Must be BaseTool or @tool callable."
        )


class BaseToolDuplicateError(Exception):
    """Raised when duplicate tool names are detected."""

    def __init__(self, names: t.Set[str]):
        super().__init__(f"Duplicate tool names: {', '.join(names)}")


class DockerToolReferenceError(TypeError):
    """Raised when a tool cannot be referenced for Docker execution."""

    def __init__(self, tool_name: str, reason: str):
        super().__init__(
            f"Tool {tool_name!r} cannot run in Docker because {reason}."
        )
