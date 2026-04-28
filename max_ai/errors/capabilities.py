import typing as t


class CapabilityError(Exception):
    """Raised when registry construction fails validation."""

    def __init__(self, message: str):
        super().__init__(message)

    @classmethod
    def invalid_tool_entry(cls, tool: t.Any):
        return cls(
            "Invalid tool entry: expected BaseTool or callable, "
            f"got {type(tool).__name__}"
        )

    @classmethod
    def invalid_priority_tool(cls, missing: list[str], tool_names: set[str]):
        return cls(
            f"priority_tools reference unknown tools: {missing}. "
            f"Available: {sorted(tool_names)}"
        )

    @classmethod
    def duplicate_tool_names(cls, duplicates: set[str]):
        return cls(f"Duplicate tool names: {sorted(duplicates)}")

    @classmethod
    def not_prepared(cls):
        return cls("Capability registry not prepared. Call prepare() before calling this method.")
