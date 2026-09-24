import typing as t


class CompactionError(Exception):
    """Raised when a compaction strategy returns a window the model would reject."""

    def __init__(self, message: str):
        super().__init__(message)

    @classmethod
    def orphan_tool_result(cls, index: int, tool_call_id: str) -> t.Self:
        return cls(
            f"messages[{index}] answers tool call {tool_call_id!r}, but the "
            "assistant message that made that call is not right before it."
        )

    @classmethod
    def unanswered_tool_calls(cls, index: int, tool_call_ids: t.Iterable[str]) -> t.Self:
        missing = ", ".join(repr(i) for i in sorted(tool_call_ids))
        return cls(
            f"messages[{index}] calls tools {missing} but their results are "
            "missing: a strategy must keep or drop a tool call with its results."
        )
