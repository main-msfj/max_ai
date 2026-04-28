"""Chat History Context"""

import typing as t
from pydantic import field_validator

from ..core.messages import CoreMessage
from ..core.blocks import ChatHistoryBlock
from ..errors.context import ChatHistoryError


# -------- -----------------------------------------------------------
# CHAT HISTORY CONTEXT
# -------- -----------------------------------------------------------
class ChatHistory(ChatHistoryBlock):
    """Internal chat history for context management.

    Enforces that message_history contains CoreMessage objects, not raw dicts.
    To construct from serialized data (JSON, DB rows), use `load_from()`.
    """

    @field_validator("message_history", mode="before")
    @classmethod
    def _require_core_messages(cls, v: list[t.Any]) -> list[t.Any]:
        if not v:
            return v
        for i, item in enumerate(v):
            if isinstance(item, dict):
                raise ChatHistoryError.wrong_input(i)

            if not isinstance(item, CoreMessage):
                raise ChatHistoryError.wrong_type(i, type(item))
        return v

    @classmethod
    def load_from(
        cls,
        message_summary: str | None = None,
        message_history: list[dict[str, t.Any]] | None = None,
    ) -> t.Self:
        """Create ChatHistory from raw message dicts (e.g. from JSON/DB)."""
        parsed = [CoreMessage.parse_message(m) for m in (message_history or [])]
        return cls(
            message_summary=message_summary,
            message_history=parsed,
        )
