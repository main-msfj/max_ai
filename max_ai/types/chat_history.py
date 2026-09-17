"""Chat History Context"""

import typing as t
from pydantic import field_validator

from ..core.messages import CoreMessage
from ..core.blocks import ChatHistoryBlock
from ..errors.context import ChatHistoryError


# -------- CHAT HISTORY -----------------------------------------------------------
class ChatHistory(ChatHistoryBlock):
    """Internal chat history for context management.

    Accepts ``CoreMessage`` instances directly and parses raw dicts
    (e.g. from ``model_validate_json`` when a persisted ``RunContext``
    is rehydrated) through the discriminated ``Message`` union, so the
    concrete subtypes survive a round trip. Anything else is rejected.
    """

    @field_validator("message_history", mode="before")
    @classmethod
    def _coerce_core_messages(cls, v: list[t.Any]) -> list[t.Any]:
        if not v:
            return v
        coerced: list[CoreMessage] = []
        for i, item in enumerate(v):
            if isinstance(item, CoreMessage):
                coerced.append(item)
                continue
            if isinstance(item, dict):
                coerced.append(CoreMessage.parse_msg(item))
                continue
            raise ChatHistoryError.wrong_type(i, type(item))
        return coerced

    @classmethod
    def load_from(
        cls,
        message_history: list[dict[str, t.Any]] | None = None,
    ) -> t.Self:
        """Create ChatHistory from raw message dicts (e.g. from JSON/DB)."""
        parsed = [CoreMessage.parse_msg(m) for m in (message_history or [])]
        return cls(message_history=parsed)
