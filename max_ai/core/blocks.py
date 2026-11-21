import typing as t

from pydantic import BaseModel, Field

from .messages import Message

# -------- UNIT BLOCKS -----------------------------------------------------------

class ChatHistoryBlock(BaseModel):
    """Single block of chat history.

    Typed as the discriminated ``Message`` union (not the ``CoreMessage``
    base) so deserialization restores the concrete subtypes — an
    ``AssistantMessage`` keeps its ``tool_calls``, a ``ToolMessage`` its
    ``tool_call_id``. With the base class, ``model_validate_json`` would
    silently collapse every entry into a bare ``CoreMessage``.
    """

    message_history: list[Message] = Field(default_factory=list)  # type: ignore

    def __len__(self) -> int:
        return len(self.message_history)

    def __bool__(self) -> bool:
        return bool(self.message_history)


class KnowledgeBlock(BaseModel):
    """Single block of external item"""

    content: str = Field(..., description="External Context")
    tokens: int = Field(default=0)
    metadata: dict[str, t.Any] = Field(default_factory=dict)
