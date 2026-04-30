import typing as t
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from .messages import CoreMessage

if t.TYPE_CHECKING:
    from ..types.routines import RoutineSummary


# -------- UNIT BLOCKS -----------------------------------------------------------
class RoutineBlocks(BaseModel):
    """Single block of routine context information"""

    name: str = Field(..., description="routine name")
    description: str = Field(..., description="routine description")
    instructions: str = Field(..., description="The full markdown instructions")

    def to_summary(self) -> "RoutineSummary":
        from ..types.routines import RoutineSummary

        return RoutineSummary(name=self.name, description=self.description)


class RunTimeBlock(BaseModel):
    """Single block of Run Time context information"""

    metadata: dict[str, t.Any] = Field(default_factory=dict)
    environment: dict[str, t.Any] = Field(default_factory=dict)
    shared_state: dict[str, t.Any] = Field(default_factory=dict)


class ChatHistoryBlock(BaseModel):
    """Single block of chat history."""

    message_history: list[CoreMessage] = Field(default_factory=list)  # type: ignore

    def iter_messages(self) -> t.Iterator[CoreMessage]:
        """Iterate over the messages in chronological order."""
        return iter(self.message_history)

    def __len__(self) -> int:
        return len(self.message_history)

    def __bool__(self) -> bool:
        return bool(self.message_history)


class MemoryBlock(BaseModel):
    """Single block of memory"""

    category: str = Field(..., description="Memory category")
    content: str = Field(..., description="Consolidated value for this memory fact")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeBlock(BaseModel):
    """Single block of external item"""

    content: str = Field(..., description="External Context")
    score: float | None = Field(default=None)
    tokens: int = Field(default=0)
    metadata: dict[str, t.Any] = Field(default_factory=dict)


class SkillBlock(BaseModel):
    """Single block of skill information"""

    name: str = Field(..., description="Skill name")
    description: str = Field(..., description="Skill description")
    instructions: str = Field(..., description="The full markdown instructions")


class ContextBlock(BaseModel):
    """A fragment of past conversation returned by ``search``."""

    session_id: str = Field(..., description="Session the fragment belongs to")
    timestamp: datetime = Field(..., description="When the session occurred")
    content: str = Field(..., description="Summary or fragment text")
    score: float | None = Field(default=None, description="Relevance score, if any")
    metadata: dict[str, t.Any] = Field(default_factory=dict)
