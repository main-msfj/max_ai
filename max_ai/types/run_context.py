"""
Run Context Management

This Module works a parent for agent running context
this is the value that change in running time
"""
from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue

from ..capabilities.tools.plan import AgentPlan
from ..core.compaction import CompactionState
from ..core.messages import Message
from ..core.tool.state import ToolState
from ..ids import short_id
from .chat_history import ChatHistory
from .runtime import RuntimeState


# -------- RUNNING CONTEXT -----------------------------------------------------------
class RunContext(BaseModel):
    """Run Context State"""

    user_id: str = Field(default="user_001")
    session_id: str | None = Field(default=None)
    run_id: str = Field(default_factory=short_id)
    messages: list[Message] = Field(default_factory=list) # type: ignore
    message_history: ChatHistory = Field(default_factory=ChatHistory)
    tool_state: ToolState = Field(default_factory=ToolState)
    runtime_state: RuntimeState = Field(default_factory=RuntimeState)
    plan: AgentPlan | None = Field(default=None)
    compaction: CompactionState = Field(default_factory=CompactionState)
    completion_state: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict,
        description=(
            "Per-gate evidence, keyed by CompletionBase.gate_id. Turn-scoped: "
            "reset when a new task starts, preserved across pause/resume."
        ),
    )
