"""
Run Context Management

This Module works a parent for agent running context
this is the value that change in running time
"""

import uuid
from pydantic import BaseModel, Field

from ..core.messages import Message
from ..core.tool_state import ToolState

from .runtime import RuntimeState
from .chat_history import ChatHistory

# -------- RUNNING CONTEXT -----------------------------------------------------------
class RunContext(BaseModel):
    """Run Context State"""

    user_id: str = Field(default="default")
    session_id: str | None = Field(default=None)
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[Message] = Field(default_factory=list) # type: ignore
    message_history: ChatHistory = Field(default_factory=ChatHistory)
    tool_state: ToolState = Field(default_factory=ToolState)
    runtime_state: RuntimeState = Field(default_factory=RuntimeState)
