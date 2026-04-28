"""
Run Context Management

This Module works a parent for agent running context
this is the value that change in running time
"""

import uuid
from pydantic import BaseModel, Field

from ..core.messages import CoreMessage
from ..core.tool_state import ToolState

from .runtime import RuntimeState
from .messages import ChatHistory

# -------- RUNNING CONTEXT -----------------------------------------------------------
class RunContext(BaseModel):
    """Run Context State"""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[CoreMessage] = Field(default_factory=list)
    message_history: ChatHistory = Field(default_factory=ChatHistory)
    tool_state: ToolState = Field(default_factory=ToolState)  # Missing
    runtime_state: RuntimeState = Field(default_factory=RuntimeState)
