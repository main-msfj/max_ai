"""
Run Context Management

This Module works a parent for agent running context
this is the value that change in running time
"""
from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue

from ..capabilities.tools.plan import AgentPlan
from ..core.compaction import CompactionState
from ..core.ids import short_id
from ..core.messages import Message
from ..core.tool.state import ToolState
from .chat_history import ChatHistory

# from .runtime import RuntimeState
from .context import Context
from .summary import Summary
from .todos import Todo


# -------- RUNNING CONTEXT -----------------------------------------------------------
class RunContext(BaseModel):
    """Run Context State"""

    user_id: str = Field(default="user_001")
    session_id: str | None = Field(default=None)
    run_id: str = Field(default_factory=short_id)
    messages: list[Message] = Field(default_factory=list) # type: ignore
    message_history: ChatHistory = Field(default_factory=ChatHistory)
    tool_state: ToolState = Field(default_factory=ToolState)
    # Not used by the new tools; only the old agent reads it. Deleted with it.
    # runtime_state: RuntimeState = Field(default_factory=RuntimeState)
    plan: AgentPlan | None = Field(default=None)
    compaction: CompactionState = Field(default_factory=CompactionState)
    # Not used by the new tools; only the old agent reads it. Deleted with it.
    # file_reads: dict[str, tuple[int, int]] = Field(
    #     default_factory=dict,
    #     description=(
    #         "Files the model has read, as workspace path → (mtime_ns, size). "
    #         "Overwriting, editing or deleting a file needs a matching entry."
    #     ),
    # )
    completion_state: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict,
        description=(
            "Per-gate evidence, keyed by CompletionBase.gate_id. Turn-scoped: "
            "reset when a new task starts, preserved across pause/resume."
        ),
    )
    mcp_tasks: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)  # MCP tasks still running, by task id
    cwd: str | None = None  # where bash ended, when not the workspace
    files_seen: dict[str, str] = Field(default_factory=dict)  # file path -> sha256 the model saw
    todos: list[Todo] = Field(default_factory=list)  # the agent's list of things to do
    context: Context | None = None  # the model's window, part by part, in the last call
    summary: Summary | None = None  # what is kept of the messages compaction removed
    position: str | None = None  # folder and todos as they were when messages last left
    memory_index: str | None = None  # the memory index the model saw; renewed after compaction
    inbox: list[str] = Field(
        default_factory=list,
        description=(
            "Messages the user sent while a turn runs. The loop adds them to "
            "the conversation before its next model call; any left when the "
            "turn ends are the caller's to send as the next task."
        ),
    )
