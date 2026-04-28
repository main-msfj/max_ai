import typing as t


# -------- -----------------------------------------------------------
# CHAT HISTORY
# -------- -----------------------------------------------------------
class AgentError(Exception):
    def __init__(self, message: str | None):
        self.message = message
        super().__init__(message)

    @classmethod
    def wrong_type(cls, field_name: str, type: t.Any, expected: t.Any):
        return cls(f"{field_name} must be {expected} but {type} was passed")

    @classmethod
    def empty_value(cls, field_name: str):
        return cls(f"{field_name} cannot be empty ")

    @classmethod
    def layer_render_failed(cls, layer_name: str, error: Exception):
        return cls(f"Failed to render layer {layer_name}: {error}")

    @classmethod
    def not_prepared(cls, agent_name: str):
        return cls(f"Agent {agent_name!r} is not prepared. Call prepare() before use.")
    
    @classmethod
    def unresolved_approvals(cls, agent_name: str, tool_call_ids: list[str]) -> t.Self:
        return cls(
            f"Agent {agent_name!r} cannot resume: {len(tool_call_ids)} tool call(s) "
            f"still pending approval ({tool_call_ids}). Apply user decisions via "
            "ctx.tool_state.apply_approval() before calling resume()."
        )

    @classmethod
    def nothing_to_resume(cls, agent_name: str) -> t.Self:
        return cls(
            f"Agent {agent_name!r} cannot resume: no actionable tool calls and no "
            "stale executions. The previous run completed normally — use run() to "
            "start a new turn."
        )
