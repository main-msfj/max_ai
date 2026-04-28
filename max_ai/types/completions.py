import typing as t
from pydantic import BaseModel, Field

from ..core.messages import AssistantMessage

# -------- MODEL USAGE -----------------------------------------------------------
class Usage(BaseModel):
    """Execution statistics and resource consumption for agent operations."""

    # Time and Call Metrics
    duration_ms: int = Field(default=0)
    retries: int = Field(default=0)
    llm_calls: int = Field(default=0)
    attempts_to_call_api: int = Field(default=0)

    # Tool Metrics
    tool_calls: int = Field(default=0)

    # Token Metrics
    tokens_input: int = Field(default=0)
    tokens_output: int = Field(default=0)
    tokens_cached: int = Field(default=0)

    def __add__(self, other: t.Self) -> t.Self:
        """Aggregate usage statics from multiple sources."""
        return self.__class__(
            duration_ms=max(self.duration_ms, other.duration_ms),
            llm_calls=self.llm_calls + other.llm_calls,
            retries=self.retries + other.retries,
            attempts_to_call_api=self.attempts_to_call_api + other.attempts_to_call_api,
            tool_calls=self.tool_calls + other.tool_calls,
            tokens_input=self.tokens_input + other.tokens_input,
            tokens_output=self.tokens_output + other.tokens_output,
            tokens_cached=self.tokens_cached + other.tokens_cached,
        )


# -------- CHAT COMPLETION -----------------------------------------------------------
class ChatCompletionResult(BaseModel):
    """Standardized response form BaseChatCompletion API."""

    message: AssistantMessage = Field(..., description="The LLM's response")
    usage: Usage = Field(..., description="Token consumption and timing metrics")
    model: str = Field(..., description="Actual model used for the request")
    finish_reason: str = Field(..., description="Completion status")


class ChatCompletionChunk(BaseModel):
    """Streaming response chunk for BaseChatCompletion API."""

    content: str = Field(..., description="Partial content from stream")
    thinking: str | None = Field(default=None, description="Reasoning")
    is_complete: bool = Field(..., description="Indicates if this is the final chunk")
    tool_call_chunk: dict[str, t.Any] | None = Field(default=None)
    usage: Usage | None = Field(default=None, description="Token usage statistics")
    structured_output: BaseModel | None = Field(default=None)

