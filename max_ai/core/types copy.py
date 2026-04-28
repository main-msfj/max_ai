
# from __future__ import annotations
import typing as t
from pathlib import Path
from pydantic import BaseModel, Field

from ..core.messages import AssistantMessage
from .blocks import SkillBlock
    # from ..base.tools import CoreTool

# -------- PROMPT CONTEXT -----------------------------------------------------------
class PromptContext(BaseModel):
    """Rendered prompt layers ready for the LLM client.
    
    Produced by agent.prepare(); consumed by the client's
    format_messages(). The client is responsible for ordering and
    concatenating these strings according to its provider's
    conventions.
    """
    
    rendered_layers: dict[type, str] = Field(
        default_factory=dict,
        description="Layer type rendered string. Empty strings are valid.",
    )

# -------- MODEL USAGE -----------------------------------------------------------
class Usage(BaseModel):
    """Execution statistics and resource consumption for agent operations."""

    # Time and Call Metrics
    duration_ms: int = Field(0)
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


# -------- SKILLS -----------------------------------------------------------
class ResourceMeta(BaseModel):
    """Metadata for an auxiliary resource file inside a skill package.

    Resources are the ``.md`` files a skill ships alongside its
    SKILL.md — style guides, examples, reference material. They are
    exposed to the LLM on demand via the ``read_skill_resource`` tool
    rather than injected into the prompt upfront.
    """

    filename: str = Field(..., description="Resource filename (e.g. 'style_guide.md')")
    path: Path = Field(..., description="Absolute path to the resource file")
    description: str = Field(..., description="What the resource contains")


class Skill(BaseModel):
    """A fully-loaded skill ready to be plugged into an agent.

    Produced by a ``CoreSkillRegistry.load()`` call. Carries everything
    the agent needs: the prompt block to inject, the tools discovered
    from its scripts, and the resources catalog for on-demand reads.
    """

    model_config = {"arbitrary_types_allowed": True}

    block: SkillBlock = Field(..., description="Prompt block for the SkillsLayer")
    tools: list[t.Any] = Field(  # type: ignore 
        default_factory=list,
        description="Tools auto-wrapped from the skill's script functions",
    )
    resources: dict[str, ResourceMeta] = Field(
        default_factory=dict,
        description="Auxiliary resources keyed by filename",
    )

    @property
    def name(self) -> str:
        return self.block.name

# -------- ROUTINES -----------------------------------------------------------
class RoutineSummary(BaseModel):
    name: str = Field(..., description="Routine identifier")
    description: str = Field(..., description="Short description")
