import typing as t
from typing import Literal
from pydantic import BaseModel, Field, SecretStr

OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
OllamaThinkingEffort = Literal["low", "medium", "high"]
OllamaThink = bool | OllamaThinkingEffort


# -------- STACKS -----------------------------------------------------------
class StackConfig(BaseModel):
    """Core Stack Config"""

    name: str = Field(...)
    instructions: str | None = Field(default=None)
    description: str = Field(...)
    version: str = Field(default="1.0.0")
    is_edited: bool = Field(default=False)
    layer_class: str = Field(...)


# -------- AGENT CONFIG -----------------------------------------------------------
class AgentConfig(BaseModel):
    """Global configuration for agent behavior and execution limits."""

    summarize_tool_result: bool = Field(default=True)
    enable_self_reflection: bool = Field(default=False)
    max_loop_iterations: int = Field(default=10)
    tool_timeout: int = Field(default=300)
    tool_call_concurrency: int = Field(default=5)
    max_connection_retries: int = Field(default=3)
    exponential_backoff_base: float = Field(default=1.0)


# -------- MODEL CONFIG -----------------------------------------------------------
class ModelConfig(BaseModel):
    """Model Config"""

    name: str | None = Field(default=None, description="Official model name")
    max_context_window: int = Field(default=0)
    max_output_tokens: int = Field(default=0)

    # Capabilities
    supports_vision: bool = Field(default=False)
    supports_audio: bool = Field(default=False)
    supports_function_calling: bool = Field(default=True)
    supports_thinking: bool = Field(default=False)

    tokenizer_base: str = "o200k_base"


class OllamaChatCompletionClientConfig(BaseModel):
    """Configuration for OllamaChatCompletionClient serialization."""

    model: str
    host: str
    api_key: SecretStr | None = None
    think: OllamaThink | None = None
    keep_alive: str | int | None = None
    options: dict[str, t.Any] = Field(default_factory=dict)
    config: dict[str, t.Any] = Field(default_factory=dict)
