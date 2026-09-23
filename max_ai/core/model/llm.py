"""What an LLM model supports, shared by every chat completion client."""

from pydantic import BaseModel, Field

from ...config import setting


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

    tokenizer_base: str = Field(default_factory=lambda: setting.default_tokenizer)
