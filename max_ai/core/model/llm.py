"""What an LLM model supports, shared by every chat completion client."""

from pydantic import BaseModel, Field, field_validator

from ...config import setting

MIN_CONTEXT_WINDOW = 128_000
MAX_CONTEXT_WINDOW = 1_000_000


class ModelConfig(BaseModel):
    """Model Config"""

    name: str | None = Field(default=None, description="Official model name")
    max_context_window: int = Field(
        default=MIN_CONTEXT_WINDOW,
        description=(
            f"Context window in tokens, always within {MIN_CONTEXT_WINDOW:,}–"
            f"{MAX_CONTEXT_WINDOW:,}: unknown/0 or smaller values become the "
            "minimum, bigger ones the maximum."
        ),
    )
    max_output_tokens: int = Field(default=0)

    # Capabilities
    supports_vision: bool = Field(default=False)
    supports_audio: bool = Field(default=False)
    supports_function_calling: bool = Field(default=True)
    supports_thinking: bool = Field(default=False)

    tokenizer_base: str = Field(default_factory=lambda: setting.default_tokenizer)

    @field_validator("max_context_window", mode="before")
    @classmethod
    def _within_bounds(cls, value: object) -> int:
        window = int(value or 0)
        return min(MAX_CONTEXT_WINDOW, max(MIN_CONTEXT_WINDOW, window))
