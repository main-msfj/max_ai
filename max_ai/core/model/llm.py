"""What an LLM model supports, shared by every chat completion client."""

import typing as t

from pydantic import BaseModel, Field

from ...config import setting


class ModelConfig(BaseModel):
    """Model Config"""

    name: str | None = Field(default=None, description="Official model name")
    # Each model has its own limits: read them in the provider's page and set them.
    max_context_window: int = Field(default=128_000, gt=0, description="Context window, in tokens.")
    max_output_tokens: int = Field(
        default=0, ge=0, description="Output limit, in tokens; 0 sends none (the provider's own).",
    )
    # Prices in USD per million tokens; unset = cost unknown (no cost limits).
    input_cost_per_mtok: float | None = Field(default=None, ge=0)
    output_cost_per_mtok: float | None = Field(default=None, ge=0)

    # Capabilities
    supports_vision: bool = Field(default=False)
    supports_audio: bool = Field(default=False)
    supports_function_calling: bool = Field(default=True)
    supports_thinking: bool = Field(default=False)
    thinking: dict[str, t.Any] | None = Field(
        default=None,
        description=(
            "The provider's own fields that set how the model thinks, sent as they are "
            "in each request, e.g. {'reasoning_effort': 'low'}. None sends nothing."
        ),
    )

    tokenizer_base: str = Field(default_factory=lambda: setting.default_tokenizer)
