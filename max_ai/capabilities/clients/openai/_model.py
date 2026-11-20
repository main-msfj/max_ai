"""What is stored of an OpenAI client. Never the key."""

import typing as t

from pydantic import BaseModel, Field

from ....core.model.llm import ModelConfig
from ....core.retry import RetryPolicy


class OpenAIChatCompletionClientConfig(BaseModel):
    model: str
    api_key_env: str | None = "OPENAI_API_KEY"  # the variable's name, never the key
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    options: dict[str, t.Any] = Field(default_factory=dict[str, t.Any])  # sent as they are in each request
    config: ModelConfig | None = None
    retry: RetryPolicy | None = None


class OpenAIResponsesClientConfig(OpenAIChatCompletionClientConfig):
    """The same fields: only the endpoint the client calls is different."""
