"""Claude through Anthropic's own API."""

from __future__ import annotations

import typing as t

from .._messages import MessagesClient
from ._model import AnthropicClientConfig


class AnthropicClient(MessagesClient[AnthropicClientConfig]):
    """Claude, called at Anthropic.

    Thinking: a model with ``supports_thinking`` thinks by itself and shows
    a summary. How deep is ``output_config``:
    ``ModelConfig(supports_thinking=True, thinking={"thinking": {"type": "adaptive",
    "display": "summarized"}, "output_config": {"effort": "high"}})``."""

    component_schema = AnthropicClientConfig
    component_type = "client"
    component_provider_override = "max_ai.capabilities.clients.anthropic.AnthropicClient"

    PROVIDER_NAME = "Anthropic"
    API_KEY_ENV = "ANTHROPIC_API_KEY"

    def _to_config(self) -> AnthropicClientConfig:
        return AnthropicClientConfig(
            model=self.model, api_key_env=self.api_key_env, base_url=self.base_url,
            options=self.options, config=self.config.model_copy(), retry=self.retry,
        )

    @classmethod
    def _from_config(cls, config: AnthropicClientConfig) -> t.Self:
        return cls(
            model=config.model, base_url=config.base_url, config=config.config,
            options=config.options, retry=config.retry, api_key_env=config.api_key_env,
        )
