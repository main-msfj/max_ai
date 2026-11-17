"""OpenAI through its Chat Completions endpoint."""

from __future__ import annotations

import typing as t

from ....core.model.llm import ModelConfig
from ....core.retry import RetryPolicy
from .._chat_completions import ChatCompletionsClient
from ._model import OpenAIChatCompletionClientConfig


class OpenAIChatCompletionClient(ChatCompletionsClient[OpenAIChatCompletionClientConfig]):
    """OpenAI, or any service that speaks its Chat Completions format
    (give its ``base_url``).

    Thinking: ``ModelConfig(thinking={"reasoning_effort": "low"})``; the
    values are OpenAI's (none, minimal, low, medium, high, xhigh) and depend
    on the model. Some models refuse tools together with thinking on this
    endpoint: for them send ``{"reasoning_effort": "none"}``."""

    component_schema = OpenAIChatCompletionClientConfig
    component_type = "client"
    component_provider_override = "max_ai.capabilities.clients.openai.OpenAIChatCompletionClient"

    PROVIDER_NAME = "OpenAI"
    API_KEY_ENV = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        config: ModelConfig | None = None,
        *,
        options: dict[str, t.Any] | None = None,  # temperature, top_p...: sent as they are
        retry: RetryPolicy | None = None,
        api_key_env: str | None = None,
    ) -> None:
        super().__init__(
            model, api_key, base_url, config, options=options, retry=retry,
            api_key_env=api_key_env, organization=organization, project=project,
        )
        self.organization = organization
        self.project = project

    def _to_config(self) -> OpenAIChatCompletionClientConfig:
        return OpenAIChatCompletionClientConfig(
            model=self.model, api_key_env=self.api_key_env, base_url=self.base_url,
            organization=self.organization, project=self.project, options=self.options,
            config=self.config.model_copy(), retry=self.retry,
        )

    @classmethod
    def _from_config(cls, config: OpenAIChatCompletionClientConfig) -> t.Self:
        return cls(
            model=config.model, base_url=config.base_url, organization=config.organization,
            project=config.project, config=config.config, options=config.options,
            retry=config.retry, api_key_env=config.api_key_env,
        )
