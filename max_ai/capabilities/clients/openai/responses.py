"""OpenAI through its Responses endpoint."""

from __future__ import annotations

import typing as t

from ....core.model.llm import ModelConfig
from ....core.retry import RetryPolicy
from .._responses import ResponsesClient
from ._model import OpenAIResponsesClientConfig


class OpenAIResponsesClient(ResponsesClient[OpenAIResponsesClientConfig]):
    """OpenAI's newer models: they reason and call tools in the same turn,
    which its Chat Completions endpoint refuses.

    Thinking: ``ModelConfig(thinking={"reasoning": {"effort": "low", "summary": "auto"}})``;
    with ``summary`` the model's thinking is shown as it works."""

    component_schema = OpenAIResponsesClientConfig
    component_type = "client"
    component_provider_override = "max_ai.capabilities.clients.openai.OpenAIResponsesClient"

    PROVIDER_NAME = "OpenAI"
    API_KEY_ENV = "OPENAI_API_KEY"
    BLOCKED_CODE: t.ClassVar[str] = "content_filter"  # the error code of a refused request

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

    def blocked(self, error: Exception) -> list[str] | None:
        """The request was refused for its content. Written from Azure's
        documentation (HTTP 400, ``code: "content_filter"``): OpenAI's own
        pages give no shape for it. Not seen live."""
        if getattr(error, "code", None) != self.BLOCKED_CODE:
            return None
        return []

    def _to_config(self) -> OpenAIResponsesClientConfig:
        return OpenAIResponsesClientConfig(
            model=self.model, api_key_env=self.api_key_env, base_url=self.base_url,
            organization=self.organization, project=self.project, options=self.options,
            config=self.config.model_copy(), retry=self.retry,
        )

    @classmethod
    def _from_config(cls, config: OpenAIResponsesClientConfig) -> t.Self:
        return cls(
            model=config.model, base_url=config.base_url, organization=config.organization,
            project=config.project, config=config.config, options=config.options,
            retry=config.retry, api_key_env=config.api_key_env,
        )
