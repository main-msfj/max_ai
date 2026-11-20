"""OpenAI through its Chat Completions endpoint."""

from __future__ import annotations

import typing as t

from pydantic import BaseModel

from ....core.model.llm import ModelConfig
from ....core.retry import RetryPolicy
from ....errors.client import ClientError
from .._chat_completions import ChatCompletionsClient
from ._model import OpenAIChatCompletionClientConfig


class OpenAIChatCompletionClient(ChatCompletionsClient[OpenAIChatCompletionClientConfig]):
    """OpenAI, or any service that speaks its Chat Completions format
    (give its ``base_url``).

    Thinking: ``ModelConfig(thinking={"reasoning_effort": "low"})``; the
    values are OpenAI's (none, minimal, low, medium, high, xhigh) and depend
    on the model. The models in ``NO_TOOLS_WITH_THINKING`` refuse tools
    together with thinking on this endpoint: that request fails here, unsent."""

    component_schema = OpenAIChatCompletionClientConfig
    component_type = "client"
    component_provider_override = "max_ai.capabilities.clients.openai.OpenAIChatCompletionClient"

    PROVIDER_NAME = "OpenAI"
    BLOCKED_CODE: t.ClassVar[str] = "content_filter"  # the error code of a refused request
    API_KEY_ENV = "OPENAI_API_KEY"
    # Asked to OpenAI on 2026-11-17, model by model. Subject to change: OpenAI
    # moves models between endpoints; a name not listed here is sent as it is.
    NO_TOOLS_WITH_THINKING: t.ClassVar[frozenset[str]] = frozenset({
        "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5",
        "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
        "gpt-6-astra", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra",
    })
    TOOLS_WITH_THINKING = (
        "{model} does not take tools together with thinking on OpenAI's Chat "
        "Completions (reasoning_effort={effort!r}). Set "
        "thinking={{'reasoning_effort': 'none'}}, or use the model through another "
        "client, such as OpenRouter. This list of models is subject to change."
    )

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

    def request(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]],
        output_format: type[BaseModel] | None,
        *,
        stream: bool,
    ) -> dict[str, t.Any]:
        """The body of the call; tools with thinking stop here for the listed models."""
        effort = self.body().get("reasoning_effort")
        listed = self.base_url is None and self.model in self.NO_TOOLS_WITH_THINKING
        if tools and listed and effort not in (None, "none"):
            raise ClientError.invalid_request(
                self.TOOLS_WITH_THINKING.format(model=self.model, effort=effort)
            )
        return super().request(messages, tools, output_format, stream=stream)

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
