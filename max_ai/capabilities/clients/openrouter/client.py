"""OpenRouter: many providers behind the Chat Completions format."""

from __future__ import annotations

import logging
import typing as t

import httpx

from ....core.model.llm import ModelConfig
from ....core.retry import RetryPolicy
from ....errors.client import ClientError
from .._chat_completions import ChatCompletionsClient
from ._model import OpenRouterChatCompletionClientConfig

logger = logging.getLogger(__name__)


class OpenRouterChatCompletionClient(ChatCompletionsClient[OpenRouterChatCompletionClientConfig]):
    """Any model OpenRouter serves, named ``provider/model``.

    Thinking: ``ModelConfig(thinking={"reasoning": {"effort": "low"}})``;
    OpenRouter also takes ``{"max_tokens": 2000}`` or ``{"enabled": False}``
    inside ``reasoning``. The thinking text comes back in the answer.

    ``fallback_models`` are tried in order when the first one fails.
    ``provider`` is OpenRouter's routing: which providers serve the model
    (``{"order": ["anthropic"], "allow_fallbacks": False}``)."""

    component_schema = OpenRouterChatCompletionClientConfig
    component_type = "client"
    component_provider_override = (
        "max_ai.capabilities.clients.openrouter.OpenRouterChatCompletionClient"
    )

    PROVIDER_NAME = "OpenRouter"
    API_KEY_ENV = "OPENROUTER_API_KEY"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        config: ModelConfig | None = None,
        *,
        fallback_models: list[str] | None = None,
        provider: dict[str, t.Any] | None = None,
        app_name: str | None = None,  # shown in OpenRouter's rankings
        app_url: str | None = None,
        options: dict[str, t.Any] | None = None,  # temperature, top_p...: sent as they are
        retry: RetryPolicy | None = None,
        api_key_env: str | None = None,
    ) -> None:
        headers = {"HTTP-Referer": app_url or "", "X-Title": app_name or ""}
        super().__init__(
            model, api_key, base_url, config, options=options, retry=retry,
            api_key_env=api_key_env,
            headers={name: value for name, value in headers.items() if value},
        )
        self.fallback_models = list(fallback_models or [])
        self.provider = provider
        self.app_name = app_name
        self.app_url = app_url

    def _to_config(self) -> OpenRouterChatCompletionClientConfig:
        return OpenRouterChatCompletionClientConfig(
            model=self.model, api_key_env=self.api_key_env, base_url=self.base_url,
            fallback_models=self.fallback_models, provider=self.provider,
            app_name=self.app_name, app_url=self.app_url, options=self.options,
            config=self.config.model_copy(), retry=self.retry,
        )

    @classmethod
    def _from_config(cls, config: OpenRouterChatCompletionClientConfig) -> t.Self:
        return cls(
            model=config.model, base_url=config.base_url, config=config.config,
            fallback_models=config.fallback_models, provider=config.provider,
            app_name=config.app_name, app_url=config.app_url, options=config.options,
            retry=config.retry, api_key_env=config.api_key_env,
        )

    # -------- WHAT OPENROUTER DOES DIFFERENTLY -----------------------------------------
    def body(self) -> dict[str, t.Any]:
        """The fallback models and the routing, then thinking and options."""
        extra: dict[str, t.Any] = {}
        if self.fallback_models:
            extra["models"] = [self.model, *self.fallback_models]
        if self.provider is not None:
            extra["provider"] = self.provider
        return {**extra, **super().body()}

    def thinking(self, message: t.Any) -> str | None:
        """OpenRouter sends the thinking text as ``reasoning``."""
        return self.get(message, "reasoning") or None

    def check(self, payload: t.Any) -> None:
        """OpenRouter can answer 200 with the error inside: a provider failed
        behind it. Raise it like any other error."""
        error = self.get(payload, "error")
        if not error:
            return
        # 1. What failed, with the provider's own words when they come.
        code = self.get(error, "code")
        detail = str(self.get(error, "message") or error)
        if raw := self.get(self.get(error, "metadata") or {}, "raw"):
            detail = f"{detail}: {raw}"
        # 2. A rate limit is told apart, so it is tried again after a wait.
        if code == 429:
            raise ClientError.rate_limit_exceeded()
        raise ClientError.api_error(
            provider=self.PROVIDER_NAME, status=code if isinstance(code, int) else None,
            detail=detail,
        )

    @classmethod
    def context_window(
        cls, models: t.Sequence[str], *, base_url: str | None = None, timeout: float = 10
    ) -> int:
        """The smallest context window among ``models``, read from
        OpenRouter's public list; 0 when it cannot be read."""
        url = (base_url or cls.DEFAULT_BASE_URL or "").rstrip("/") + "/models"
        try:
            answer = httpx.get(url, timeout=timeout).raise_for_status()
            listed: list[dict[str, t.Any]] = answer.json()["data"]
        except (httpx.HTTPError, KeyError, ValueError) as error:
            logger.warning("Could not read OpenRouter model windows: %s", error)
            return 0
        windows = {model.get("id"): int(model.get("context_length") or 0) for model in listed}
        sizes = [windows[model] for model in models if windows.get(model)]
        return min(sizes) if sizes else 0
