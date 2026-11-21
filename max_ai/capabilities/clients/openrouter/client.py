"""OpenRouter: many providers behind the Chat Completions format."""

from __future__ import annotations

import logging
import typing as t

from ....core.messages import AssistantMessage
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
    BLOCKED: t.ClassVar[str] = "request blocked"  # how a guardrail starts its message
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

    def reasoning_pieces(self, message: t.Any) -> list[dict[str, t.Any]]:
        """OpenRouter sends ``reasoning_details``: the text in parts, the
        encrypted reasoning whole. Seen live 2026-11-20."""
        details: list[t.Any] = self.get(message, "reasoning_details") or []
        return [dict(t.cast("dict[str, t.Any]", item)) for item in details if isinstance(item, dict)]

    def reasoning_back(self, message: AssistantMessage) -> dict[str, t.Any]:
        """Given back as they came, so the model goes on from its own reasoning."""
        kept = message.reasoning_state.get(self.PROVIDER_NAME.lower())
        return {"reasoning_details": kept} if kept else {}

    def blocked(self, error: Exception) -> list[str] | None:
        """A guardrail of the account refused the request: HTTP 403, "Request
        blocked: ...", and the patterns it found. Seen live 2026-11-20."""
        if getattr(error, "status_code", None) != 403 or self.BLOCKED not in str(error).lower():
            return None
        body = getattr(error, "body", None)
        inner = self.get(body, "error") or body
        found = t.cast("list[t.Any]", self.get(self.get(inner, "metadata") or {}, "patterns") or [])
        return [str(item) for item in found]

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
