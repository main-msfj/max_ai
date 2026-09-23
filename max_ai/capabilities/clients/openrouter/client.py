"""OpenRouter client — OpenAI Chat Completions wire format, different gateway."""

from __future__ import annotations

import os
import logging
import typing as t

import httpx
from pydantic import BaseModel, SecretStr

from ....core.model.llm import ModelConfig
from ....errors.client import ClientError
from ....types.completions import ChatCompletionChunk, ChatCompletionResult
from ..openai.client import OpenAIChatCompletionClient
from ._model import OpenRouterChatCompletionClientConfig


logger = logging.getLogger(__name__)


class OpenRouterChatCompletionClient(OpenAIChatCompletionClient):
    """OpenRouter implementation on top of ``OpenAIChatCompletionClient``.

    OpenRouter speaks the OpenAI Chat Completions protocol, so messages,
    tools, usage and streaming are inherited. What differs:

    - Reasoning comes back as ``message.reasoning`` / ``delta.reasoning``
      and is surfaced as ``thinking``.
    - Routing options (``models`` fallback list, ``reasoning``,
      ``provider``) aren't OpenAI SDK params, so they travel in
      ``extra_body``.
    - Errors can arrive inside a 200 response or mid-stream as an
      ``error`` object instead of an HTTP status.

    ``fallback_models`` is what makes ``:free`` models usable for
    continuous testing: when the primary is rate-limited upstream,
    OpenRouter tries the next one in the same request.
    """

    component_schema = OpenRouterChatCompletionClientConfig
    component_provider_override = "maxai.llm.OpenRouterChatCompletionClient"

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    PROVIDER_NAME = "OpenRouter"
    API_KEY_ENV = "OPENROUTER_API_KEY"

    def __init__(
        self,
        model: str,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        *,
        fallback_models: t.Sequence[str] | None = None,
        reasoning: dict[str, t.Any] | None = None,
        provider: dict[str, t.Any] | None = None,
        app_name: str | None = None,
        app_url: str | None = None,
        config: ModelConfig | None = None,
        max_tokens: int | None = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Args:
            model: OpenRouter model id, e.g. ``"qwen/qwen3.8-27b:free"``.
            api_key: Defaults to ``$OPENROUTER_API_KEY``. Never falls back
                to ``OPENAI_API_KEY`` — that key must not reach OpenRouter.
            fallback_models: Tried in order when ``model`` fails
                (rate limit, downtime, context too small).
            reasoning: OpenRouter reasoning config, e.g.
                ``{"effort": "low"}``, ``{"max_tokens": 500}`` or
                ``{"enabled": False}``.
            provider: OpenRouter provider-routing preferences.
            app_name / app_url: Optional attribution headers
                (``X-Title`` / ``HTTP-Referer``).
        """
        api_key = api_key or os.getenv(self.API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"OpenRouter needs an API key: pass api_key or set ${self.API_KEY_ENV}."
            )
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            config=config,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.fallback_models = list(fallback_models or [])
        self.reasoning = reasoning
        self.provider = provider
        self.app_name = app_name
        self.app_url = app_url

        headers: dict[str, str] = {}
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_name:
            headers["X-Title"] = app_name
        if headers:
            self.client = self.client.with_options(default_headers=headers)

    @classmethod
    def fetch_context_window(
        cls, models: t.Sequence[str], *, base_url: str | None = None, timeout: float = 10,
    ) -> int:
        """Smallest context window among ``models`` found in OpenRouter's
        public ``/models`` list (any of them may answer when fallbacks are
        set). Returns 0 (unknown) when the list can't be read or none of the
        models is listed; both cases are logged."""
        url = f"{(base_url or cls.DEFAULT_BASE_URL).rstrip('/')}/models"
        try:
            data = httpx.get(url, timeout=timeout).raise_for_status().json()["data"]
        except (httpx.HTTPError, KeyError, ValueError) as error:
            logger.warning("Could not read OpenRouter model windows: %s", error)
            return 0
        windows = {m.get("id"): m.get("context_length") or 0 for m in data}
        sizes = [windows[model] for model in models if windows.get(model)]
        missing = [model for model in models if not windows.get(model)]
        if missing:
            logger.warning("No context window listed for %s", ", ".join(missing))
        return min(sizes) if sizes else 0

    def _to_config(self) -> OpenRouterChatCompletionClientConfig:  # type: ignore[override]
        return OpenRouterChatCompletionClientConfig(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            fallback_models=self.fallback_models,
            reasoning=self.reasoning,
            provider=self.provider,
            app_name=self.app_name,
            app_url=self.app_url,
            options=self.generation_options,
            config=self.config.model_dump(exclude_none=True),
        )

    @classmethod
    def _from_config(  # type: ignore[override]
        cls,
        config: OpenRouterChatCompletionClientConfig,
    ) -> "OpenRouterChatCompletionClient":
        model_config = ModelConfig(**config.config) if config.config else None
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            fallback_models=config.fallback_models,
            reasoning=config.reasoning,
            provider=config.provider,
            app_name=config.app_name,
            app_url=config.app_url,
            config=model_config,
            **config.options,
        )

    def _extract_thinking(self, message_or_delta: t.Any) -> str | None:
        return self._get(message_or_delta, "reasoning", None) or None

    def _build_request(self, **kwargs: t.Any) -> dict[str, t.Any]:
        request = super()._build_request(**kwargs)
        body = dict(request.pop("extra_body", None) or {})
        if self.fallback_models:
            body.setdefault("models", [self.model, *self.fallback_models])
        if self.reasoning is not None:
            body.setdefault("reasoning", self.reasoning)
        if self.provider is not None:
            body.setdefault("provider", self.provider)
        if body:
            request["extra_body"] = body
        return request

    def _parse_response(
        self,
        response: t.Any,
        output_format: t.Type[BaseModel] | None,
        duration_ms: int,
    ) -> ChatCompletionResult:
        self._raise_body_error(response)
        return super()._parse_response(response, output_format, duration_ms)

    async def _iter_chunks(
        self,
        raw_stream: t.AsyncIterator[t.Any],
        output_format: t.Type[BaseModel] | None,
        start_time: float,
    ) -> t.AsyncGenerator[ChatCompletionChunk, None]:
        async def checked() -> t.AsyncIterator[t.Any]:
            async for chunk in raw_stream:
                self._raise_body_error(chunk)
                yield chunk

        async for chunk in super()._iter_chunks(checked(), output_format, start_time):
            yield chunk

    def _raise_body_error(self, payload: t.Any) -> None:
        """Errors OpenRouter reports in the body instead of the HTTP status."""
        error = self._get(payload, "error", None)
        if not error:
            return
        code = self._get(error, "code", None)
        detail = str(self._get(error, "message", error))
        metadata: t.Any = self._get(error, "metadata", None) or {}
        raw = self._get(metadata, "raw", None)
        if raw:
            detail = f"{detail}: {raw}"
        if code == 429:
            raise ClientError.rate_limit_exceeded()
        raise ClientError.api_error(
            provider=self.PROVIDER_NAME,
            status=code if isinstance(code, int) else None,
            detail=detail,
        )
