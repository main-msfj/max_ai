"""Contract of a chat completion client: one provider behind the same interface."""

from __future__ import annotations

import asyncio
import logging
import os
import typing as t
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from pydantic import BaseModel

from ..core.messages import Message
from ..core.model.llm import ModelConfig
from ..core.retry import RetryPolicy, rate_limited, retrying
from ..errors.client import ClientError
from ..types.completions import LLMChunk, LLMResponse, Usage
from ..types.run_context import RunContext
from ..types.tools import CoreToolDefinition
from .component import ComponentBase, ConfigT
from .middleware import ModelCall

if t.TYPE_CHECKING:
    from ..core.middleware.chain import MiddlewareChain

logger = logging.getLogger(__name__)

# Told before each wait: the try that failed (from 1), its error, seconds to wait.
OnRetry = t.Callable[[int, BaseException, float], t.Any]


class CoreLLMClient(ComponentBase[ConfigT], ABC):
    """One provider's chat endpoint. Subclasses say how that provider wants
    its messages and tools, and make the call. How the model thinks is
    ``config.thinking``: the provider's own fields, sent as they are.

    A subclass names its config: ``CoreLLMClient[MyConfig]``. That
    is what lets ``_to_config`` and ``_from_config`` use ``MyConfig``."""

    PROVIDER_NAME: t.ClassVar[str] = "The provider"  # how its errors name it
    API_KEY_ENV: t.ClassVar[str | None] = None  # env var with the key; configs store this name
    RETRY: t.ClassVar[RetryPolicy] = RetryPolicy(attempts=3)  # when none is given

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        config: ModelConfig | None = None,
        *,
        retry: RetryPolicy | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self.model = model
        self.config = config or ModelConfig()
        self.retry = retry or self.RETRY

        # The key: given by hand, or read from the variable. Never stored in a config.
        self.api_key_env = api_key_env or self.API_KEY_ENV
        if api_key is None and self.api_key_env:
            api_key = os.getenv(self.api_key_env) or None
        self.api_key = api_key

    async def aclose(self) -> None:
        """Close open connections; a later call opens new ones."""

    # -------- PUBLIC -----------------------------------------------------------------
    async def run(
        self,
        messages: list[Message],
        tools: list[CoreToolDefinition] | None = None,
        output_format: type[BaseModel] | None = None,
        *,
        retry: RetryPolicy | None = None,
        on_retry: OnRetry | None = None,
        ctx: RunContext | None = None,
        middleware: MiddlewareChain | None = None,
    ) -> LLMResponse:
        """One answer for ``messages``. A passing failure (rate limit, timeout,
        5xx) is tried again; ``retry`` given here wins over the client's own.
        With ``middleware``, each middleware sees the call and its answer."""

        # 1. The call itself, as the middlewares left it.
        async def action(call: ModelCall) -> LLMResponse:
            api_messages, api_tools = self._request(call.messages, call.tools)

            async def once() -> LLMResponse:
                try:
                    return await self.complete(api_messages, api_tools, call.output_format)
                except Exception as error:
                    raise self.map_error(error) from error

            return await retrying(
                once, policy=retry or self.retry, transient=self.transient, on_retry=on_retry
            )

        # 2. The middlewares around it, when there are.
        call = ModelCall(messages, list(tools or []), output_format, self.model, self.config)
        if middleware and ctx is not None:
            return await middleware.model(ctx, call, action)
        return await action(call)

    async def stream(
        self,
        messages: list[Message],
        tools: list[CoreToolDefinition] | None = None,
        output_format: type[BaseModel] | None = None,
        *,
        retry: RetryPolicy | None = None,
        on_retry: OnRetry | None = None,
        ctx: RunContext | None = None,
        middleware: MiddlewareChain | None = None,
    ) -> t.AsyncGenerator[LLMChunk, None]:
        """The answer for ``messages``, piece by piece. It is tried again only
        while nothing has come out: after that the caller decides. With
        ``middleware``, each middleware sees it line by line."""
        call = ModelCall(messages, list(tools or []), output_format, self.model, self.config)
        if middleware and ctx is not None:
            pieces = middleware.stream(ctx, call, lambda given: self._attempts(given, retry, on_retry))
        else:
            pieces = self._attempts(call, retry, on_retry)
        async for chunk in pieces:
            yield chunk

    async def _attempts(
        self, call: ModelCall, retry: RetryPolicy | None, on_retry: OnRetry | None
    ) -> t.AsyncGenerator[LLMChunk, None]:
        """The streamed call, tried again while nothing has come out."""
        api_messages, api_tools = self._request(call.messages, call.tools)
        policy = retry or self.retry
        for attempt in range(policy.attempts):
            started = False
            try:
                async for chunk in self.complete_stream(api_messages, api_tools, call.output_format):
                    started = True
                    yield chunk
                return
            except Exception as raw:
                error = self.map_error(raw)
                # 1. Text already went out: repeating would show it twice.
                last = attempt + 1 >= policy.attempts
                if started or last or not self.transient(error):
                    raise error from raw
                # 2. Wait what the provider asked for, or the policy's step.
                wait = policy.delay(attempt, error.retry_after, rate_limited=rate_limited(error))
                if on_retry is not None:
                    on_retry(attempt + 1, error, wait)
                await asyncio.sleep(wait)

    @staticmethod
    def transient(error: BaseException) -> bool:
        """Calling again may work."""
        return bool(getattr(error, "transient", False))

    @staticmethod
    def retry_after(headers: t.Mapping[str, str] | None) -> float | None:
        """Seconds the provider asked to wait, from the standard headers. A
        provider that says it another way overrides this."""
        if headers is None:
            return None
        # 1. A number: milliseconds or seconds.
        for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
            try:
                seconds = float(headers.get(name, "")) / scale
            except ValueError:
                continue
            if seconds >= 0:
                return seconds
        # 2. A date: wait until then.
        try:
            until = parsedate_to_datetime(headers.get("retry-after", ""))
        except (TypeError, ValueError):
            return None
        return max(0.0, (until - datetime.now(timezone.utc)).total_seconds())

    def _request(
        self, messages: list[Message], tools: list[CoreToolDefinition] | None
    ) -> tuple[list[dict[str, t.Any]], list[dict[str, t.Any]]]:
        """Messages and tools in the provider's format."""
        if tools and not self.config.supports_function_calling:
            raise ClientError.tools_not_supported(model=self.model, tool_count=len(tools))
        return self.build_api_messages(messages), self.build_tool_schema(tools or [])

    # -------- HELPERS --------------------------------------------------------------
    @staticmethod
    def structured(content: str, output_format: type[BaseModel] | None) -> BaseModel | None:
        """``content`` as ``output_format``, or None when it does not fit."""
        if output_format is None or not content:
            return None
        try:
            return output_format.model_validate_json(content)
        except ValueError:
            logger.warning("The answer does not fit %s", output_format.__name__)
            return None

    @classmethod
    def strict(cls, node: t.Any) -> t.Any:
        """A JSON schema as strict mode wants it: no defaults, every property
        required, no extra properties."""
        if isinstance(node, list):
            return [cls.strict(item) for item in t.cast("list[t.Any]", node)]
        if not isinstance(node, dict):
            return node
        schema = t.cast("dict[str, t.Any]", node)
        cleaned: dict[str, t.Any] = {
            key: cls.strict(value) for key, value in schema.items() if key != "default"
        }
        properties: t.Any = cleaned.get("properties")
        if cleaned.get("type") == "object" or isinstance(properties, dict):
            cleaned["additionalProperties"] = False
            if isinstance(properties, dict):
                cleaned["required"] = list(t.cast("dict[str, t.Any]", properties))
        return cleaned

    @staticmethod
    def get(source: t.Any, key: str) -> t.Any:
        """A field of an SDK object or of a plain dictionary."""
        if isinstance(source, dict):
            return t.cast("dict[str, t.Any]", source).get(key)
        return getattr(source, key, None)

    # -------- EACH PROVIDER ----------------------------------------------------------
    @abstractmethod
    def format_messages(self, ctx: RunContext, prompts: str) -> list[Message]:
        """The conversation to send: the system prompt (what ``PromptBuilder``
        built) and ctx's messages, in this provider's order."""

    @abstractmethod
    def build_api_messages(self, messages: list[Message]) -> list[dict[str, t.Any]]:
        """``messages`` in the provider's format."""

    @abstractmethod
    def build_tool_schema(self, tools: list[CoreToolDefinition]) -> list[dict[str, t.Any]]:
        """``tools`` in the provider's format."""

    def map_error(self, error: Exception) -> ClientError:
        """The provider's error as a ``ClientError``. The order is fixed here;
        each provider only says how it recognizes each case."""
        if isinstance(error, ClientError):
            return error

        # 1. The cases the framework reacts to: every provider tells them apart.
        found = self.blocked(error)
        if found is not None:
            return ClientError.blocked(self.PROVIDER_NAME, found)
        if self.window_full(error):
            return ClientError.token_limit_exceeded()
        if self.rate_limited(error):
            return ClientError.rate_limit_exceeded(self.wait_asked(error))

        # 2. The rest, as the provider reads them.
        return self.other_error(error)

    @abstractmethod
    def blocked(self, error: Exception) -> list[str] | None:
        """The provider refused the request for its content: what it says it
        found (may be empty). None when ``error`` is something else. The
        loop ends the turn and takes the content out of the conversation."""

    @abstractmethod
    def window_full(self, error: Exception) -> bool:
        """The request did not fit in the model's window. The loop compacts
        by force and asks again."""

    @abstractmethod
    def rate_limited(self, error: Exception) -> bool:
        """Too many requests. The call is tried again after a wait."""

    def wait_asked(self, error: Exception) -> float | None:
        """Seconds the provider asked to wait after a rate limit; None when
        it did not say."""
        return None

    @abstractmethod
    def other_error(self, error: Exception) -> ClientError:
        """Any other error: authentication, a timeout, a failure of the provider."""

    @abstractmethod
    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        """The provider's token counts as ``Usage``."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]],
        output_format: type[BaseModel] | None,
    ) -> LLMResponse:
        """Make the call and return the whole answer."""

    @abstractmethod
    def complete_stream(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]],
        output_format: type[BaseModel] | None,
    ) -> t.AsyncGenerator[LLMChunk, None]:
        """Make the call and yield the answer as it arrives."""
