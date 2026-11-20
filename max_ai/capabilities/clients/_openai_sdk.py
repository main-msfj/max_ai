"""What the clients built on OpenAI's SDK share, whatever format they speak."""

from __future__ import annotations

import base64
import logging
import typing as t

from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    DefaultAsyncHttpxClient,
    RateLimitError,
)
from pydantic import BaseModel

from ...base.clients import CoreLLMClient
from ...base.component import ConfigT
from ...core.messages import ImagePart, Message, SystemMessage
from ...core.model.llm import ModelConfig
from ...core.retry import RetryPolicy
from ...errors.client import ClientError
from ...types.run_context import RunContext

logger = logging.getLogger(__name__)


class OpenAISDKClient(CoreLLMClient[ConfigT]):
    """The connection, the errors and the helpers of a client that calls
    through OpenAI's SDK. Each format (Chat Completions, Responses) inherits
    it and writes its own messages, tools, call and answer."""

    DEFAULT_BASE_URL: t.ClassVar[str | None] = None

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        config: ModelConfig | None = None,
        *,
        options: dict[str, t.Any] | None = None,  # temperature, top_p...: sent as they are
        retry: RetryPolicy | None = None,
        api_key_env: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(model, api_key, config, retry=retry, api_key_env=api_key_env)
        if self.api_key is None:
            raise ValueError(
                f"{self.PROVIDER_NAME} needs an API key: pass api_key or set ${self.api_key_env}."
            )
        self.base_url = base_url
        self.options = dict(options or {})
        # The SDK never retries: the base class does, with its policy.
        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=base_url or self.DEFAULT_BASE_URL,
            organization=organization, project=project, default_headers=headers, max_retries=0,
        )

    async def aclose(self) -> None:
        await self.client.close()
        self.client = self.client.copy(http_client=DefaultAsyncHttpxClient())

    def format_messages(self, ctx: RunContext, prompts: str) -> list[Message]:
        """The system prompt first, then the conversation."""
        system = [SystemMessage(source="system", content=prompts)] if prompts.strip() else []
        return [*system, *ctx.messages]

    # -------- ERRORS ---------------------------------------------------------------
    # ``blocked`` is left to each service: how a request is refused for its
    # content is the one thing they do not share.
    def window_full(self, error: Exception) -> bool:
        text = str(error).lower()
        return isinstance(error, BadRequestError) and any(
            words in text for words in ("context_length", "context length", "context window")
        )

    def rate_limited(self, error: Exception) -> bool:
        return isinstance(error, RateLimitError)

    def wait_asked(self, error: Exception) -> float | None:
        headers = getattr(getattr(error, "response", None), "headers", None)
        return self.retry_after(headers)

    def other_error(self, error: Exception) -> ClientError:
        if isinstance(error, AuthenticationError):
            return ClientError.authentication_failed(str(error))
        if isinstance(error, APITimeoutError):
            return ClientError.request_timeout(0)
        if isinstance(error, BadRequestError):
            return ClientError.invalid_request(str(error))
        if isinstance(error, APIError):
            return ClientError.api_error(
                provider=self.PROVIDER_NAME, status=getattr(error, "status_code", None),
                detail=str(error),
            )
        return ClientError.unexpected(provider=self.PROVIDER_NAME, detail=str(error))

    # -------- HELPERS --------------------------------------------------------------
    @staticmethod
    def image_url(part: ImagePart) -> str:
        """The image's address, or the image itself as a data URL."""
        if part.url is not None:
            return part.url
        assert part.data is not None
        return f"data:{part.mime_type};base64,{base64.b64encode(part.data).decode()}"

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
