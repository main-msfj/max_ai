"""OpenAI Chat Completions client implementation."""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import typing as t

from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from pydantic import BaseModel, SecretStr

from ....base.clients import CoreChatCompletionClient
from ....base.component import Component
from ....core.messages import (
    AssistantMessage,
    AudioPart,
    CoreMessage,
    FilePart,
    ImagePart,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from ....core.model.llm import ModelConfig, output_token_limit
from ....errors.client import ClientError
from ....loggers import ScopedLogger
from ....types.completions import ChatCompletionChunk, ChatCompletionResult, Usage
from ....types.run_context import RunContext
from ....types.stacks import PromptCtx
from ..ollama._schema import clean_json_schema
from ._model import OpenAIChatCompletionClientConfig

if t.TYPE_CHECKING:
    from ....base.tools import CoreTool

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["OpenAIChatCompletionClient"])


class OpenAIChatCompletionClient(
    Component[OpenAIChatCompletionClientConfig],
    CoreChatCompletionClient,
):
    """OpenAI implementation of ``CoreChatCompletionClient``.

    Uses the official async OpenAI Python SDK and the Chat Completions
    endpoint. The OpenAI docs describe Chat Completions as accepting a
    conversation ``messages`` list plus optional ``tools`` for function
    calling; this adapter maps MaxAI's provider-neutral messages and
    tools onto that shape.
    """

    component_schema = OpenAIChatCompletionClientConfig
    component_type = "client"
    component_provider_override = "maxai.llm.OpenAIChatCompletionClient"

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    PROVIDER_NAME = "OpenAI"
    API_KEY_ENV = "OPENAI_API_KEY"
    SYSTEM_LAYER_SEPARATOR = "\n\n"
    SYSTEM_SOURCE = "system"

    def __init__(
        self,
        model: str,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        config: ModelConfig | None = None,
        max_tokens: int | None = None,
        api_key_env: str | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialize ``OpenAIChatCompletionClient``.

Parameters
----------
model : str
    Value supplied for ``model``.
api_key : str | SecretStr | None
    Value supplied for ``api_key``.
base_url : str | None
    Value supplied for ``base_url``.
organization : str | None
    Value supplied for ``organization``.
project : str | None
    Value supplied for ``project``.
config : ModelConfig | None
    Value supplied for ``config``.
max_tokens : int | None
    Value supplied for ``max_tokens``.
api_key_env : str | None
    Value supplied for ``api_key_env``.
kwargs : t.Any
    Value supplied for ``kwargs``."""
        super().__init__(
            model=model, api_key=api_key, config=config, api_key_env=api_key_env, **kwargs,
        )
        if self.api_key is None:
            raise ValueError(
                f"{self.PROVIDER_NAME} needs an API key: pass api_key or set "
                f"${self.api_key_env}."
            )
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.generation_options: dict[str, t.Any] = dict(kwargs)
        output_limit = output_token_limit(max_tokens or self.config.max_output_tokens)
        if output_limit is not None:
            self.generation_options["max_tokens"] = output_limit

        client_kwargs: dict[str, t.Any] = {}
        if self.api_key is not None:
            client_kwargs["api_key"] = self.api_key.get_secret_value()
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if organization is not None:
            client_kwargs["organization"] = organization
        if project is not None:
            client_kwargs["project"] = project
        self.client = AsyncOpenAI(**client_kwargs)

    def _to_config(self) -> OpenAIChatCompletionClientConfig:
        """Build the serializable configuration for ``OpenAIChatCompletionClient``."""
        return OpenAIChatCompletionClientConfig(
            model=self.model,
            api_key_env=self.api_key_env,
            base_url=self.base_url,
            organization=self.organization,
            project=self.project,
            options=self.generation_options,
            config=self.config.model_dump(exclude_none=True),
        )

    @classmethod
    def _from_config(
        cls,
        config: OpenAIChatCompletionClientConfig,
    ) -> "OpenAIChatCompletionClient":
        """Create an instance from its configuration for ``OpenAIChatCompletionClient``.

Parameters
----------
config : OpenAIChatCompletionClientConfig
    Value supplied for ``config``."""
        model_config = ModelConfig(**config.config) if config.config else None
        return cls(
            model=config.model,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            organization=config.organization,
            project=config.project,
            config=model_config,
            **config.options,
        )

    def _extract_thinking(self, message_or_delta: t.Any) -> str | None:
        """Reasoning text on a message/delta. Chat Completions has none;
        OpenAI-compatible gateways that expose it override this."""
        return None

    def build_tool_schema(self, tools: list["CoreTool"]) -> list[dict[str, t.Any]]:
        """Build tool schema for ``OpenAIChatCompletionClient``.

Parameters
----------
tools : list['CoreTool']
    Value supplied for ``tools``."""
        if not tools:
            return []
        return [self._tool_to_openai_schema(tool) for tool in tools]

    def _tool_to_openai_schema(self, tool: "CoreTool") -> dict[str, t.Any]:
        """Perform the internal ``tool to openai schema`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
tool : 'CoreTool'
    Value supplied for ``tool``."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": clean_json_schema(tool.parameters),
            },
        }

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        """Normalize usage stats for ``OpenAIChatCompletionClient``.

Parameters
----------
usage : t.Any
    Value supplied for ``usage``."""
        if usage is None:
            return Usage()
        prompt_tokens = int(self._get(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(self._get(usage, "completion_tokens", 0) or 0)
        cached = 0
        prompt_details = self._get(usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached = int(self._get(prompt_details, "cached_tokens", 0) or 0)
        return Usage(
            tokens_input=prompt_tokens,
            tokens_output=completion_tokens,
            tokens_cached=cached,
        )

    def format_messages(self, ctx: RunContext, prompts: PromptCtx) -> list[CoreMessage]:
        """Format messages for ``OpenAIChatCompletionClient``.

Parameters
----------
ctx : RunContext
    Value supplied for ``ctx``.
prompts : PromptCtx
    Value supplied for ``prompts``."""
        messages: list[CoreMessage] = []
        system_content = self._build_system_content(prompts)
        if system_content:
            messages.append(SystemMessage(source=self.SYSTEM_SOURCE, content=system_content))
        messages.extend(ctx.message_history.iter_messages())
        messages.extend(ctx.messages)
        return messages

    def _build_system_content(self, prompts: PromptCtx) -> str:
        """Perform the internal ``build system content`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
prompts : PromptCtx
    Value supplied for ``prompts``."""
        chunks: list[str] = []
        for rendered in prompts.rendered_layers.values():
            if rendered and rendered.strip():
                chunks.append(rendered.strip())
        return self.SYSTEM_LAYER_SEPARATOR.join(chunks)

    def build_api_messages(self, messages: list[CoreMessage]) -> list[dict[str, t.Any]]:
        """Build api messages for ``OpenAIChatCompletionClient``.

Parameters
----------
messages : list[CoreMessage]
    Value supplied for ``messages``."""
        return [self._message_to_dict(message) for message in messages]

    def _message_to_dict(self, msg: CoreMessage) -> dict[str, t.Any]:
        """Perform the internal ``message to dict`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
msg : CoreMessage
    Value supplied for ``msg``."""
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.text()}
        if isinstance(msg, UserMessage):
            return self._user_to_dict(msg)
        if isinstance(msg, AssistantMessage):
            return self._assistant_to_dict(msg)
        if isinstance(msg, ToolMessage):
            return self._tool_to_dict(msg)
        raise ClientError.unsupported_feature(
            feature=f"message type {type(msg).__name__}",
            model=self.model,
        )

    def _user_to_dict(self, msg: UserMessage) -> dict[str, t.Any]:
        """Perform the internal ``user to dict`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
msg : UserMessage
    Value supplied for ``msg``."""
        if isinstance(msg.content, str):
            return {"role": "user", "content": msg.content}

        content: list[dict[str, t.Any]] = []
        for part in msg.content:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                if not self.config.supports_vision:
                    raise ClientError.unsupported_feature("vision", self.model)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._image_url(part)},
                })
            elif isinstance(part, AudioPart):
                raise ClientError.unsupported_feature("audio input", self.model)
            elif isinstance(part, FilePart):
                raise ClientError.unsupported_feature("file input", self.model)
        return {"role": "user", "content": content}

    def _assistant_to_dict(self, msg: AssistantMessage) -> dict[str, t.Any]:
        """Perform the internal ``assistant to dict`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
msg : AssistantMessage
    Value supplied for ``msg``."""
        out: dict[str, t.Any] = {"role": "assistant", "content": msg.text()}
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.parameters),
                    },
                }
                for tc in msg.tool_calls
            ]
        return out

    @staticmethod
    def _tool_to_dict(msg: ToolMessage) -> dict[str, t.Any]:
        """Perform the internal ``tool to dict`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
msg : ToolMessage
    Value supplied for ``msg``."""
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": msg.text(),
        }

    @staticmethod
    def _image_url(part: ImagePart) -> str:
        """Perform the internal ``image url`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
part : ImagePart
    Value supplied for ``part``."""
        if part.url is not None:
            return part.url
        assert part.data is not None
        encoded = base64.b64encode(part.data).decode("utf-8")
        return f"data:{part.mime_type};base64,{encoded}"

    async def complete(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None,
        output_format: t.Type[BaseModel] | None,
        **kwargs: t.Any,
    ) -> ChatCompletionResult:
        """Send a completion request for ``OpenAIChatCompletionClient``.

Parameters
----------
messages : list[dict[str, t.Any]]
    Value supplied for ``messages``.
tools : list[dict[str, t.Any]] | None
    Value supplied for ``tools``.
output_format : t.Type[BaseModel] | None
    Value supplied for ``output_format``.
kwargs : t.Any
    Value supplied for ``kwargs``."""
        try:
            start = time.monotonic()
            request = self._build_request(
                messages=messages,
                tools=tools,
                output_format=output_format,
                stream=False,
                **kwargs,
            )
            response = await self.client.chat.completions.create(**request)
            duration_ms = int((time.monotonic() - start) * 1000)
            return self._parse_response(response, output_format, duration_ms)
        except Exception as exc:  # noqa: BLE001
            raise self._map_openai_error(exc) from exc

    async def stream(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None,
        output_format: t.Type[BaseModel] | None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[ChatCompletionChunk, None]:
        """Stream completion output for ``OpenAIChatCompletionClient``.

Parameters
----------
messages : list[dict[str, t.Any]]
    Value supplied for ``messages``.
tools : list[dict[str, t.Any]] | None
    Value supplied for ``tools``.
output_format : t.Type[BaseModel] | None
    Value supplied for ``output_format``.
kwargs : t.Any
    Value supplied for ``kwargs``."""
        try:
            start = time.monotonic()
            request = self._build_request(
                messages=messages,
                tools=tools,
                output_format=output_format,
                stream=True,
                **kwargs,
            )
            raw_stream = await self.client.chat.completions.create(**request)
            async for chunk in self._iter_chunks(raw_stream, output_format, start):
                yield chunk
        except Exception as exc:  # noqa: BLE001
            raise self._map_openai_error(exc) from exc

    def _build_request(
        self,
        *,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None,
        output_format: t.Type[BaseModel] | None,
        stream: bool,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        """Perform the internal ``build request`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
messages : list[dict[str, t.Any]]
    Value supplied for ``messages``.
tools : list[dict[str, t.Any]] | None
    Value supplied for ``tools``.
output_format : t.Type[BaseModel] | None
    Value supplied for ``output_format``.
stream : bool
    Value supplied for ``stream``.
kwargs : t.Any
    Value supplied for ``kwargs``."""
        request: dict[str, t.Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = kwargs.pop("tool_choice", "auto")
        options = {**self.generation_options, **kwargs}
        max_tokens = options.pop("max_tokens", None)
        if max_tokens is not None:
            request["max_completion_tokens"] = max_tokens
        if output_format is not None:
            request["response_format"] = self._response_format(output_format)
        if stream:
            stream_options = options.pop("stream_options", {}) or {}
            stream_options.setdefault("include_usage", True)
            request["stream_options"] = stream_options
        request.update(options)
        return request

    def _response_format(self, output_format: t.Type[BaseModel]) -> dict[str, t.Any]:
        """Perform the internal ``response format`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
output_format : t.Type[BaseModel]
    Value supplied for ``output_format``."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema_name(output_format),
                "schema": self._strict_json_schema(output_format.model_json_schema()),
                "strict": True,
            },
        }

    def _strict_json_schema(self, schema: dict[str, t.Any]) -> dict[str, t.Any]:
        """Perform the internal ``strict json schema`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
schema : dict[str, t.Any]
    Value supplied for ``schema``."""
        def visit(node: t.Any) -> t.Any:
            """Perform the ``visit`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
node : t.Any
    Value supplied for ``node``."""
            if isinstance(node, list):
                return [visit(item) for item in node]
            if not isinstance(node, dict):
                return node

            cleaned = {key: visit(value) for key, value in node.items() if key != "default"}
            properties = cleaned.get("properties")
            if cleaned.get("type") == "object" or isinstance(properties, dict):
                cleaned["additionalProperties"] = False
                if isinstance(properties, dict):
                    cleaned["required"] = list(properties.keys())
            return cleaned

        return t.cast(dict[str, t.Any], visit(schema))

    @staticmethod
    def _schema_name(output_format: t.Type[BaseModel]) -> str:
        """Perform the internal ``schema name`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
output_format : t.Type[BaseModel]
    Value supplied for ``output_format``."""
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", output_format.__name__)
        return name[:64] or "structured_output"

    def _parse_response(
        self,
        response: t.Any,
        output_format: t.Type[BaseModel] | None,
        duration_ms: int,
    ) -> ChatCompletionResult:
        """Perform the internal ``parse response`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
response : t.Any
    Value supplied for ``response``.
output_format : t.Type[BaseModel] | None
    Value supplied for ``output_format``.
duration_ms : int
    Value supplied for ``duration_ms``."""
        choice = self._first_choice(response)
        message = self._get(choice, "message")
        content = self._get(message, "content", "") or ""
        tool_calls = self._parse_tool_calls(self._get(message, "tool_calls", None) or [])
        structured_output = self._parse_structured_output(content, output_format)
        usage = self.normalize_usage_stats(self._get(response, "usage", None))
        usage.duration_ms = duration_ms
        usage.llm_calls = 1
        usage.tool_calls = len(tool_calls)
        return ChatCompletionResult(
            message=AssistantMessage(
                source="llm",
                content=content,
                thinking=self._extract_thinking(message),
                tool_calls=tool_calls,
                structured_output=structured_output,
            ),
            usage=usage,
            model=self._get(response, "model", self.model) or self.model,
            finish_reason=self._get(choice, "finish_reason", "stop") or "stop",
        )

    async def _iter_chunks(
        self,
        raw_stream: t.AsyncIterator[t.Any],
        output_format: t.Type[BaseModel] | None,
        start_time: float,
    ) -> t.AsyncGenerator[ChatCompletionChunk, None]:
        """Perform the internal ``iter chunks`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
raw_stream : t.AsyncIterator[t.Any]
    Value supplied for ``raw_stream``.
output_format : t.Type[BaseModel] | None
    Value supplied for ``output_format``.
start_time : float
    Value supplied for ``start_time``."""
        content_parts: list[str] = []
        tool_parts: dict[int, dict[str, t.Any]] = {}
        final_usage = Usage()
        finish_reason: str | None = None

        async for chunk in raw_stream:
            usage = self._get(chunk, "usage", None)
            if usage is not None:
                final_usage = self.normalize_usage_stats(usage)

            choices = self._get(chunk, "choices", []) or []
            if not choices:
                continue

            choice = choices[0]
            finish_reason = self._get(choice, "finish_reason", None) or finish_reason
            delta = self._get(choice, "delta", None)
            if delta is None:
                continue

            delta_thinking = self._extract_thinking(delta)
            if delta_thinking:
                yield ChatCompletionChunk(content="", thinking=delta_thinking, is_complete=False)

            delta_content = self._get(delta, "content", None)
            if delta_content:
                content_parts.append(delta_content)
                yield ChatCompletionChunk(content=delta_content, is_complete=False)

            for raw_tool in self._get(delta, "tool_calls", None) or []:
                index = int(self._get(raw_tool, "index", 0) or 0)
                state = tool_parts.setdefault(
                    index,
                    {"id": None, "name": None, "arguments": []},
                )
                tool_id = self._get(raw_tool, "id", None)
                if tool_id:
                    state["id"] = tool_id
                fn = self._get(raw_tool, "function", None)
                if fn is not None:
                    name = self._get(fn, "name", None)
                    if name:
                        state["name"] = name
                    arguments = self._get(fn, "arguments", None)
                    if arguments:
                        state["arguments"].append(arguments)

        for index in sorted(tool_parts):
            state = tool_parts[index]
            name = state.get("name")
            if not name:
                continue
            call_id = state.get("id") or f"openai_tool_call_{index}"
            arguments = "".join(state.get("arguments") or [])
            yield ChatCompletionChunk(
                content="",
                is_complete=False,
                tool_call_chunk={
                    "id": call_id,
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                },
            )

        structured_output = self._parse_structured_output(
            "".join(content_parts),
            output_format,
        )
        duration_ms = int((time.monotonic() - start_time) * 1000)
        final_usage.duration_ms = duration_ms
        final_usage.llm_calls = 1
        final_usage.tool_calls = len(tool_parts)
        yield ChatCompletionChunk(
            content="",
            is_complete=True,
            usage=final_usage,
            structured_output=structured_output,
            finish_reason=finish_reason,
        )

    def _parse_tool_calls(self, raw_tool_calls: t.Any) -> list[ToolCall]:
        """Perform the internal ``parse tool calls`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
raw_tool_calls : t.Any
    Value supplied for ``raw_tool_calls``."""
        parsed: list[ToolCall] = []
        for raw in raw_tool_calls or []:
            tool_id = self._get(raw, "id", None)
            fn = self._get(raw, "function", None)
            if fn is None:
                continue
            name = self._get(fn, "name", None)
            if not name:
                continue
            arguments = self._get(fn, "arguments", "") or ""
            try:
                params = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
            except json.JSONDecodeError:
                params = {"raw_error_content": arguments, "parsing_error": True}
            payload: dict[str, t.Any] = {
                "tool_name": name,
                "parameters": params,
            }
            if tool_id:
                payload["id"] = tool_id
            parsed.append(ToolCall(**payload))
        return parsed

    def _parse_structured_output(
        self,
        content: str,
        output_format: t.Type[BaseModel] | None,
    ) -> BaseModel | None:
        """Perform the internal ``parse structured output`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
content : str
    Value supplied for ``content``.
output_format : t.Type[BaseModel] | None
    Value supplied for ``output_format``."""
        if output_format is None or not content:
            return None
        try:
            return output_format.model_validate_json(content)
        except Exception:
            log.warning(
                "Failed to parse structured output",
                output_format_name=output_format.__name__,
            )
            return None

    @staticmethod
    def _first_choice(response: t.Any) -> t.Any:
        """Perform the internal ``first choice`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
response : t.Any
    Value supplied for ``response``."""
        choices = OpenAIChatCompletionClient._get(response, "choices", None) or []
        if not choices:
            raise ClientError.invalid_response("Response contained no choices.")
        return choices[0]

    @staticmethod
    def _get(obj: t.Any, key: str, default: t.Any = None) -> t.Any:
        """Perform the internal ``get`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
obj : t.Any
    Value supplied for ``obj``.
key : str
    Value supplied for ``key``.
default : t.Any
    Value supplied for ``default``."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _map_openai_error(cls, exc: Exception) -> ClientError:
        """Perform the internal ``map openai error`` operation for ``OpenAIChatCompletionClient``.

Parameters
----------
exc : Exception
    Value supplied for ``exc``."""
        if isinstance(exc, ClientError):
            return exc
        if isinstance(exc, AuthenticationError):
            return ClientError.authentication_failed(str(exc))
        if isinstance(exc, RateLimitError):
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    retry_after = float(response.headers.get("retry-after"))
                except Exception:
                    retry_after = None
            return ClientError.rate_limit_exceeded(retry_after)
        if isinstance(exc, APITimeoutError):
            return ClientError.request_timeout(0)
        if isinstance(exc, BadRequestError):
            return ClientError.invalid_request(str(exc))
        if isinstance(exc, APIError):
            return ClientError.api_error(
                provider=cls.PROVIDER_NAME,
                status=getattr(exc, "status_code", None),
                detail=str(exc),
            )
        return ClientError.unexpected(provider=cls.PROVIDER_NAME, detail=str(exc))
