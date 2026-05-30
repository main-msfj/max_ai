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

from ..ollama._schema import clean_json_schema
from ...base.clients import CoreChatCompletionClient
from ...base.component import Component
from ...core.messages import (
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
from ...core.models import ModelConfig, OpenAIChatCompletionClientConfig
from ...errors.client import ClientError
from ...loggers import ScopedLogger
from ...types.completions import ChatCompletionChunk, ChatCompletionResult, Usage
from ...types.run_context import RunContext
from ...types.stacks import PromptCtx

if t.TYPE_CHECKING:
    from ...base.tools import CoreTool

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
        **kwargs: t.Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, config=config, **kwargs)
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.generation_options: dict[str, t.Any] = dict(kwargs)
        output_limit = max_tokens
        if output_limit is None and self.config.max_output_tokens:
            output_limit = self.config.max_output_tokens
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
        return OpenAIChatCompletionClientConfig(
            model=self.model,
            api_key=self.api_key,
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
        model_config = ModelConfig(**config.config) if config.config else None
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            organization=config.organization,
            project=config.project,
            config=model_config,
            **config.options,
        )

    def build_tool_schema(self, tools: list["CoreTool"]) -> list[dict[str, t.Any]]:
        if not tools:
            return []
        return [self._tool_to_openai_schema(tool) for tool in tools]

    def _tool_to_openai_schema(self, tool: "CoreTool") -> dict[str, t.Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": clean_json_schema(tool.parameters),
            },
        }

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
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
        messages: list[CoreMessage] = []
        system_content = self._build_system_content(prompts)
        if system_content:
            messages.append(SystemMessage(source=self.SYSTEM_SOURCE, content=system_content))
        messages.extend(ctx.message_history.iter_messages())
        messages.extend(ctx.messages)
        return messages

    def _build_system_content(self, prompts: PromptCtx) -> str:
        chunks: list[str] = []
        for rendered in prompts.rendered_layers.values():
            if rendered and rendered.strip():
                chunks.append(rendered.strip())
        return self.SYSTEM_LAYER_SEPARATOR.join(chunks)

    def build_api_messages(self, messages: list[CoreMessage]) -> list[dict[str, t.Any]]:
        return [self._message_to_dict(message) for message in messages]

    def _message_to_dict(self, msg: CoreMessage) -> dict[str, t.Any]:
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
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": msg.text(),
        }

    @staticmethod
    def _image_url(part: ImagePart) -> str:
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
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema_name(output_format),
                "schema": self._strict_json_schema(output_format.model_json_schema()),
                "strict": True,
            },
        }

    def _strict_json_schema(self, schema: dict[str, t.Any]) -> dict[str, t.Any]:
        def visit(node: t.Any) -> t.Any:
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
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", output_format.__name__)
        return name[:64] or "structured_output"

    def _parse_response(
        self,
        response: t.Any,
        output_format: t.Type[BaseModel] | None,
        duration_ms: int,
    ) -> ChatCompletionResult:
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
        content_parts: list[str] = []
        tool_parts: dict[int, dict[str, t.Any]] = {}
        final_usage = Usage()

        async for chunk in raw_stream:
            usage = self._get(chunk, "usage", None)
            if usage is not None:
                final_usage = self.normalize_usage_stats(usage)

            choices = self._get(chunk, "choices", []) or []
            if not choices:
                continue

            choice = choices[0]
            delta = self._get(choice, "delta", None)
            if delta is None:
                continue

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
        )

    def _parse_tool_calls(self, raw_tool_calls: t.Any) -> list[ToolCall]:
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
        choices = OpenAIChatCompletionClient._get(response, "choices", None) or []
        if not choices:
            raise ClientError.invalid_response("OpenAI response contained no choices.")
        return choices[0]

    @staticmethod
    def _get(obj: t.Any, key: str, default: t.Any = None) -> t.Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _map_openai_error(exc: Exception) -> ClientError:
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
                provider="OpenAI",
                status=getattr(exc, "status_code", None),
                detail=str(exc),
            )
        return ClientError.unexpected(provider="OpenAI", detail=str(exc))
