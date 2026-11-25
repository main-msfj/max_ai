"""The Messages format: Anthropic's API, shared by every service that serves Claude."""

from __future__ import annotations

import base64
import json
import time
import typing as t

from anthropic import (
    APIError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    OverloadedError,
    RateLimitError,
)
from pydantic import BaseModel

from ...base.clients import CoreLLMClient
from ...base.component import ConfigT
from ...core.messages import (
    AssistantMessage,
    ImagePart,
    Message,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from ...core.model.llm import ModelConfig
from ...core.retry import RetryPolicy
from ...errors.client import ClientError
from ...types.completions import LLMChunk, LLMResponse, Usage
from ...types.run_context import RunContext
from ...types.tools import CoreToolDefinition


class MessagesClient(CoreLLMClient[ConfigT]):
    """Messages, tools, the call and the answer in the Messages format. A
    subclass is one service: its address, its key and its config.

    Every call is streamed, also the one that returns the whole answer: the
    SDK refuses a long answer any other way. When the model thinks, the
    blocks of its answer are kept in ``AssistantMessage.reasoning_state``
    and sent back as they came: Claude asks for them unchanged."""

    FORMAT: t.ClassVar[str] = "messages"  # the key of its ``reasoning_state``
    DEFAULT_BASE_URL: t.ClassVar[str | None] = None
    BLOCKED: t.ClassVar[str] = "blocked by content filtering"  # in the error of a refused request
    # How a model that thinks is asked to: by itself, and showing a summary.
    THINKING: t.ClassVar[dict[str, t.Any]] = {
        "thinking": {"type": "adaptive", "display": "summarized"},
    }
    # Why an answer ended, in the words every client uses.
    FINISH: t.ClassVar[dict[str, str]] = {
        "end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length",
        "refusal": "content_filter",
    }

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        config: ModelConfig | None = None,
        *,
        options: dict[str, t.Any] | None = None,  # speed, metadata...: sent as they are
        retry: RetryPolicy | None = None,
        api_key_env: str | None = None,
    ) -> None:
        super().__init__(model, api_key, config, retry=retry, api_key_env=api_key_env)
        if self.api_key is None:
            raise ValueError(
                f"{self.PROVIDER_NAME} needs an API key: pass api_key or set ${self.api_key_env}."
            )
        self.base_url = base_url
        self.options = dict(options or {})
        self.client = self.connect()

    def connect(self) -> AsyncAnthropic:
        """The SDK client. It never retries: the base class does, with its policy."""
        return AsyncAnthropic(
            api_key=self.api_key, base_url=self.base_url or self.DEFAULT_BASE_URL, max_retries=0,
        )

    async def aclose(self) -> None:
        await self.client.close()
        self.client = self.connect()

    def format_messages(self, ctx: RunContext, prompts: str) -> list[Message]:
        """The system prompt first, then the conversation."""
        system = [SystemMessage(source="system", content=prompts)] if prompts.strip() else []
        return [*system, *ctx.messages]

    def edited(self, messages: list[Message]) -> list[Message]:
        """Claude signs each thinking block against the conversation before
        it: once that changed, the API refuses them. The blocks leave; the
        text and the tool calls stay, and so does the thinking to show."""
        kept: list[Message] = []
        for message in messages:
            # Only an answer that still holds its blocks is touched.
            if isinstance(message, AssistantMessage) and self.FORMAT in message.reasoning_state:
                state = {k: v for k, v in message.reasoning_state.items() if k != self.FORMAT}
                message = message.model_copy(update={"reasoning_state": state})
            kept.append(message)
        return kept

    # -------- MESSAGES ---------------------------------------------------------------
    def build_api_messages(self, messages: list[Message]) -> list[dict[str, t.Any]]:
        """The conversation in the Messages format. A system message keeps
        its role here; ``request`` takes it out to its own field."""
        built: list[dict[str, t.Any]] = []
        for message in messages:
            # 1. The results of one round go together, in one user message.
            if isinstance(message, ToolMessage):
                result = self.tool_result(message)
                if built and built[-1].get("results"):
                    built[-1]["content"].append(result)
                else:
                    built.append({"role": "user", "content": [result], "results": True})

            # 2. The rest, one by one; an assistant that said nothing is not sent.
            elif entry := self.message(message):
                built.append(entry)
        return built

    def message(self, message: Message) -> dict[str, t.Any] | None:
        """One message that is not a tool result, or None when it is empty."""
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.text()}
        if isinstance(message, UserMessage):
            return {"role": "user", "content": self.user_content(message)}
        if isinstance(message, AssistantMessage):
            return self.assistant(message)
        raise ClientError.unsupported_feature(f"message type {type(message).__name__}", self.model)

    def assistant(self, message: AssistantMessage) -> dict[str, t.Any] | None:
        """What the model said, as blocks."""
        # 1. With thinking, its blocks go back exactly as they came, in their order.
        if kept := message.reasoning_state.get(self.FORMAT):
            return {"role": "assistant", "content": kept}

        # 2. Without it, they are built: the text, then each tool call.
        blocks: list[dict[str, t.Any]] = []
        if message.text():
            blocks.append({"type": "text", "text": message.text()})
        blocks += [
            {"type": "tool_use", "id": call.id, "name": call.tool_name, "input": call.parameters}
            for call in message.tool_calls
        ]
        return {"role": "assistant", "content": blocks} if blocks else None

    def user_content(self, message: UserMessage) -> str | list[dict[str, t.Any]]:
        """Plain text as it is; text and images as a list of blocks."""
        if isinstance(message.content, str):
            return message.content
        parts: list[dict[str, t.Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart) and self.config.supports_vision:
                parts.append({"type": "image", "source": self.image_source(part)})
            else:
                kind = "vision" if isinstance(part, ImagePart) else type(part).__name__
                raise ClientError.unsupported_feature(kind, self.model)
        return parts

    @staticmethod
    def tool_result(message: ToolMessage) -> dict[str, t.Any]:
        """A tool's answer as a block; a failed one says so."""
        result: dict[str, t.Any] = {
            "type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.text(),
        }
        if not message.success:
            result["is_error"] = True
        return result

    @staticmethod
    def image_source(part: ImagePart) -> dict[str, t.Any]:
        """The image's address, or the image itself in base64."""
        if part.url is not None:
            return {"type": "url", "url": part.url}
        assert part.data is not None
        data = base64.b64encode(part.data).decode()
        return {"type": "base64", "media_type": part.mime_type, "data": data}

    # -------- TOOLS, USAGE -------------------------------------------------------------
    def build_tool_schema(self, tools: list[CoreToolDefinition]) -> list[dict[str, t.Any]]:
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
            for tool in tools
        ]

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        """``input_tokens`` leaves out what the cache read or wrote: here
        the input is all of it, as in the other clients."""
        if usage is None:
            return Usage()
        read = int(self.get(usage, "cache_read_input_tokens") or 0)
        written = int(self.get(usage, "cache_creation_input_tokens") or 0)
        return Usage(
            tokens_input=int(self.get(usage, "input_tokens") or 0) + read + written,
            tokens_output=int(self.get(usage, "output_tokens") or 0),
            tokens_cached=read,
        )

    # -------- THE CALL ---------------------------------------------------------------
    async def complete(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]],
        output_format: type[BaseModel] | None,
    ) -> LLMResponse:
        start = time.monotonic()
        async with self.call(self.request(messages, tools, output_format)) as stream:
            return self.read(await stream.get_final_message(), output_format, start)

    async def complete_stream(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]],
        output_format: type[BaseModel] | None,
    ) -> t.AsyncGenerator[LLMChunk, None]:
        start = time.monotonic()
        async with self.call(self.request(messages, tools, output_format)) as stream:
            # 1. Text and thinking go out as they arrive.
            async for event in stream:
                if self.get(event, "type") != "content_block_delta":
                    continue
                delta = self.get(event, "delta")
                if piece := self.get(delta, "text"):
                    yield LLMChunk(content=piece, is_complete=False)
                elif thought := self.get(delta, "thinking"):
                    yield LLMChunk(content="", thinking=thought, is_complete=False)

            # 2. The whole answer is read as one not streamed.
            done = self.read(await stream.get_final_message(), output_format, start)

        # 3. Each tool call, then the closing piece with the usage.
        for call in done.message.tool_calls:
            yield LLMChunk(content="", is_complete=False, tool_call_chunk={
                "id": call.id,
                "function": {"name": call.tool_name, "arguments": json.dumps(call.parameters)},
            })
        yield LLMChunk(
            content="", is_complete=True, usage=done.usage, finish_reason=done.finish_reason,
            structured_output=done.message.structured_output, message=done.message,
        )

    def call(self, request: dict[str, t.Any]) -> t.Any:
        """The SDK's stream, to enter with ``async with``. Any: its typed
        parameters cannot be chosen from a dictionary."""
        stream: t.Any = self.client.messages.stream
        return stream(**request)

    def request(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]],
        output_format: type[BaseModel] | None,
    ) -> dict[str, t.Any]:
        """The body of the call. What ``body`` gives goes as it is and may
        hold fields the SDK does not know."""
        # 1. The system prompt goes apart; the mark of the tool results is ours.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
        ]
        request: dict[str, t.Any] = {
            "model": self.model, "messages": turns, "max_tokens": self.config.max_output_tokens,
            "cache_control": {"type": "ephemeral"},  # the provider caches up to the last block
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = tools

        # 2. Thinking and options as they are; the answer's format joins ``output_config``.
        extra = self.body()
        if output_format is not None:
            schema = self.strict(output_format.model_json_schema())
            extra["output_config"] = {
                **extra.get("output_config", {}), "format": {"type": "json_schema", "schema": schema},
            }
        if extra:
            request["extra_body"] = extra
        return request

    def body(self) -> dict[str, t.Any]:
        """Fields added to every request as they are: how the model thinks
        (``config.thinking``, or ``THINKING`` for a model that thinks and
        was not told how) and the ``options``."""
        thinking = self.config.thinking
        if thinking is None and self.config.supports_thinking:
            thinking = self.THINKING
        return {**(thinking or {}), **self.options}

    # -------- READING THE ANSWER -------------------------------------------------------
    def read(
        self, response: t.Any, output_format: type[BaseModel] | None, start: float
    ) -> LLMResponse:
        """A whole answer of the Messages format as an ``LLMResponse``."""
        # 1. Its blocks, in order: thinking, text, tool calls.
        text: list[str] = []
        thoughts: list[str] = []
        calls: list[ToolCall] = []
        thought = False
        raw: list[t.Any] = self.get(response, "content") or []
        blocks = [self.plain(block) for block in raw]
        for block in blocks:
            kind = block.get("type")
            if kind in ("thinking", "redacted_thinking"):
                thought = True
                if block.get("thinking"):
                    thoughts.append(block["thinking"])
            elif kind == "text":
                text.append(block.get("text") or "")
            elif kind == "tool_use":
                calls.append(ToolCall(
                    id=block["id"], tool_name=block["name"], parameters=block.get("input") or {},
                ))

        # 2. How much it cost.
        usage = self.normalize_usage_stats(self.get(response, "usage"))
        usage.duration_ms = int((time.monotonic() - start) * 1000)
        usage.llm_calls, usage.tool_calls = 1, len(calls)
        content = "".join(text)
        return LLMResponse(
            message=AssistantMessage(
                source="llm", content=content, thinking="\n\n".join(thoughts) or None,
                tool_calls=calls, structured_output=self.structured(content, output_format),
                # Only an answer with thinking must go back block by block.
                reasoning_state={self.FORMAT: blocks} if thought else {},
            ),
            usage=usage,
            model=self.get(response, "model") or self.model,
            finish_reason=self.FINISH.get(str(self.get(response, "stop_reason")), "stop"),
        )

    @staticmethod
    def plain(block: t.Any) -> dict[str, t.Any]:
        """A block of the answer as a dictionary, with only what it was given."""
        if isinstance(block, dict):
            return t.cast("dict[str, t.Any]", block)
        return block.to_dict()

    # -------- ERRORS ---------------------------------------------------------------
    def blocked(self, error: Exception) -> list[str] | None:
        """The request was refused for its content (HTTP 400). It never says
        what it found. Not seen live. A refusal of the model itself is not
        an error: it is an answer that ends with ``content_filter``."""
        refused = getattr(error, "status_code", None) == 400 and self.BLOCKED in str(error).lower()
        return [] if refused else None

    def window_full(self, error: Exception) -> bool:
        return isinstance(error, BadRequestError) and "prompt is too long" in str(error).lower()

    def rate_limited(self, error: Exception) -> bool:
        return isinstance(error, (RateLimitError, OverloadedError))

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
