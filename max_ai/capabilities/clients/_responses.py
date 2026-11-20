"""The Responses format: OpenAI's newer API, where a model can reason and call tools."""

from __future__ import annotations

import json
import re
import time
import typing as t

from pydantic import BaseModel

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
from ...errors.client import ClientError
from ...types.completions import LLMChunk, LLMResponse, Usage
from ...types.tools import CoreToolDefinition
from ._openai_sdk import OpenAISDKClient


class ResponsesClient(OpenAISDKClient[ConfigT]):
    """Messages, tools, the call and the answer in the Responses format. A
    subclass is one service: its address, its key, its config and how it is
    blocked.

    The conversation is sent whole in every call and nothing is stored on
    the server (``store=False``). The reasoning of the model comes back
    encrypted and is kept in ``AssistantMessage.reasoning_state``, to be
    sent again: without it the model loses its thread between tool calls."""

    FORMAT: t.ClassVar[str] = "responses"  # the key of its ``reasoning_state``
    ENCRYPTED: t.ClassVar[str] = "reasoning.encrypted_content"
    # Why an answer is not whole, in the words every client uses.
    INCOMPLETE: t.ClassVar[dict[str, str]] = {
        "max_output_tokens": "length", "content_filter": "content_filter",
    }

    # -------- MESSAGES ---------------------------------------------------------------
    def build_api_messages(self, messages: list[Message]) -> list[dict[str, t.Any]]:
        """The conversation as input items: one message may be several."""
        return [item for message in messages for item in self.items(message)]

    def items(self, message: Message) -> list[dict[str, t.Any]]:
        """One message as the items of the Responses format."""
        if isinstance(message, SystemMessage):
            return [{"role": "system", "content": message.text()}]
        if isinstance(message, UserMessage):
            return [{"role": "user", "content": self.user_content(message)}]
        if isinstance(message, ToolMessage):
            return [{
                "type": "function_call_output", "call_id": message.tool_call_id,
                "output": message.text(),
            }]

        # The assistant: its reasoning first, as it was given; then what it said and asked for.
        said: list[dict[str, t.Any]] = list(message.reasoning_state.get(self.FORMAT, []))
        if message.text():
            said.append({"role": "assistant", "content": message.text()})
        for call in message.tool_calls:
            said.append({
                "type": "function_call", "call_id": call.id, "name": call.tool_name,
                "arguments": json.dumps(call.parameters),
            })
        return said

    def user_content(self, message: UserMessage) -> str | list[dict[str, t.Any]]:
        """Plain text as it is; text and images as a list of parts."""
        if isinstance(message.content, str):
            return message.content
        parts: list[dict[str, t.Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append({"type": "input_text", "text": part.text})
            elif isinstance(part, ImagePart) and self.config.supports_vision:
                parts.append({"type": "input_image", "image_url": self.image_url(part)})
            else:
                kind = "vision" if isinstance(part, ImagePart) else type(part).__name__
                raise ClientError.unsupported_feature(kind, self.model)
        return parts

    # -------- TOOLS, USAGE -------------------------------------------------------------
    def build_tool_schema(self, tools: list[CoreToolDefinition]) -> list[dict[str, t.Any]]:
        return [
            {
                "type": "function", "name": tool.name, "description": tool.description,
                "parameters": tool.parameters, "strict": False,
            }
            for tool in tools
        ]

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        if usage is None:
            return Usage()
        details = self.get(usage, "input_tokens_details")
        return Usage(
            tokens_input=int(self.get(usage, "input_tokens") or 0),
            tokens_output=int(self.get(usage, "output_tokens") or 0),
            tokens_cached=int(self.get(details, "cached_tokens") or 0),
        )

    # -------- THE CALL ---------------------------------------------------------------
    async def complete(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]],
        output_format: type[BaseModel] | None,
    ) -> LLMResponse:
        start = time.monotonic()
        response = await self.call(self.request(messages, tools, output_format, stream=False))
        return self.read(response, output_format, start)

    async def complete_stream(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]],
        output_format: type[BaseModel] | None,
    ) -> t.AsyncGenerator[LLMChunk, None]:
        start = time.monotonic()
        request = self.request(messages, tools, output_format, stream=True)
        done: LLMResponse | None = None

        async for event in await self.call(request):
            kind = self.get(event, "type")
            # 1. Text and the summary of the reasoning go out as they arrive.
            if kind == "response.output_text.delta":
                yield LLMChunk(content=self.get(event, "delta") or "", is_complete=False)
            elif kind == "response.reasoning_summary_text.delta":
                yield LLMChunk(
                    content="", thinking=self.get(event, "delta") or "", is_complete=False
                )
            # 2. The last event carries the whole answer: it is read as one not streamed.
            elif kind in ("response.completed", "response.incomplete", "response.failed"):
                done = self.read(self.get(event, "response"), output_format, start)
            elif kind == "error":
                raise self.failed(self.get(event, "code"), self.get(event, "message"))
        if done is None:
            raise ClientError.stream_interrupted("The stream ended without its last event.")

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

    async def call(self, request: dict[str, t.Any]) -> t.Any:
        """The SDK call. Any: its typed overloads cannot be chosen from a dictionary."""
        create: t.Any = self.client.responses.create
        return await create(**request)

    def request(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]],
        output_format: type[BaseModel] | None,
        *,
        stream: bool,
    ) -> dict[str, t.Any]:
        """The body of the call. What ``body`` gives goes last, as it is, so
        it can replace anything and may hold fields the SDK does not know."""
        request: dict[str, t.Any] = {
            "model": self.model, "input": messages, "stream": stream,
            # Nothing is kept on the server, so the reasoning must come back with the answer.
            "store": False, "include": [self.ENCRYPTED],
        }
        if tools:
            request["tools"], request["tool_choice"] = tools, "auto"
        if limit := self.config.max_output_tokens:
            request["max_output_tokens"] = limit
        if output_format is not None:
            request["text"] = {"format": self.text_format(output_format)}
        if extra := self.body():
            request["extra_body"] = extra
        return request

    def body(self) -> dict[str, t.Any]:
        """Fields added to every request as they are: how the model thinks
        (``config.thinking``) and the ``options``."""
        return {**(self.config.thinking or {}), **self.options}

    # -------- READING THE ANSWER -------------------------------------------------------
    def read(
        self, response: t.Any, output_format: type[BaseModel] | None, start: float
    ) -> LLMResponse:
        """A whole answer of the Responses format as an ``LLMResponse``."""
        # 1. An answer that failed is an error, like one refused before it began.
        status = self.get(response, "status")
        if status == "failed":
            error = self.get(response, "error")
            raise self.failed(self.get(error, "code"), self.get(error, "message"))

        # 2. Its items, in order: reasoning, text, tool calls.
        text: list[str] = []
        thoughts: list[str] = []
        calls: list[ToolCall] = []
        state: list[dict[str, t.Any]] = []
        items: list[t.Any] = self.get(response, "output") or []
        for item in items:
            kind = self.get(item, "type")
            if kind == "reasoning":
                state.append(self.reasoning_item(item))
                summary: list[t.Any] = self.get(item, "summary") or []
                thoughts += [self.get(part, "text") or "" for part in summary]
            elif kind == "message":
                parts: list[t.Any] = self.get(item, "content") or []
                text += [self.get(part, "text") or self.get(part, "refusal") or "" for part in parts]
            elif kind == "function_call":
                calls.append(self.tool_call(item))

        # 3. Why it ended, in the words every client uses.
        finish = "tool_calls" if calls else "stop"
        if status == "incomplete":
            reason = self.get(self.get(response, "incomplete_details"), "reason")
            finish = self.INCOMPLETE.get(str(reason), "length")

        # 4. How much it cost.
        usage = self.normalize_usage_stats(self.get(response, "usage"))
        usage.duration_ms = int((time.monotonic() - start) * 1000)
        usage.llm_calls, usage.tool_calls = 1, len(calls)
        content = "".join(text)
        return LLMResponse(
            message=AssistantMessage(
                source="llm", content=content, thinking="\n\n".join(thoughts) or None,
                tool_calls=calls, structured_output=self.structured(content, output_format),
                reasoning_state={self.FORMAT: state} if state else {},
            ),
            usage=usage,
            model=self.get(response, "model") or self.model,
            finish_reason=finish,
        )

    def reasoning_item(self, item: t.Any) -> dict[str, t.Any]:
        """A reasoning item as it must be sent back: its id, its summary and
        its encrypted content, nothing else."""
        summary: list[t.Any] = self.get(item, "summary") or []
        return {
            "type": "reasoning", "id": self.get(item, "id"),
            "summary": [
                {"type": "summary_text", "text": self.get(part, "text") or ""} for part in summary
            ],
            "encrypted_content": self.get(item, "encrypted_content"),
        }

    def tool_call(self, item: t.Any) -> ToolCall:
        """A tool call of an answer. Arguments that are not JSON are kept
        with a mark, so the model is told and can fix them."""
        arguments: str = self.get(item, "arguments") or ""
        parameters: dict[str, t.Any]
        try:
            parameters = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            parameters = {"raw_error_content": arguments, "parsing_error": True}
        given: dict[str, str] = {"id": identifier} if (identifier := self.get(item, "call_id")) else {}
        return ToolCall(tool_name=self.get(item, "name"), parameters=parameters, **given)

    def failed(self, code: t.Any, message: t.Any) -> ClientError:
        """The error inside an answer or a stream, as one raised by the call."""
        if code == "rate_limit_exceeded":
            return ClientError.rate_limit_exceeded()
        return ClientError.api_error(
            provider=self.PROVIDER_NAME, status=None, detail=f"{code}: {message}"
        )

    @classmethod
    def text_format(cls, output_format: type[BaseModel]) -> dict[str, t.Any]:
        """Ask for JSON that follows ``output_format`` exactly."""
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", output_format.__name__)[:64] or "structured_output"
        schema = cls.strict(output_format.model_json_schema())
        return {"type": "json_schema", "name": name, "schema": schema, "strict": True}
