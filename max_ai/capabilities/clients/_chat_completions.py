"""The Chat Completions format, shared by the clients of every service that speaks it."""

from __future__ import annotations

import json
import logging
import re
import time
import typing as t

from pydantic import BaseModel

from ...base.component import ConfigT
from ...core.ids import short_id
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

logger = logging.getLogger(__name__)


class ChatCompletionsClient(OpenAISDKClient[ConfigT]):
    """Messages, tools, the call and the answer in the Chat Completions
    format. A subclass is one service: its address, its key, its config and
    what it does differently (``body``, ``thinking``, ``check``)."""

    # -------- MESSAGES ---------------------------------------------------------------
    def build_api_messages(self, messages: list[Message]) -> list[dict[str, t.Any]]:
        return [self.message(message) for message in messages]

    def message(self, message: Message) -> dict[str, t.Any]:
        """One message in OpenAI's format."""
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.text()}
        if isinstance(message, UserMessage):
            return {"role": "user", "content": self.user_content(message)}
        if isinstance(message, ToolMessage):
            return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.text()}
        if isinstance(message, AssistantMessage):
            said: dict[str, t.Any] = {"role": "assistant", "content": message.text()}
            if message.tool_calls:
                said["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.tool_name, "arguments": json.dumps(call.parameters)},
                    }
                    for call in message.tool_calls
                ]
            return said
        raise ClientError.unsupported_feature(f"message type {type(message).__name__}", self.model)

    def user_content(self, message: UserMessage) -> str | list[dict[str, t.Any]]:
        """Plain text as it is; text and images as a list of parts."""
        if isinstance(message.content, str):
            return message.content
        parts: list[dict[str, t.Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart) and self.config.supports_vision:
                parts.append({"type": "image_url", "image_url": {"url": self.image_url(part)}})
            else:
                kind = "vision" if isinstance(part, ImagePart) else type(part).__name__
                raise ClientError.unsupported_feature(kind, self.model)
        return parts

    # -------- TOOLS, USAGE, ERRORS -----------------------------------------------------
    def build_tool_schema(self, tools: list[CoreToolDefinition]) -> list[dict[str, t.Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        if usage is None:
            return Usage()
        details = self.get(usage, "prompt_tokens_details")
        return Usage(
            tokens_input=int(self.get(usage, "prompt_tokens") or 0),
            tokens_output=int(self.get(usage, "completion_tokens") or 0),
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
        request = self.request(messages, tools, output_format, stream=False)
        response = await self.call(request)
        self.check(response)

        # 1. The first choice is the answer.
        choices: list[t.Any] = self.get(response, "choices") or []
        if not choices:
            raise ClientError.invalid_response("Response contained no choices.")
        message = self.get(choices[0], "message")
        content = self.get(message, "content") or ""
        calls = self.tool_calls(self.get(message, "tool_calls") or [])

        # 2. How much it cost.
        usage = self.normalize_usage_stats(self.get(response, "usage"))
        usage.duration_ms = int((time.monotonic() - start) * 1000)
        usage.llm_calls, usage.tool_calls = 1, len(calls)
        return LLMResponse(
            message=AssistantMessage(
                source="llm", content=content, thinking=self.thinking(message), tool_calls=calls,
                structured_output=self.structured(content, output_format),
            ),
            usage=usage,
            model=self.get(response, "model") or self.model,
            finish_reason=self.get(choices[0], "finish_reason") or "stop",
        )

    async def complete_stream(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]],
        output_format: type[BaseModel] | None,
    ) -> t.AsyncGenerator[LLMChunk, None]:
        start = time.monotonic()
        request = self.request(messages, tools, output_format, stream=True)
        text: list[str] = []
        thoughts: list[str] = []
        calls: dict[int, dict[str, t.Any]] = {}  # a tool call arrives in pieces, by its index
        usage, finish = Usage(), None

        async for chunk in await self.call(request):
            self.check(chunk)
            # 1. The usage comes alone, in the last piece.
            if (counted := self.get(chunk, "usage")) is not None:
                usage = self.normalize_usage_stats(counted)
            choices: list[t.Any] = self.get(chunk, "choices") or []
            if not choices:
                continue
            finish = self.get(choices[0], "finish_reason") or finish
            delta = self.get(choices[0], "delta")
            if delta is None:
                continue

            # 2. Thinking and text go out as they arrive.
            if thought := self.thinking(delta):
                thoughts.append(thought)
                yield LLMChunk(content="", thinking=thought, is_complete=False)
            if piece := self.get(delta, "content"):
                text.append(piece)
                yield LLMChunk(content=piece, is_complete=False)

            # 3. Tool calls are put together until the end.
            raw_calls: list[t.Any] = self.get(delta, "tool_calls") or []
            for raw in raw_calls:
                index = int(self.get(raw, "index") or 0)
                call = calls.setdefault(index, {"id": None, "name": None, "arguments": ""})
                call["id"] = self.get(raw, "id") or call["id"]
                function = self.get(raw, "function")
                call["name"] = self.get(function, "name") or call["name"]
                call["arguments"] += self.get(function, "arguments") or ""

        # 4. Each whole tool call, then the closing piece with the usage.
        whole: list[dict[str, t.Any]] = [
            {
                "id": call["id"] or short_id(),
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for _, call in sorted(calls.items())
            if call["name"]
        ]
        for call in whole:
            yield LLMChunk(content="", is_complete=False, tool_call_chunk=call)
        usage.duration_ms = int((time.monotonic() - start) * 1000)
        usage.llm_calls, usage.tool_calls = 1, len(whole)
        content = "".join(text)
        structured = self.structured(content, output_format)
        yield LLMChunk(
            content="", is_complete=True, usage=usage, finish_reason=finish,
            structured_output=structured,
            message=AssistantMessage(
                source="llm", content=content, thinking="".join(thoughts) or None,
                tool_calls=self.tool_calls(whole), structured_output=structured,
            ),
        )

    async def call(self, request: dict[str, t.Any]) -> t.Any:
        """The SDK call. Any: its typed overloads cannot be chosen from a dictionary."""
        create: t.Any = self.client.chat.completions.create
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
        request: dict[str, t.Any] = {"model": self.model, "messages": messages, "stream": stream}
        if tools:
            request["tools"], request["tool_choice"] = tools, "auto"
        if limit := self.config.max_output_tokens:
            request["max_completion_tokens"] = limit
        if output_format is not None:
            request["response_format"] = self.response_format(output_format)
        if stream:
            request["stream_options"] = {"include_usage": True}
        if extra := self.body():
            request["extra_body"] = extra
        return request

    # -------- WHAT A SERVICE DOES DIFFERENTLY ------------------------------------------
    def body(self) -> dict[str, t.Any]:
        """Fields added to every request as they are: how the model thinks
        (``config.thinking``) and the ``options``."""
        return {**(self.config.thinking or {}), **self.options}

    def check(self, payload: t.Any) -> None:
        """Raise if an answer or a piece carries an error inside. Nothing here."""

    # -------- READING THE ANSWER -------------------------------------------------------
    def thinking(self, message: t.Any) -> str | None:
        """The thinking text of a message or a piece. Not sent in this format."""
        return None

    def tool_calls(self, raw_calls: list[t.Any]) -> list[ToolCall]:
        """The tool calls of an answer. Arguments that are not JSON are kept
        with a mark, so the model is told and can fix them."""
        calls: list[ToolCall] = []
        for raw in raw_calls:
            function = self.get(raw, "function")
            name = self.get(function, "name")
            if not name:
                continue
            arguments: str = self.get(function, "arguments") or ""
            parameters: dict[str, t.Any]
            try:
                parameters = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                parameters = {"raw_error_content": arguments, "parsing_error": True}
            given: dict[str, str] = {"id": identifier} if (identifier := self.get(raw, "id")) else {}
            calls.append(ToolCall(tool_name=name, parameters=parameters, **given))
        return calls

    @classmethod
    def response_format(cls, output_format: type[BaseModel]) -> dict[str, t.Any]:
        """Ask for JSON that follows ``output_format`` exactly."""
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", output_format.__name__)[:64] or "structured_output"
        schema = cls.strict(output_format.model_json_schema())
        return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
