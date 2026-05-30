"""Console tracing middleware for live agent debugging."""

from __future__ import annotations

import json
import sys
import time
import typing as t
from collections.abc import AsyncGenerator
from datetime import datetime

from ..base.middleware import CoreMiddleware
from ..core.event_type import CoreEvent
from ..types.completions import ChatCompletionChunk, ChatCompletionResult
from ..types.middleware import MiddlewareCtx
from ..types.tool_call import ToolResult

_START_KEY = "_console_trace_start"
_DEFAULT_PREVIEW_CHARS = 240


class ConsoleTraceMiddleware(CoreMiddleware):
    """Print human-readable model/tool activity to the console.

    This middleware is meant for local development and server console
    debugging. It does not modify requests or responses; it only prints
    what the LLM is about to do and what came back.
    """

    def __init__(
        self,
        *,
        stream: t.TextIO | None = None,
        preview_chars: int = _DEFAULT_PREVIEW_CHARS,
        show_params: bool = True,
        show_messages: bool = True,
        show_stream_chunks: bool = False,
    ) -> None:
        self.stream = stream or sys.stderr
        self.preview_chars = preview_chars
        self.show_params = show_params
        self.show_messages = show_messages
        self.show_stream_chunks = show_stream_chunks

    async def on_request(
        self, ctx: MiddlewareCtx
    ) -> AsyncGenerator[MiddlewareCtx | CoreEvent, None]:
        ctx.metadata[_START_KEY] = time.monotonic()
        if ctx.action in {"model_call", "model_call_stream"}:
            self._print_model_request(ctx)
        elif ctx.action == "tool_call":
            self._print_tool_request(ctx)
        else:
            self._write(f"-> {ctx.action} started run={ctx.ctx.run_id}")
        yield ctx

    async def on_response(
        self, ctx: MiddlewareCtx, result: t.Any
    ) -> AsyncGenerator[t.Any | CoreEvent, None]:
        duration = self._duration_ms(ctx)
        if ctx.action in {"model_call", "model_call_stream"}:
            self._print_model_response(ctx, result, duration)
        elif ctx.action == "tool_call":
            self._print_tool_response(ctx, result, duration)
        else:
            self._write(f"<- {ctx.action} finished {duration}ms run={ctx.ctx.run_id}")
        yield result

    async def on_error(
        self, ctx: MiddlewareCtx, error: Exception
    ) -> AsyncGenerator[t.Any | CoreEvent, None]:
        duration = self._duration_ms(ctx)
        self._write(
            f"!! {ctx.action} failed {duration}ms "
            f"run={ctx.ctx.run_id} error={type(error).__name__}: {error}"
        )
        raise error
        yield  # pragma: no cover

    async def on_stream_chunk(
        self, ctx: MiddlewareCtx, chunk: t.Any
    ) -> AsyncGenerator[t.Any | CoreEvent, None]:
        if self.show_stream_chunks and isinstance(chunk, ChatCompletionChunk):
            if chunk.content:
                self._write(f".. model token {self._truncate(chunk.content)!r}")
            if chunk.thinking:
                self._write(f".. model thinking {self._truncate(chunk.thinking)!r}")
            if chunk.tool_call_chunk:
                self._write(
                    ".. model tool-call chunk "
                    f"{self._truncate(json.dumps(chunk.tool_call_chunk, default=str))}"
                )
        yield chunk

    def _print_model_request(self, ctx: MiddlewareCtx) -> None:
        model = ctx.metadata.get("model") or "unknown"
        tools = ctx.metadata.get("tools") or []
        messages = self._messages(ctx.data)
        self._write(
            f"-> LLM {ctx.action} model={model} "
            f"messages={len(messages)} tools={len(tools)} run={ctx.ctx.run_id}"
        )
        if tools:
            names = [getattr(tool, "name", str(tool)) for tool in tools]
            self._write(f"   tools: {', '.join(names)}")
        if self.show_messages and messages:
            for message in messages[-4:]:
                role = getattr(message, "role", "message")
                source = getattr(message, "source", "unknown")
                text = self._message_text(message)
                self._write(f"   msg[{role}/{source}]: {self._truncate(text)}")

    def _print_model_response(
        self, ctx: MiddlewareCtx, result: t.Any, duration_ms: int
    ) -> None:
        if isinstance(result, ChatCompletionResult):
            message = result.message
            tool_calls = getattr(message, "tool_calls", []) or []
            self._write(
                f"<- LLM finished {duration_ms}ms finish={result.finish_reason} "
                f"tool_calls={len(tool_calls)} "
                f"tokens_in={result.usage.tokens_input} "
                f"tokens_out={result.usage.tokens_output}"
            )
            if tool_calls:
                for call in tool_calls:
                    self._write(
                        f"   wants tool: {call.tool_name} "
                        f"params={self._json(call.parameters)}"
                    )
            text = message.text()
            if text:
                self._write(f"   assistant: {self._truncate(text)}")
            return
        self._write(f"<- LLM finished {duration_ms}ms result={type(result).__name__}")

    def _print_tool_request(self, ctx: MiddlewareCtx) -> None:
        record = ctx.data
        name = getattr(record, "tool_name", "unknown")
        call_id = getattr(record, "id", "?")
        self._write(f"-> TOOL {name} call_id={call_id} run={ctx.ctx.run_id}")
        if self.show_params:
            params = getattr(record, "parameters", {}) or {}
            self._write(f"   params: {self._json(params)}")

    def _print_tool_response(
        self, ctx: MiddlewareCtx, result: t.Any, duration_ms: int
    ) -> None:
        record = ctx.data
        name = getattr(record, "tool_name", "unknown")
        if isinstance(result, ToolResult):
            status = "ok" if result.success else "failed"
            preview_source = result.result if result.success else result.error
            self._write(f"<- TOOL {name} {status} {duration_ms}ms")
            if preview_source is not None:
                self._write(f"   result: {self._truncate(str(preview_source))}")
            return
        self._write(f"<- TOOL {name} finished {duration_ms}ms result={type(result).__name__}")

    def _duration_ms(self, ctx: MiddlewareCtx) -> int:
        start = ctx.metadata.get(_START_KEY)
        if start is None:
            return 0
        return int((time.monotonic() - start) * 1000)

    def _messages(self, data: t.Any) -> list[t.Any]:
        if hasattr(data, "messages"):
            return list(getattr(data, "messages") or [])
        if isinstance(data, list):
            return data
        return []

    def _message_text(self, message: t.Any) -> str:
        text = getattr(message, "text", None)
        if callable(text):
            return str(text())
        content = getattr(message, "content", "")
        return str(content)

    def _json(self, value: t.Any) -> str:
        try:
            return self._truncate(json.dumps(value, ensure_ascii=False, default=str))
        except TypeError:
            return self._truncate(str(value))

    def _truncate(self, value: str) -> str:
        value = str(value).replace("\n", " ").strip()
        if len(value) <= self.preview_chars:
            return value
        return value[: self.preview_chars] + "..."

    def _write(self, line: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[max-ai trace {timestamp}] {line}", file=self.stream, flush=True)
