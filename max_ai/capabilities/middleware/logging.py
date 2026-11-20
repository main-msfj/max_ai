"""Structured log lines for a turn, its model calls and its tool calls.

    → turn started | task=new
    → model call | model=gpt-5 | messages=12 | tools=9
    ← model call | ms=1240 | tokens_in=4500 | tokens_out=80 | tool_calls=1 | finish_reason=tool_calls
    → tool get_weather | params={"city": "Tokyo"}        (params only at level="debug")
    ← tool get_weather | ms=12 | ok=True
    ← turn finished | finish_reason=stop | seconds=3.2

Observes only: never changes a call or a result.
"""

from __future__ import annotations

import json
import logging
import time
import typing as t
from collections.abc import AsyncGenerator

from pydantic import Field

from ...base.middleware import (
    CoreMiddleware,
    MiddlewareConfig,
    ModelCall,
    NextModel,
    NextStream,
    NextTool,
    NextTurn,
)
from ...core.event_type import ReasoningCompleteEvent
from ...loggers import ScopedLogger

if t.TYPE_CHECKING:
    from ...core.event_type import CoreEvent
    from ...core.messages import AssistantMessage
    from ...types.completions import LLMChunk, LLMResponse, Usage
    from ...types.run_context import RunContext
    from ...types.tool_call import ToolCallRecord, ToolResult


class LoggingMiddlewareConfig(MiddlewareConfig):
    """Configuration options for ``LoggingMiddleware``."""
    level: t.Literal["info", "debug"] = Field(
        default="info",
        description="debug adds tool parameters and results.",
    )
    preview_chars: int = Field(default=200, ge=20)


class LoggingMiddleware(CoreMiddleware):
    """Log what the agent does, one line per step."""

    component_provider_override = "max_ai.capabilities.middleware.LoggingMiddleware"
    component_schema = LoggingMiddlewareConfig

    def __init__(self, level: t.Literal["info", "debug"] = "info", preview_chars: int = 200) -> None:
        self.level = level
        self.preview_chars = preview_chars
        self._log = ScopedLogger(logging.getLogger(__name__), scope=["LoggingMiddleware"])

    # -------- THE FOUR PLACES ----------------------------------------------------------
    async def turn(self, ctx: RunContext, call_next: NextTurn) -> AsyncGenerator[CoreEvent]:
        started, finish = time.monotonic(), None
        self._write("→ turn started", ctx, task="new" if ctx.inbox else "goes on")
        try:
            async for event in call_next():
                if isinstance(event, ReasoningCompleteEvent):
                    finish = event.finish_reason
                yield event
        finally:
            self._write(
                "← turn finished", ctx, finish_reason=finish,
                seconds=round(time.monotonic() - started, 1),
            )

    async def model(self, ctx: RunContext, call: ModelCall, call_next: NextModel) -> LLMResponse:
        started = self._model_call(ctx, call)
        try:
            result = await call_next(call)
        except Exception as error:
            self._model_error(ctx, started, error)
            raise
        self._model_answer(ctx, started, result.message, result.usage, result.finish_reason)
        return result

    async def stream(
        self, ctx: RunContext, call: ModelCall, call_next: NextStream
    ) -> AsyncGenerator[LLMChunk]:
        started = self._model_call(ctx, call)
        try:
            async for chunk in call_next(call):
                if chunk.is_complete and chunk.message is not None and chunk.usage is not None:
                    self._model_answer(ctx, started, chunk.message, chunk.usage, chunk.finish_reason)
                yield chunk
        except Exception as error:
            self._model_error(ctx, started, error)
            raise

    async def tool(self, ctx: RunContext, record: ToolCallRecord, call_next: NextTool) -> ToolResult | None:
        started = time.monotonic()
        details = {"params": self._preview(record.parameters)} if self.level == "debug" else {}
        self._write(f"→ tool {record.tool_name}", ctx, status=record.status.value, **details)
        result = await call_next(record)
        # 1. None: it waits for a person.
        if result is None:
            self._write(f"← tool {record.tool_name}", ctx, waiting="user")
            return result
        answer: dict[str, t.Any] = {"ms": _ms(started), "ok": result.success}
        if not result.success:
            answer["error"] = self._preview(result.error or "")
        elif self.level == "debug":
            answer["result"] = self._preview(result.result)
        self._write(f"← tool {record.tool_name}", ctx, **answer)
        return result

    # -------- HELPERS ------------------------------------------------------------------
    def _model_call(self, ctx: RunContext, call: ModelCall) -> float:
        self._write("→ model call", ctx, model=call.model, messages=len(call.messages), tools=len(call.tools))
        return time.monotonic()

    def _model_answer(
        self, ctx: RunContext, started: float, message: AssistantMessage, usage: Usage, finish: str | None
    ) -> None:
        self._write(
            "← model call", ctx, ms=_ms(started), tokens_in=usage.tokens_input,
            tokens_out=usage.tokens_output, tool_calls=len(message.tool_calls), finish_reason=finish,
        )

    def _model_error(self, ctx: RunContext, started: float, error: Exception) -> None:
        self._log.error(
            "✗ model call", run_id=ctx.run_id, ms=_ms(started),
            error_type=type(error).__name__, error=str(error),
        )

    def _write(self, text: str, ctx: RunContext, **details: t.Any) -> None:
        # Readable with any formatter; also structured (``extra``) for JSON logs.
        shown = " | ".join(f"{key}={value}" for key, value in details.items() if value is not None)
        line = f"{text} | {shown}" if shown else text
        self._log.info(line, run_id=ctx.run_id, session_id=ctx.session_id, **details)

    def _preview(self, value: t.Any) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        return text if len(text) <= self.preview_chars else text[: self.preview_chars - 1] + "…"


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


__all__ = ["LoggingMiddleware", "LoggingMiddlewareConfig"]
