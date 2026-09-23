"""Structured log lines for a run, its model calls and its tool calls.

    → run started | task=1 message(s)
    → model call | model=gpt-5 | messages=12 | tools=9
    ← model call | ms=1240 | tokens_in=4500 | tokens_out=80 | tool_calls=1 | finish_reason=tool_calls
    → tool get_weather | params={"city": "Tokyo"}        (params only at level="debug")
    ← tool get_weather | ms=12 | ok=True
    ← run finished | finish_reason=stop | seconds=3.2

Observes only: never changes a request or a result.
"""

from __future__ import annotations

import json
import logging
import time
import typing as t

from pydantic import Field

from ...base.middleware import (
    CoreMiddleware,
    MiddlewareConfig,
    MiddlewareContext,
    ModelRequest,
    ToolRequest,
)
from ...loggers import ScopedLogger

if t.TYPE_CHECKING:
    from ...core.messages import CoreMessage
    from ...types.agent_response import AgentResponse
    from ...types.completions import ChatCompletionResult
    from ...types.tool_call import ToolResult

_STARTED = "log_started"


class LoggingMiddlewareConfig(MiddlewareConfig):
    level: t.Literal["info", "debug"] = Field(
        default="info",
        description="debug adds tool parameters and results.",
    )
    preview_chars: int = Field(default=200, ge=20)


class LoggingMiddleware(CoreMiddleware):
    """Log what the agent does, one line per step."""

    component_provider_override = "max_ai.capabilities.middleware.LoggingMiddleware"
    component_schema = LoggingMiddlewareConfig

    def __init__(
        self, level: t.Literal["info", "debug"] = "info", preview_chars: int = 200
    ) -> None:
        self.level = level
        self.preview_chars = preview_chars
        self._log = ScopedLogger(
            logging.getLogger(__name__), scope=["LoggingMiddleware"]
        )

    def _write(self, text: str, mw: MiddlewareContext, **details: t.Any) -> None:
        # Readable with any formatter; also structured (``extra``) for JSON logs.
        shown = " | ".join(
            f"{key}={value}" for key, value in details.items() if value is not None
        )
        line = f"{text} | {shown}" if shown else text
        self._log.info(line, run_id=mw.ctx.run_id, agent=mw.agent, **details)

    def _preview(self, value: t.Any) -> str:
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, default=str)
        )
        return (
            text
            if len(text) <= self.preview_chars
            else text[: self.preview_chars - 1] + "…"
        )

    # -------- HOOKS -----------------------------------------------------------
    async def on_run_start(
        self, mw: MiddlewareContext, task: list[CoreMessage] | None
    ) -> None:
        mw.state(self)[_STARTED] = time.monotonic()
        self._write(
            "→ run started", mw, task=f"{len(task)} message(s)" if task else "resume"
        )

    async def on_run_end(self, mw: MiddlewareContext, response: AgentResponse) -> None:
        started = mw.state(self).pop(_STARTED, None)
        seconds = round(time.monotonic() - started, 1) if started else None
        self._write(
            "← run finished", mw, finish_reason=response.finish_reason, seconds=seconds
        )

    async def on_model_request(
        self, mw: MiddlewareContext, request: ModelRequest
    ) -> ModelRequest:
        request.metadata[_STARTED] = time.monotonic()
        self._write(
            "→ model call",
            mw,
            model=request.model,
            messages=len(request.messages),
            tools=len(request.tools),
        )
        return request

    async def on_model_response(
        self,
        mw: MiddlewareContext,
        request: ModelRequest,
        result: ChatCompletionResult,
    ) -> ChatCompletionResult:
        self._write(
            "← model call",
            mw,
            ms=_elapsed_ms(request.metadata),
            tokens_in=result.usage.tokens_input,
            tokens_out=result.usage.tokens_output,
            tool_calls=len(result.message.tool_calls),
            finish_reason=result.finish_reason,
        )
        return result

    async def on_model_error(
        self,
        mw: MiddlewareContext,
        request: ModelRequest,
        error: Exception,
    ) -> None:
        self._log.error(
            "✗ model call",
            run_id=mw.ctx.run_id,
            ms=_elapsed_ms(request.metadata),
            error_type=type(error).__name__,
            error=str(error),
        )
        return None

    async def on_tool_request(
        self, mw: MiddlewareContext, request: ToolRequest
    ) -> None:
        request.metadata[_STARTED] = time.monotonic()
        details = (
            {"params": self._preview(request.parameters)}
            if self.level == "debug"
            else {}
        )
        self._write(f"→ tool {request.tool_name}", mw, **details)
        return None

    async def on_tool_response(
        self,
        mw: MiddlewareContext,
        request: ToolRequest,
        result: ToolResult,
    ) -> ToolResult:
        details: dict[str, t.Any] = {
            "ms": _elapsed_ms(request.metadata),
            "ok": result.success,
        }
        if not result.success:
            details["error"] = self._preview(result.error or "")
        elif self.level == "debug":
            details["result"] = self._preview(result.result)
        self._write(f"← tool {request.tool_name}", mw, **details)
        return result


def _elapsed_ms(metadata: dict[str, t.Any]) -> int | None:
    started = metadata.get(_STARTED)
    return int((time.monotonic() - started) * 1000) if started else None


__all__ = ["LoggingMiddleware", "LoggingMiddlewareConfig"]
