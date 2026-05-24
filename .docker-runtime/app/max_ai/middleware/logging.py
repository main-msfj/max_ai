"""
Logging middleware.

Observes ``model_call`` and ``tool_call`` actions flowing through the
``MiddlewareChain`` and emits structured log lines via ``ScopedLogger``.
Does not modify requests or responses.

Output format (pipe-separated, arrow-prefixed):

    → model_call started | model=qwen3:4b | msgs=3 | tools=2
    ← model_call finished | duration=1240ms | tokens_in=450 | tokens_out=80
    → tool_call started | name=get_weather | params={"city": "Tokyo"}
    ← tool_call finished | duration=12ms | success=true

Levels:
    info  — start / finish lines for every action. Default.
    debug — adds full payloads (messages, tool params), retries, and
            stack traces on errors.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import typing as t
from collections.abc import AsyncGenerator

from ..base.middleware import CoreMiddleware
from ..core.event_type import CoreEvent
from ..loggers import ScopedLogger
from ..types.completions import ChatCompletionResult
from ..types.middleware import MiddlewareCtx
from ..types.tool_call import ToolResult


# Metadata key used to thread the start timestamp from on_request to
# on_response. Prefixed with underscore to signal it's internal to this
# middleware — other middlewares should not read or write it.
_START_TIME_KEY = "_log_start_time"

# Truncation limit for debug-level payload previews. Anything beyond
# this gets cut with an ellipsis.
_DEBUG_PREVIEW_CHARS = 200


class LoggingMiddleware(CoreMiddleware):
    """Emits structured log lines for every action.

    Args:
        level: ``"info"`` or ``"debug"``. Info logs start/finish lines
            with timing and tokens. Debug adds full payloads and stack
            traces.
        logger: Underlying ``logging.Logger`` to use. If ``None``, the
            middleware creates one named ``max_ai.middleware.logging``.
        log_streaming: If True, log every streaming chunk. Off by
            default — chunk-level logging is noisy and rarely useful
            outside of debugging the streaming pipeline itself.
    """

    def __init__(
        self,
        level: t.Literal["info", "debug"] = "info",
        logger: logging.Logger | None = None,
        log_streaming: bool = False,
    ) -> None:
        self.level = level
        self.log_streaming = log_streaming

        base = logger or logging.getLogger("max_ai.middleware.logging")
        self._log = ScopedLogger(base, scope=["LoggingMiddleware"])

    @property
    def _is_debug(self) -> bool:
        return self.level == "debug"

    # -------- HOOKS -----------------------------------------------------------
    async def on_request(
        self, ctx: MiddlewareCtx
    ) -> AsyncGenerator[MiddlewareCtx | CoreEvent, None]:
        """Stamp the start time and emit a  start line."""
        ctx.metadata[_START_TIME_KEY] = time.monotonic()

        if ctx.action == "model_call" or ctx.action == "model_call_stream":
            self._log_model_request(ctx)
        elif ctx.action == "tool_call":
            self._log_tool_request(ctx)
        else:
            # Unknown action — log a generic line at debug level so we
            # don't spam at info, but we still see it if needed.
            if self._is_debug:
                self._log.debug(
                    f"→ {ctx.action} started",
                    run_id=ctx.ctx.run_id,
                )

        yield ctx

    async def on_response(
        self, ctx: MiddlewareCtx, result: t.Any
    ) -> AsyncGenerator[t.Any | CoreEvent, None]:
        """Emit a  finish line with timing and outcome details."""
        duration_ms = self._duration_ms(ctx)

        if ctx.action == "model_call" or ctx.action == "model_call_stream":
            self._log_model_response(ctx, result, duration_ms)
        elif ctx.action == "tool_call":
            self._log_tool_response(ctx, result, duration_ms)
        else:
            if self._is_debug:
                self._log.debug(
                    f"← {ctx.action} finished",
                    run_id=ctx.ctx.run_id,
                    duration_ms=duration_ms,
                )

        # Pass-through: never modify the result.
        yield result

    async def on_error(
        self, ctx: MiddlewareCtx, error: Exception
    ) -> AsyncGenerator[t.Any | CoreEvent, None]:
        """Log the error and re-raise. Never swallows."""
        duration_ms = self._duration_ms(ctx)
        details = {
            "run_id": ctx.ctx.run_id,
            "action": ctx.action,
            "error_type": type(error).__name__,
            "error": str(error),
            "duration_ms": duration_ms,
        }
        if self._is_debug:
            details["traceback"] = traceback.format_exc()

        self._log.error(f"✗ {ctx.action} failed", **details)
        raise error
        yield  # pragma: no cover — required by ABC signature

    async def on_stream_chunk(
        self, ctx: MiddlewareCtx, chunk: t.Any
    ) -> AsyncGenerator[t.Any | CoreEvent, None]:
        """Log streaming chunks if explicitly enabled."""
        if self.log_streaming and self._is_debug:
            self._log.debug(
                f"  · {ctx.action} chunk",
                run_id=ctx.ctx.run_id,
                preview=self._preview(chunk),
            )
        yield chunk

    # -------- LINE BUILDERS -----------------------------------------------------------
    def _log_model_request(self, ctx: MiddlewareCtx) -> None:
        """Build and emit the start line for a model_call."""
        model = ctx.metadata.get("model") or "?"

        # ctx.data for model_call is the RunContext (so the chain can
        # introspect messages). Count messages defensively — the data
        # shape can vary by reasoning loop.
        msg_count = self._count_messages(ctx.data)
        tools_count = self._count_tools(ctx.metadata)

        line = (
            f"→ {ctx.action} started "
            f"| model={model} | msgs={msg_count} | tools={tools_count}"
        )
        self._log.info(line, run_id=ctx.ctx.run_id)

        if self._is_debug:
            self._log.debug(
                "  · request payload",
                run_id=ctx.ctx.run_id,
                messages_preview=self._messages_preview(ctx.data),
            )

    def _log_model_response(
        self, ctx: MiddlewareCtx, result: t.Any, duration_ms: int
    ) -> None:
        """Build and emit the finish line for a model_call."""
        tokens_in = 0
        tokens_out = 0
        finish_reason = True

        if isinstance(result, ChatCompletionResult):
            tokens_in = result.usage.tokens_input
            tokens_out = result.usage.tokens_output
            finish_reason = result.finish_reason

        line = (
            f"← {ctx.action} finished "
            f"| duration={duration_ms}ms "
            f"| tokens_in={tokens_in} | tokens_out={tokens_out} "
            f"| finish={finish_reason}"
        )
        self._log.info(line, run_id=ctx.ctx.run_id)

        if self._is_debug and isinstance(result, ChatCompletionResult):
            self._log.debug(
                "  · response payload",
                run_id=ctx.ctx.run_id,
                content_preview=self._truncate(result.message.text()),
                tool_calls=len(result.message.tool_calls),
            )

    def _log_tool_request(self, ctx: MiddlewareCtx) -> None:
        """Build and emit the start line for a tool_call."""
        record = ctx.data
        tool_name = getattr(record, "tool_name", "?")
        params = getattr(record, "parameters", {}) or {}

        params_repr = (
            self._truncate(json.dumps(params, default=str))
            if self._is_debug
            else f"{len(params)} param(s)"
        )

        line = (
            f"→ {ctx.action} started "
            f"| name={tool_name} | params={params_repr}"
        )
        self._log.info(line, run_id=ctx.ctx.run_id)

    def _log_tool_response(
        self, ctx: MiddlewareCtx, result: t.Any, duration_ms: int
    ) -> None:
        """Build and emit the finish line for a tool_call."""
        record = ctx.data
        tool_name = getattr(record, "tool_name", "?")

        success = "?"
        result_preview: str | None = None

        if isinstance(result, ToolResult):
            success = "true" if result.success else "false"
            if self._is_debug:
                preview_source = (
                    str(result.result) if result.success else (result.error or "")
                )
                result_preview = self._truncate(preview_source)

        line = (
            f"← {ctx.action} finished "
            f"| name={tool_name} "
            f"| duration={duration_ms}ms | success={success}"
        )
        self._log.info(line, run_id=ctx.ctx.run_id)

        if result_preview is not None:
            self._log.debug(
                "  · tool output",
                run_id=ctx.ctx.run_id,
                preview=result_preview,
            )

    # -------- HELPERS -----------------------------------------------------------
    def _duration_ms(self, ctx: MiddlewareCtx) -> int:
        start = ctx.metadata.get(_START_TIME_KEY)
        if start is None:
            return 0
        return int((time.monotonic() - start) * 1000)

    def _count_messages(self, data: t.Any) -> int:
        """Defensive message count.

        Reasoning loops may pass ``ctx.messages`` directly or the
        whole ``RunContext``; handle both.
        """
        if data is None:
            return 0
        if hasattr(data, "messages"):
            try:
                return len(data.messages)
            except TypeError:
                return 0
        if isinstance(data, list):
            return len(data)
        return 0

    def _count_tools(self, metadata: dict[str, t.Any]) -> int:
        tools = metadata.get("tools")
        if tools is None:
            return 0
        try:
            return len(tools)
        except TypeError:
            return 0

    def _messages_preview(self, data: t.Any) -> list[str]:
        """A tiny preview of the last few messages for debug logs."""
        if hasattr(data, "messages"):
            messages = data.messages
        elif isinstance(data, list):
            messages = data
        else:
            return []

        previews: list[str] = []
        for msg in messages[-3:]:
            text = getattr(msg, "text", None)
            text_value = text() if callable(text) else (text or "")
            role = getattr(msg, "role", "?")
            previews.append(f"{role}: {self._truncate(text_value)}")
        return previews

    @staticmethod
    def _truncate(s: str | None) -> str:
        if s is None:
            return ""
        s = str(s)
        if len(s) <= _DEBUG_PREVIEW_CHARS:
            return s
        return s[:_DEBUG_PREVIEW_CHARS] + "…"

    @classmethod
    def _preview(cls, item: t.Any) -> str:
        if hasattr(item, "model_dump"):
            try:
                return cls._truncate(json.dumps(item.model_dump(), default=str))
            except Exception:
                return cls._truncate(repr(item))
        return cls._truncate(repr(item))