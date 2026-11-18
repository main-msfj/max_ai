"""The middleware list, run around each place: in order on the way in, in
reverse on the way out. A middleware that breaks stops the turn: nothing
runs without it."""

from __future__ import annotations

import copy
import logging
import typing as t
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

from ...base.middleware import CoreMiddleware, ModelCall, NextModel, NextStream, NextTool, NextTurn
from ...errors.middleware import StopRun
from ...types.completions import ChatCompletionChunk, ChatCompletionResult
from ...types.tool_call import ToolCallRecord, ToolResult

if t.TYPE_CHECKING:
    from ...types.run_context import RunContext
    from ..event_type import CoreEvent

logger = logging.getLogger(__name__)

A = t.TypeVar("A")
R = t.TypeVar("R")
Y = t.TypeVar("Y")

BROKEN = "middleware_error"  # the finish reason of a turn a broken middleware stopped
PARAMETERS_CHANGED = "A middleware changed this call's parameters, so it did not run."


class MiddlewareChain:
    """Built once per agent; the agent, the client and the dispatcher use it."""

    def __init__(self, middlewares: Sequence[CoreMiddleware] = ()) -> None:
        self.middlewares = list(middlewares)

    def __bool__(self) -> bool:
        return bool(self.middlewares)

    # -------- THE FOUR PLACES ----------------------------------------------------------
    def turn(self, ctx: RunContext, action: NextTurn) -> AsyncGenerator[CoreEvent]:
        """Every event of the turn, through each middleware."""
        def innermost(_: None) -> AsyncGenerator[CoreEvent]:
            return action()

        call_next: Callable[[None], AsyncGenerator[CoreEvent]] = innermost
        for middleware in reversed(self.middlewares):
            call_next = self._streamed(
                middleware, call_next, lambda _, inner, m=middleware: m.turn(ctx, lambda: inner(None))
            )
        return call_next(None)

    async def model(self, ctx: RunContext, call: ModelCall, action: NextModel) -> ChatCompletionResult:
        """One whole model call, through each middleware."""
        call_next = action
        for middleware in reversed(self.middlewares):
            call_next = self._awaited(
                middleware, call_next, lambda given, inner, m=middleware: m.model(ctx, given, inner)
            )
        return await call_next(call)

    def stream(self, ctx: RunContext, call: ModelCall, action: NextStream) -> AsyncGenerator[ChatCompletionChunk]:
        """One streamed model call. The chunks pass as the provider sends
        them; with a middleware that filters text, as whole lines."""
        whole_lines = any(middleware.filter_full_text for middleware in self.middlewares)

        # Whole lines, so a word cut in two can't slip past a filter.
        def innermost(given: ModelCall) -> AsyncGenerator[ChatCompletionChunk]:
            return self.lines(action(given)) if whole_lines else action(given)

        call_next: NextStream = innermost
        for middleware in reversed(self.middlewares):
            call_next = self._streamed(
                middleware, call_next, lambda given, inner, m=middleware: m.stream(ctx, given, inner)
            )
        return call_next(call)

    async def tool(self, ctx: RunContext, record: ToolCallRecord, action: NextTool) -> ToolResult | None:
        """Every tool call, through each middleware. Its parameters can't change."""
        sent = copy.deepcopy(record.parameters)

        # 1. Innermost: the call runs only as the model sent it.
        async def guarded(given: ToolCallRecord) -> ToolResult | None:
            if given is not record or given.parameters != sent:
                record.parameters = sent
                return ToolResult.execution_error(record.id, PARAMETERS_CHANGED)
            return await action(given)

        call_next: NextTool = guarded
        for middleware in reversed(self.middlewares):
            call_next = self._awaited(
                middleware, call_next, lambda given, inner, m=middleware: m.tool(ctx, given, inner)
            )
        return await call_next(record)

    # -------- ONE LAYER ----------------------------------------------------------------
    @staticmethod
    def _awaited(
        middleware: CoreMiddleware,
        inner: Callable[[A], Awaitable[R]],
        run: Callable[[A, Callable[[A], Awaitable[R]]], Awaitable[R]],
    ) -> Callable[[A], Awaitable[R]]:
        """One middleware around ``inner``. If it breaks on its own, the turn
        is stopped: nothing goes on without a middleware."""

        async def layer(given: A) -> R:
            failed: list[BaseException] = []

            # 1. What is inside (the model, the tool); its own error is remembered.
            async def call_next(value: A) -> R:
                try:
                    return await inner(value)
                except BaseException as error:
                    failed.append(error)
                    raise

            try:
                return await run(given, call_next)
            except StopRun:
                raise
            except Exception as error:
                # 2. The model or the tool failed: that error goes up as it is.
                if failed and error is failed[-1]:
                    raise
                # 3. The middleware's own error: the turn ends.
                raise MiddlewareChain._broken(middleware, error) from error

        return layer

    @staticmethod
    def _streamed(
        middleware: CoreMiddleware,
        inner: Callable[[A], AsyncGenerator[Y]],
        run: Callable[[A, Callable[[A], AsyncGenerator[Y]]], AsyncGenerator[Y]],
    ) -> Callable[[A], AsyncGenerator[Y]]:
        """The same for what comes piece by piece."""

        async def layer(given: A) -> AsyncGenerator[Y]:
            failed: list[BaseException] = []

            async def call_next(value: A) -> AsyncGenerator[Y]:
                try:
                    async for piece in inner(value):
                        yield piece
                except BaseException as error:
                    failed.append(error)
                    raise

            try:
                async for piece in run(given, call_next):
                    yield piece
            except StopRun:
                raise
            except Exception as error:
                if failed and error is failed[-1]:
                    raise
                raise MiddlewareChain._broken(middleware, error) from error

        return layer

    @staticmethod
    def _broken(middleware: CoreMiddleware, error: Exception) -> StopRun:
        """A middleware failed: logged, and the turn stops as if it had asked to."""
        name = type(middleware).__name__
        logger.error("Middleware %s failed; the turn is stopped", name, exc_info=error)
        return StopRun(
            f"The turn was stopped: {name} failed.", finish_reason=BROKEN, label=f"{name} failed"
        )

    # -------- LINES --------------------------------------------------------------------
    @staticmethod
    async def lines(chunks: AsyncGenerator[ChatCompletionChunk]) -> AsyncGenerator[ChatCompletionChunk]:
        """The provider's chunks as whole lines; the final chunk goes last, as it is."""
        text = thinking = ""
        async for chunk in chunks:
            text, thinking = text + chunk.content, thinking + (chunk.thinking or "")
            # 1. The end: what is left goes out, then the chunk with the whole message.
            if chunk.is_complete:
                if text or thinking:
                    yield ChatCompletionChunk(content=text, thinking=thinking or None, is_complete=False)
                yield chunk.model_copy(update={"content": "", "thinking": None})
                return
            # 2. Up to the last line break goes out; the rest waits for its line.
            cut, cut_thinking = text.rfind("\n") + 1, thinking.rfind("\n") + 1
            if cut or cut_thinking:
                yield ChatCompletionChunk(
                    content=text[:cut], thinking=thinking[:cut_thinking] or None, is_complete=False
                )
                text, thinking = text[cut:], thinking[cut_thinking:]
