"""The middleware list, run around each place: in order on the way in, in
reverse on the way out. A middleware that breaks is logged and skipped;
``StopRun`` always goes through."""

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
                middleware, call_next, lambda inner, m=middleware: m.turn(ctx, lambda: inner(None))
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
        """One streamed model call; the middlewares see it line by line."""
        # 1. Nobody watching: the chunks go out as the provider sends them.
        if not self.middlewares:
            return action(call)
        # 2. Whole lines, so a word cut in two can't slip past a filter.
        def innermost(given: ModelCall) -> AsyncGenerator[ChatCompletionChunk]:
            return self.lines(action(given))

        call_next: NextStream = innermost
        for middleware in reversed(self.middlewares):
            call_next = self._streamed(
                middleware, call_next, lambda inner, m=middleware: m.stream(ctx, call, inner)
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
        """One middleware around ``inner``. If it breaks on its own, the call
        goes on without it, and what is inside never runs twice."""

        async def layer(given: A) -> R:
            done: list[R] = []
            failed: list[BaseException] = []

            # 1. What is inside (the model, the tool), remembered.
            async def call_next(value: A) -> R:
                try:
                    result = await inner(value)
                except BaseException as error:
                    failed.append(error)
                    raise
                done.append(result)
                return result

            try:
                return await run(given, call_next)
            except StopRun:
                raise
            except Exception:
                # 2. The model or the tool failed: that error goes up.
                if failed:
                    raise
                # 3. The middleware's own error: logged, and the call goes on.
                logger.exception("Middleware %s failed; skipped", type(middleware).__name__)
                return done[-1] if done else await inner(given)

        return layer

    @staticmethod
    def _streamed(
        middleware: CoreMiddleware,
        inner: Callable[[A], AsyncGenerator[Y]],
        run: Callable[[Callable[[A], AsyncGenerator[Y]]], AsyncGenerator[Y]],
    ) -> Callable[[A], AsyncGenerator[Y]]:
        """The same for what comes piece by piece: if the middleware breaks,
        the rest comes straight from inside, from where it was."""

        async def layer(given: A) -> AsyncGenerator[Y]:
            source: list[AsyncGenerator[Y]] = []
            failed: list[BaseException] = []

            # 1. What is inside, kept so it can go on without the middleware.
            async def call_next(value: A) -> AsyncGenerator[Y]:
                source.append(pieces := inner(value))
                try:
                    async for piece in pieces:
                        yield piece
                except BaseException as error:
                    failed.append(error)
                    raise

            try:
                async for piece in run(call_next):
                    yield piece
                return
            except StopRun:
                raise
            except Exception:
                if failed:
                    raise
                logger.exception("Middleware %s failed; skipped", type(middleware).__name__)
            # 2. The rest, straight from inside.
            async for piece in source[0] if source else inner(given):
                yield piece

        return layer

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
