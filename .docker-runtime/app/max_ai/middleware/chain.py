from __future__ import annotations

import logging
import typing as t
from collections.abc import AsyncGenerator

from ..types.middleware import MiddlewareCtx
from ..loggers import MiddlewareLogger
from ..types.run_context import RunContext
from ..core.event_type import CoreEvent, ToolApprovalEvent

if t.TYPE_CHECKING:
    from ..base.middleware import CoreMiddleware

class MiddlewareChain:
    """Executes a middleware pipeline over agent actions (sync + streaming)."""

    def __init__(
        self,
        middlewares: list[CoreMiddleware] | None = None,
        mw_logger: logging.Logger | None = None,
    ):
        self.middlewares = middlewares or []
        self.mw_logger = mw_logger or logging.getLogger(__name__)

    def add(self, middleware: CoreMiddleware) -> None:
        self.middlewares.append(middleware)

    def remove(self, middleware: CoreMiddleware) -> None:
        if middleware in self.middlewares:
            self.middlewares.remove(middleware)

    async def _attempt_recovery(
        self,
        ctx: MiddlewareCtx,
        error: Exception,
        log: MiddlewareLogger,
    ) -> t.AsyncGenerator[t.Any | CoreEvent, None]:

        for mw in reversed(self.middlewares):
            try:
                async for item in mw.on_error(ctx, error):
                    if isinstance(item, CoreEvent):
                        yield item
                    else:
                        log.recovery_success(mw)
                        yield item
                        return

            except Exception as e:
                log.recovery_error(mw, e)
                continue

        log.unrecoverable(error)
        raise error

    async def execute(
        self,
        action: str,
        ctx: RunContext,
        data: t.Any,
        func: t.Callable[[t.Any], t.Awaitable[t.Any]],
        metadata: dict[str, t.Any] | None = None,
    ) -> AsyncGenerator[t.Any | CoreEvent, None]:
        from ..loggers import MiddlewareLogger

        log = MiddlewareLogger(self.mw_logger, action, ctx.run_id)

        mw_ctx = MiddlewareCtx(
            ctx=ctx,
            data=data,
            action=action,
            metadata=metadata or {},
        )

        # -------- PRE-PROCESS -----------------------------------------------------------
        for mw in self.middlewares:
            try:
                next_ctx = None

                async for item in mw.on_request(mw_ctx):
                    if isinstance(item, MiddlewareCtx):
                        next_ctx = item

                    elif isinstance(item, CoreEvent):
                        yield item
                        if isinstance(item, ToolApprovalEvent):
                            return

                    else:
                        log.unexpected_yield(mw, item, "process_request")

                if next_ctx is None:
                    return

                mw_ctx = next_ctx

            except Exception as e:
                log.request_error(mw, e)
                async for item in self._attempt_recovery(mw_ctx, e, log):
                    yield item
                return

        # -------- CORE EXECUTION -----------------------------------------------------------
        try:
            result = await func(mw_ctx.data)
        except Exception as e:
            log.execution_error(e)
            async for item in self._attempt_recovery(mw_ctx, e, log):
                yield item
            return

        # -------- POST-PROCESS -----------------------------------------------------------
        for mw in reversed(self.middlewares):
            try:
                final = None

                async for item in mw.on_response(mw_ctx, result):
                    if isinstance(item, CoreEvent):
                        yield item
                    else:
                        final = item

                result = final if final is not None else result

            except Exception as e:
                log.response_error(mw, e)
                async for item in self._attempt_recovery(mw_ctx, e, log):
                    yield item
                return

        yield result

    async def execute_stream(
        self,
        action: str,
        ctx: RunContext,
        data: t.Any,
        stream_func: t.Callable[[t.Any], AsyncGenerator[t.Any, None]],
        metadata: dict[str, t.Any] | None = None,
    ) -> AsyncGenerator[t.Union[t.Any, CoreEvent], None]:

        log = MiddlewareLogger(self.mw_logger, action, ctx.run_id)

        mw_ctx = MiddlewareCtx(
            ctx=ctx,
            action=action,
            data=data,
            metadata=metadata or {},
        )

        # -------- PRE-PROCESS -----------------------------------------------------------
        for mw in self.middlewares:
            try:
                next_ctx = None

                async for item in mw.on_request(mw_ctx):
                    if isinstance(item, MiddlewareCtx):
                        next_ctx = item

                    elif isinstance(item, CoreEvent):
                        yield item
                        if isinstance(item, ToolApprovalEvent):
                            return

                if next_ctx is None:
                    return

                mw_ctx = next_ctx

            except Exception as e:
                log.request_error(mw, e)
                async for item in self._attempt_recovery(mw_ctx, e, log):
                    yield item
                return

        # -------- STREAMING -----------------------------------------------------------
        try:
            async for chunk in stream_func(mw_ctx.data):
                current = chunk
                dropped = False

                for mw in self.middlewares:
                    try:
                        transformed = None

                        async for item in mw.on_stream_chunk(mw_ctx, current):
                            if isinstance(item, CoreEvent):
                                yield item
                                if isinstance(item, ToolApprovalEvent):
                                    return
                            else:
                                transformed = item

                        if transformed is None:
                            dropped = True
                            break

                        current = transformed

                    except Exception as e:
                        log.stream_error(mw, e)
                        continue

                if not dropped:
                    yield current

        except Exception as e:
            log.execution_error(e)
            async for item in self._attempt_recovery(mw_ctx, e, log):
                yield item
            return

        # -------- POST-PROCESS -----------------------------------------------------------
        for mw in reversed(self.middlewares):
            try:
                async for item in mw.on_response(mw_ctx, None):
                    yield item
            except Exception as e:
                log.response_error(mw, e)
                async for item in self._attempt_recovery(mw_ctx, e, log):
                    yield item
                return
