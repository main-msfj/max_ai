"""
Middleware system for maxais.

Provides a composable async middleware pipeline for intercepting and
modifying agent execution flows such as model calls, tool execution,
and memory access.

The system is based on async generators to enable:
- Event emission (observability / tracing)
- Execution control (approval / interruption)
- Streaming transformations
- Error recovery via fallback middleware
"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from ..core.event_type import CoreEvent
from ..types.middleware import MiddlewareCtx


class CoreMiddleware(ABC):
    """
    Base class for execution middleware.

    Middleware can observe and modify:
    - Incoming requests
    - Outgoing responses
    - Streaming chunks
    - Error flows
    """

    @abstractmethod
    async def on_request(
        self, ctx: MiddlewareCtx
    ) -> AsyncGenerator[MiddlewareCtx | CoreEvent, None]:
        """
        Executed before the main action.

        Yields:
            - MiddlewareCtx (final state)
            - CoreEvent (observability signals)
        """
        yield ctx

    @abstractmethod
    async def on_response(
        self, ctx: MiddlewareCtx, result: t.Any
    ) -> AsyncGenerator[t.Any | CoreEvent, None]:
        """
        Executed after successful action completion.

        Yields:
            - Modified result or original result
            - CoreEvent signals
        """
        yield result

    @abstractmethod
    async def on_error(
        self, ctx: MiddlewareCtx, error: Exception
    ) -> AsyncGenerator[t.Any | CoreEvent, None]:
        """
        Executed when an error occurs.

        Yields:
            - Recovery value (optional)
            - CoreEvent signals
        """
        raise error

    @abstractmethod
    async def on_stream_chunk(
        self, ctx: MiddlewareCtx, chunk: t.Any
    ) -> AsyncGenerator[t.Any | CoreEvent, None]:
        """
        Executed for each streaming chunk.

        Yields:
            - Transformed chunk
            - CoreEvent signals
        """
        yield chunk
