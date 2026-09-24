"""Run an agent's middlewares around each step: requests in order,
responses in reverse."""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from ...base.middleware import (
        CoreMiddleware,
        MiddlewareContext,
        ModelRequest,
        ToolRequest,
    )
    from ...core.messages import AssistantMessage, CoreMessage
    from ...types.agent_response import AgentResponse
    from ...types.completions import ChatCompletionChunk, ChatCompletionResult
    from ...types.tool_call import ToolResult


class MiddlewareChain:
    def __init__(self, middlewares: t.Sequence[CoreMiddleware] = ()) -> None:
        self.middlewares = list(middlewares)

    def __bool__(self) -> bool:
        return bool(self.middlewares)

    async def run_start(
        self, mw: MiddlewareContext, task: list[CoreMessage] | None
    ) -> None:
        for middleware in self.middlewares:
            await middleware.on_run_start(mw, task)

    async def run_end(self, mw: MiddlewareContext, response: AgentResponse) -> None:
        for middleware in reversed(self.middlewares):
            await middleware.on_run_end(mw, response)

    async def run_error(self, mw: MiddlewareContext, error: BaseException) -> None:
        """Every middleware hears about it; one failing doesn't hide the error."""
        for middleware in reversed(self.middlewares):
            try:
                await middleware.on_run_error(mw, error)
            except Exception:  # noqa: BLE001 - the run's own error must win
                continue

    async def final_response(
        self,
        mw: MiddlewareContext,
        message: AssistantMessage,
    ) -> AssistantMessage:
        for middleware in reversed(self.middlewares):
            message = await middleware.on_final_response(mw, message)
        return message

    async def model_request(
        self, mw: MiddlewareContext, request: ModelRequest
    ) -> ModelRequest:
        for middleware in self.middlewares:
            request = await middleware.on_model_request(mw, request)
        return request

    async def model_chunk(
        self,
        mw: MiddlewareContext,
        request: ModelRequest,
        chunk: ChatCompletionChunk,
    ) -> ChatCompletionChunk | None:
        for middleware in reversed(self.middlewares):
            if (chunk := await middleware.on_model_chunk(mw, request, chunk)) is None:
                return None
        return chunk

    async def model_response(
        self,
        mw: MiddlewareContext,
        request: ModelRequest,
        result: ChatCompletionResult,
    ) -> ChatCompletionResult:
        for middleware in reversed(self.middlewares):
            result = await middleware.on_model_response(mw, request, result)
        return result

    async def model_error(
        self,
        mw: MiddlewareContext,
        request: ModelRequest,
        error: Exception,
    ) -> ChatCompletionResult | None:
        """The first middleware (innermost first) that recovers wins."""
        for middleware in reversed(self.middlewares):
            if (
                result := await middleware.on_model_error(mw, request, error)
            ) is not None:
                return result
        return None

    async def tool_request(
        self, mw: MiddlewareContext, request: ToolRequest
    ) -> ToolResult | None:
        for middleware in self.middlewares:
            if (result := await middleware.on_tool_request(mw, request)) is not None:
                return result
        return None

    async def tool_response(
        self,
        mw: MiddlewareContext,
        request: ToolRequest,
        result: ToolResult,
    ) -> ToolResult:
        for middleware in reversed(self.middlewares):
            result = await middleware.on_tool_response(mw, request, result)
        return result
