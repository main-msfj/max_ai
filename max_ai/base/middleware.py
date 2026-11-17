"""Middleware: a list that sees everything the agent does and may change its text.

Each middleware wraps one place and calls ``call_next`` to go on:

    turn    every event of a turn                Agent._turn
    model   one model call, whole                CoreChatCompletionClient.run
    stream  one model call, line by line         CoreChatCompletionClient.stream
    tool    every tool call, whatever happens    ToolDispatcher.dispatch

The list runs in order on the way in and in reverse on the way out. A
middleware may change text (mask it, cut it) or end the turn with
``StopRun``; it never changes a tool call's parameters. Hooks stop and
correct; middlewares watch and filter.
"""

from __future__ import annotations

import typing as t
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel, JsonValue

from .component import ComponentBase

if t.TYPE_CHECKING:
    from ..core.event_type import CoreEvent
    from ..core.messages import Message
    from ..core.model.llm import ModelConfig
    from ..types.completions import ChatCompletionChunk, ChatCompletionResult
    from ..types.run_context import RunContext
    from ..types.tool_call import ToolCallRecord, ToolResult
    from ..types.tools import CoreToolDefinition


class MiddlewareConfig(BaseModel):
    """A middleware's settings. Middlewares with options extend it."""


@dataclass
class ModelCall:
    """One model call: what the model is about to see."""

    messages: list[Message]
    tools: list[CoreToolDefinition]
    output_format: type[BaseModel] | None
    model: str
    model_config: ModelConfig | None = None  # its window and prices, to read


NextTurn = Callable[[], AsyncGenerator["CoreEvent"]]
NextModel = Callable[[ModelCall], Awaitable["ChatCompletionResult"]]
NextStream = Callable[[ModelCall], AsyncGenerator["ChatCompletionChunk"]]
NextTool = Callable[["ToolCallRecord"], Awaitable["ToolResult | None"]]


class CoreMiddleware(ComponentBase[MiddlewareConfig]):
    """Write only the places you need; the others let everything pass.

    One Agent serves every user: keep what a turn needs in
    ``self.state(ctx)``, never on ``self``.

    Serialization is generic: declare a ``component_schema`` whose fields
    match the constructor's arguments (stored on ``self`` under the same
    names) and ``serialize()``/``deserialize()`` work with no extra code.
    """

    component_type = "middleware"
    component_schema: t.ClassVar[type[BaseModel]] = MiddlewareConfig

    async def turn(self, ctx: RunContext, call_next: NextTurn) -> AsyncGenerator[CoreEvent]:
        """Every event of a turn, from the task to the end."""
        async for event in call_next():
            yield event

    async def model(
        self, ctx: RunContext, call: ModelCall, call_next: NextModel
    ) -> ChatCompletionResult:
        """One model call: what goes in (``call``) and what comes back."""
        return await call_next(call)

    async def stream(
        self, ctx: RunContext, call: ModelCall, call_next: NextStream
    ) -> AsyncGenerator[ChatCompletionChunk]:
        """One streamed call. Each chunk is a whole line; the last one
        carries the full message."""
        async for chunk in call_next(call):
            yield chunk

    async def tool(
        self, ctx: RunContext, record: ToolCallRecord, call_next: NextTool
    ) -> ToolResult | None:
        """Every tool call: allowed, denied, waiting or failed. ``record``
        says which; None means it waits for a person."""
        return await call_next(record)

    def state(self, ctx: RunContext) -> dict[str, JsonValue]:
        """This middleware's own place in ``ctx``: it survives pause and resume."""
        return ctx.middleware_state.setdefault(type(self).__name__, {})

    # -------- SERIALIZATION -----------------------------------------------------------
    def _to_config(self) -> MiddlewareConfig:
        """The middleware's settings, read from the attributes of the same name."""
        schema = self.component_schema
        # A child's schema extends MiddlewareConfig (see BudgetConfig).
        return t.cast(MiddlewareConfig, schema(**{name: getattr(self, name) for name in schema.model_fields}))

    @classmethod
    def _from_config(cls, config: MiddlewareConfig) -> t.Self:
        """A middleware built back from its settings."""
        return cls(**config.model_dump())


__all__ = [
    "CoreMiddleware",
    "MiddlewareConfig",
    "ModelCall",
    "NextModel",
    "NextStream",
    "NextTool",
    "NextTurn",
]
