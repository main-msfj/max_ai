"""Middleware: intercept an agent's run, its model calls and its tool calls.

A middleware is a serializable component with optional hooks; implement
only the ones you need. Requests go through the middlewares in order and
responses in reverse, like layers of an onion::

    on_run_start ─┐
                  ├─ on_model_request → model → on_model_chunk* → on_model_response
                  │                              (on_model_error if it fails)
                  ├─ on_tool_request  → tool  → on_tool_response
                  ├─ on_final_response        (the answer the gates accepted)
    on_run_end ───┘   (on_run_error instead if the run crashes or is cancelled)

Differences with the other extension points: EventBus handlers only
observe, completion gates decide whether a turn is done, middleware can
change or block what happens.

One Agent serves every user, so a middleware never keeps run state on
``self``: use ``mw.state(self)``, which lives in the ``RunContext`` and
survives a pause/resume.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from pydantic import BaseModel

from .component import ComponentBase

if t.TYPE_CHECKING:
    from ..core.event_type import CoreEvent
    from ..core.messages import AssistantMessage, CoreMessage
    from ..core.model.llm import ModelConfig
    from ..types.agent_response import AgentResponse
    from ..types.completions import ChatCompletionChunk, ChatCompletionResult
    from ..types.run_context import RunContext
    from ..types.tool_call import ToolCallRecord, ToolResult


class MiddlewareConfig(BaseModel):
    """A middleware's settings. Middlewares with options extend it."""


class StopRun(Exception):
    """Raise from ``on_run_start`` or a model hook to end the turn cleanly:
    the run returns ``finish_reason`` and ``message`` tells the user why.
    Tool hooks block a single call by returning a ``ToolResult`` instead,
    so the transcript never keeps a tool call without its result."""

    def __init__(self, message: str, finish_reason: str = "stopped") -> None:
        """
        Initialize a run-stop signal with its user-facing reason.

        Parameters
        ----------
        message : str
            Value used to initialize the run-stop signal.
        finish_reason : str, default='stopped'
            Value used to initialize the run-stop signal.
        """
        super().__init__(message)
        self.message = message
        self.finish_reason = finish_reason


@dataclass
class MiddlewareContext:
    """What every hook receives: the run and a way to emit events."""

    ctx: RunContext
    agent: str
    emit: t.Callable[[CoreEvent], None] = lambda event: None

    def state(self, middleware: CoreMiddleware) -> dict[str, t.Any]:
        """This middleware's state for the current run (kept in the RunContext)."""
        store = self.ctx.runtime_state.shared_state.setdefault("middleware", {})
        return store.setdefault(middleware.state_key, {})


@dataclass
class ModelRequest:
    """One model call. Hooks may change ``tools`` and ``options`` (provider
    kwargs such as ``temperature``); ``metadata`` carries data from the
    request hook to the response hook of the same call."""

    messages: list[CoreMessage]
    tools: list[t.Any]
    options: dict[str, t.Any]
    output_format: type[BaseModel] | None = None
    stream: bool = False
    model: str = "unknown"
    model_config: ModelConfig | None = None  # capabilities and prices
    metadata: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class ToolRequest:
    """One approved tool call, about to run."""

    record: ToolCallRecord
    metadata: dict[str, t.Any] = field(default_factory=dict)

    @property
    def tool_name(self) -> str:
        """
        Return the name of the tool represented by this request.

        Returns
        -------
        str
            The resulting text value.
        """
        return self.record.tool_name

    @property
    def parameters(self) -> dict[str, t.Any]:
        """
        Return the arguments supplied to this tool call.

        Returns
        -------
        dict[str, t.Any]
            The arguments supplied to the tool call.
        """
        return self.record.parameters


class CoreMiddleware(ComponentBase[MiddlewareConfig]):
    """Override only the hooks you need; the defaults change nothing.

    Serialization is generic: declare a ``component_schema`` whose fields
    match the constructor's arguments (stored on ``self`` under the same
    names) and ``serialize()``/``deserialize()`` work with no extra code.
    """

    component_type = "middleware"
    component_schema: t.ClassVar[type[BaseModel]] = MiddlewareConfig

    @property
    def state_key(self) -> str:
        """Key of this middleware's run state. Override if you register two
        instances of the same class."""
        return type(self).__name__

    # -------- RUN -----------------------------------------------------------
    async def on_run_start(
        self, mw: MiddlewareContext, task: list[CoreMessage] | None,
    ) -> None:
        """A run starts. ``task`` is the new input, ``None`` on a resume."""

    async def on_final_response(
        self, mw: MiddlewareContext, message: AssistantMessage,
    ) -> AssistantMessage:
        """The answer the gates accepted; return it, changed or not."""
        return message

    async def on_run_end(self, mw: MiddlewareContext, response: AgentResponse) -> None:
        """The run ended (finished, paused or stopped)."""

    async def on_run_error(self, mw: MiddlewareContext, error: BaseException) -> None:
        """The run crashed or was cancelled; the error is raised after this."""

    # -------- MODEL -----------------------------------------------------------
    async def on_model_request(self, mw: MiddlewareContext, request: ModelRequest) -> ModelRequest:
        """
        Inspect or modify a model request before it is sent.

        Parameters
        ----------
        mw : MiddlewareContext
            Middleware context for the active run.
        request : ModelRequest
            Request being passed through the middleware hook.

        Returns
        -------
        ModelRequest
            The request to send to the model.
        """
        return request

    async def on_model_chunk(
        self, mw: MiddlewareContext, request: ModelRequest, chunk: ChatCompletionChunk,
    ) -> ChatCompletionChunk | None:
        """A streamed chunk; ``None`` drops it."""
        return chunk

    async def on_model_response(
        self, mw: MiddlewareContext, request: ModelRequest, result: ChatCompletionResult,
    ) -> ChatCompletionResult:
        """
        Inspect or modify a completed model response.

        Parameters
        ----------
        mw : MiddlewareContext
            Middleware context for the active run.
        request : ModelRequest
            Request being passed through the middleware hook.
        result : ChatCompletionResult
            Result produced by the preceding operation.

        Returns
        -------
        ChatCompletionResult
            The normalized model response.
        """
        return result

    async def on_model_error(
        self, mw: MiddlewareContext, request: ModelRequest, error: Exception,
    ) -> ChatCompletionResult | None:
        """The call failed after its retries. Return a result to recover
        (e.g. from a fallback model) or ``None`` to let the error through."""
        return None

    # -------- TOOLS -----------------------------------------------------------
    async def on_tool_request(
        self, mw: MiddlewareContext, request: ToolRequest,
    ) -> ToolResult | None:
        """Before an approved tool runs. Return a ``ToolResult`` to answer
        in its place (the tool does not run), ``None`` to let it run."""
        return None

    async def on_tool_response(
        self, mw: MiddlewareContext, request: ToolRequest, result: ToolResult,
    ) -> ToolResult:
        """
        Inspect or modify the result returned by a tool.

        Parameters
        ----------
        mw : MiddlewareContext
            Middleware context for the active run.
        request : ToolRequest
            Request being passed through the middleware hook.
        result : ToolResult
            Result produced by the preceding operation.

        Returns
        -------
        ToolResult
            The result to return from the tool call.
        """
        return result

    # -------- SERIALIZATION -----------------------------------------------------------
    def _to_config(self) -> BaseModel:
        """
        Return the middleware settings for serialization.

        Returns
        -------
        BaseModel
            The middleware configuration model.
        """
        schema = self.component_schema
        return schema(**{name: getattr(self, name) for name in schema.model_fields})

    @classmethod
    def _from_config(cls, config: BaseModel) -> t.Self:
        """
        Create middleware from its validated configuration.

        Parameters
        ----------
        config : BaseModel
            Model or component configuration.

        Returns
        -------
        t.Self
            The middleware instance reconstructed from its configuration.
        """
        return cls(**config.model_dump())


__all__ = [
    "CoreMiddleware",
    "MiddlewareConfig",
    "MiddlewareContext",
    "ModelRequest",
    "StopRun",
    "ToolRequest",
]
