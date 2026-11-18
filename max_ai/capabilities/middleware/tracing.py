"""OpenTelemetry spans for a turn, its model calls and its tool calls.

    invoke_agent billing                 one trace per turn (user.id, session.id)
    ├─ chat gpt-5                        model, tokens, cost, finish reason
    ├─ execute_tool get_weather          arguments, result, errors
    └─ chat gpt-5

Attributes follow the OpenTelemetry GenAI conventions (``gen_ai.*``), so any
backend understands them, plus the ``langfuse.*`` ones Langfuse uses to show
agents, generations and tools. Spans go to the global tracer provider: set
one up with ``configure_langfuse()``, ``logfire.configure()`` or any OTLP
exporter. Without a provider, spans are no-ops and cost nothing.

A turn that waits for a person (approval, question) ends its trace; the
next turn starts another with the same ``session.id``, so the backend
groups them.
"""

from __future__ import annotations

import base64
import json
import os
import typing as t
from collections.abc import AsyncGenerator

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
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
from ...core.event_type import ModelResponseEvent, ReasoningCompleteEvent
from ...errors.middleware import StopRun
from ...types.completions import ChatCompletionChunk, ChatCompletionResult, Usage

if t.TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from ...core.event_type import CoreEvent
    from ...core.messages import AssistantMessage
    from ...core.model.llm import ModelConfig
    from ...types.run_context import RunContext
    from ...types.tool_call import ToolCallRecord, ToolResult


class TracingConfig(MiddlewareConfig):
    """Configuration options for ``Tracing``."""
    capture_content: bool = Field(
        default=True, description="Record prompts, answers and tool data (may hold personal data).",
    )
    max_content_chars: int = Field(default=20_000, ge=100)


class TracingMiddleware(CoreMiddleware):
    """Trace what the agent does with OpenTelemetry."""

    component_provider_override = "max_ai.capabilities.middleware.TracingMiddleware"
    component_schema = TracingConfig

    def __init__(
        self,
        capture_content: bool = True,
        max_content_chars: int = 20_000,
        *,
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        self.capture_content = capture_content
        self.max_content_chars = max_content_chars
        self.agent = "agent"  # set by the harness: the name the traces carry
        # Runtime only (not serialized): default is the global provider.
        self._provider = tracer_provider
        # Live spans can't be stored in the RunContext: they stay in this
        # process, keyed by run, and end with the turn.
        self._turns: dict[str, Span] = {}

    # -------- TURN ---------------------------------------------------------------------
    async def turn(self, ctx: RunContext, call_next: NextTurn) -> AsyncGenerator[CoreEvent]:
        # 1. The turn's span: what the user said is waiting in the inbox.
        task = "\n".join(ctx.inbox) or None
        span = self._tracer.start_span(
            f"invoke_agent {self.agent}",
            kind=SpanKind.INTERNAL,
            attributes=_clean({
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": self.agent,
                "user.id": ctx.user_id,
                "session.id": ctx.session_id,
                "max_ai.run_id": ctx.run_id,
                "max_ai.resumed": task is None,
                "langfuse.trace.name": self.agent,
                "langfuse.observation.type": "agent",
                "langfuse.trace.input": self._content(task),
                "langfuse.observation.input": self._content(task),
            }),
        )
        self._turns[ctx.run_id] = span
        # 2. The turn, adding up what it used and seeing how it ends.
        usage, finish = Usage(), None
        try:
            async for event in call_next():
                if isinstance(event, ModelResponseEvent) and event.usage is not None:
                    usage = usage + event.usage
                elif isinstance(event, ReasoningCompleteEvent):
                    finish = event.finish_reason
                yield event
        except StopRun as error:
            # A middleware ended the turn: a warning with its message, not a failure.
            span.set_attributes(_clean({
                "langfuse.trace.output": self._content(error.message),
                "langfuse.observation.output": self._content(error.message),
                "max_ai.finish_reason": error.finish_reason,
                "langfuse.observation.level": "WARNING",
            }))
            raise
        except BaseException as error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, f"{type(error).__name__}: {error}"))
            raise
        else:
            # 3. The answer the user got.
            answer = self._answer(ctx)
            span.set_attributes(_clean({
                "langfuse.trace.output": self._content(answer),
                "langfuse.observation.output": self._content(answer),
                "max_ai.finish_reason": finish,
                "gen_ai.usage.input_tokens": usage.tokens_input,
                "gen_ai.usage.output_tokens": usage.tokens_output,
            }))
            if finish not in (None, "stop", "waiting_for_user"):
                span.set_attributes({"langfuse.observation.level": "WARNING"})
        finally:
            self._turns.pop(ctx.run_id, None)
            span.end()

    # -------- MODEL --------------------------------------------------------------------
    async def model(self, ctx: RunContext, call: ModelCall, call_next: NextModel) -> ChatCompletionResult:
        span = self._model_span(ctx, call)
        try:
            result = await call_next(call)
        except BaseException as error:
            self._failed(span, error)
            raise
        self._answered(span, call.model_config, result.message, result.usage, result.model, result.finish_reason)
        return result

    async def stream(
        self, ctx: RunContext, call: ModelCall, call_next: NextStream
    ) -> AsyncGenerator[ChatCompletionChunk]:
        span, ended = self._model_span(ctx, call), False
        try:
            async for chunk in call_next(call):
                # The last chunk brings the whole message and what it used.
                if chunk.is_complete and chunk.message is not None:
                    self._answered(
                        span, call.model_config, chunk.message, chunk.usage or Usage(),
                        call.model, chunk.finish_reason,
                    )
                    ended = True
                yield chunk
        except BaseException as error:
            if not ended:
                self._failed(span, error)
            raise
        if not ended:  # the stream stopped without its last chunk
            span.end()

    # -------- TOOLS --------------------------------------------------------------------
    async def tool(self, ctx: RunContext, record: ToolCallRecord, call_next: NextTool) -> ToolResult | None:
        span = self._child(ctx, f"execute_tool {record.tool_name}", {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": record.tool_name,
            "gen_ai.tool.call.id": record.id,
            "gen_ai.tool.call.arguments": self._content(record.parameters),
            "max_ai.tool.status": record.status.value,
            "langfuse.observation.type": "tool",
        })
        try:
            result = await call_next(record)
        except BaseException as error:
            self._failed(span, error)
            span.end()
            raise
        # 1. None: it waits for a person; the next turn traces it again.
        if result is None:
            span.set_attributes({"gen_ai.tool.call.result": "waiting for the user"})
        elif result.success:
            span.set_attributes(_clean({"gen_ai.tool.call.result": self._content(result.result)}))
        else:
            span.set_status(Status(StatusCode.ERROR, result.error or "tool failed"))
            span.set_attributes({"langfuse.observation.level": "ERROR"})
        span.end()
        return result

    # -------- HELPERS ------------------------------------------------------------------
    @property
    def _tracer(self) -> trace.Tracer:
        return trace.get_tracer("max_ai", tracer_provider=self._provider)

    def _content(self, value: t.Any) -> str | None:
        if not self.capture_content or value is None:
            return None
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        return text if len(text) <= self.max_content_chars else text[: self.max_content_chars] + "…"

    def _child(self, ctx: RunContext, name: str, attributes: dict[str, t.Any]) -> Span:
        """A span under the turn's; on every span the user and the session, so
        per-observation filters find them."""
        parent = self._turns.get(ctx.run_id)
        context = trace.set_span_in_context(parent) if parent is not None else None
        scope = {"user.id": ctx.user_id, "session.id": ctx.session_id}
        return self._tracer.start_span(
            name, context=context, kind=SpanKind.CLIENT, attributes=_clean({**attributes, **scope}),
        )

    def _model_span(self, ctx: RunContext, call: ModelCall) -> Span:
        messages = [{"role": m.role, "content": m.text()} for m in call.messages]
        return self._child(ctx, f"chat {call.model}", {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": call.model,
            "gen_ai.request.tools": len(call.tools) or None,
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": self._content(messages),
        })

    def _answered(
        self,
        span: Span,
        config: ModelConfig | None,
        message: AssistantMessage,
        usage: Usage,
        model: str,
        finish_reason: str | None,
    ) -> None:
        """What the model said and what it used, on its span; then it ends."""
        output: t.Any = message.text()
        if message.tool_calls:
            calls = [{"name": c.tool_name, "arguments": c.parameters} for c in message.tool_calls]
            output = {"content": output, "tool_calls": calls} if output else {"tool_calls": calls}
        span.set_attributes(_clean({
            "gen_ai.response.model": model,
            "gen_ai.response.finish_reasons": [finish_reason] if finish_reason else None,
            "gen_ai.usage.input_tokens": usage.tokens_input,
            "gen_ai.usage.output_tokens": usage.tokens_output,
            "langfuse.observation.output": self._content(output),
            "langfuse.observation.cost_details": _cost(config, usage.tokens_input, usage.tokens_output),
        }))
        span.end()

    @staticmethod
    def _failed(span: Span, error: BaseException) -> None:
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, f"{type(error).__name__}: {error}"))
        span.end()

    @staticmethod
    def _answer(ctx: RunContext) -> t.Any:
        """The last thing the agent said, in its structured form when it has one."""
        from ...core.messages import AssistantMessage

        last = next((m for m in reversed(ctx.messages) if isinstance(m, AssistantMessage)), None)
        if last is None:
            return None
        return last.structured_output.model_dump() if last.structured_output else last.text()


def _clean(attributes: dict[str, t.Any]) -> dict[str, t.Any]:
    """OpenTelemetry attributes can't be None."""
    return {key: value for key, value in attributes.items() if value is not None}


def _cost(config: ModelConfig | None, tokens_in: int | None, tokens_out: int | None) -> str | None:
    if config is None or config.input_cost_per_mtok is None or config.output_cost_per_mtok is None:
        return None
    cost_in = (tokens_in or 0) * config.input_cost_per_mtok / 1_000_000
    cost_out = (tokens_out or 0) * config.output_cost_per_mtok / 1_000_000
    return json.dumps({"input": cost_in, "output": cost_out, "total": cost_in + cost_out})


def configure_langfuse(
    *,
    public_key_env: str = "LANGFUSE_PUBLIC_KEY",
    secret_key_env: str = "LANGFUSE_SECRET_KEY",
    host_env: str = "LANGFUSE_HOST",
    set_global: bool = True,
) -> TracerProvider:
    """Send spans to Langfuse over OTLP/HTTP; keys come from env vars.

    Returns the provider: call ``provider.force_flush()`` before a serverless
    function returns (it may be frozen with spans still buffered) and
    ``provider.shutdown()`` when the process exits.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    public, secret = os.getenv(public_key_env), os.getenv(secret_key_env)
    if not public or not secret:
        raise ValueError(f"Set {public_key_env} and {secret_key_env} to send traces to Langfuse")
    host = (os.getenv(host_env) or "https://cloud.langfuse.com").rstrip("/")
    token = base64.b64encode(f"{public}:{secret}".encode()).decode()
    exporter = OTLPSpanExporter(
        endpoint=f"{host}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {token}", "x-langfuse-ingestion-version": "4"},
    )
    provider = TracerProvider(resource=Resource.create({"service.name": "max_ai"}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    if set_global:
        trace.set_tracer_provider(provider)
    return provider


__all__ = ["TracingConfig", "TracingMiddleware", "configure_langfuse"]
