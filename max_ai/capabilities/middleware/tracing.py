"""OpenTelemetry spans for a run, its model calls and its tool calls.

    invoke_agent billing                 one trace per run (user.id, session.id)
    ├─ chat gpt-5                        model, tokens, cost, finish reason
    ├─ execute_tool get_weather          arguments, result, errors
    └─ chat gpt-5

Attributes follow the OpenTelemetry GenAI conventions (``gen_ai.*``), so any
backend understands them, plus the ``langfuse.*`` ones Langfuse uses to show
agents, generations and tools. Spans go to the global tracer provider: set
one up with ``configure_langfuse()``, ``logfire.configure()`` or any OTLP
exporter. Without a provider, spans are no-ops and cost nothing.

A paused run (approval, question) ends its trace; the resume starts another
with the same ``session.id``, so the backend groups them.
"""

from __future__ import annotations

import base64
import json
import os
import typing as t

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from pydantic import Field

from ...base.middleware import (
    CoreMiddleware,
    MiddlewareConfig,
    MiddlewareContext,
    ModelRequest,
    ToolRequest,
)

if t.TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from ...core.messages import AssistantMessage, CoreMessage
    from ...types.agent_response import AgentResponse
    from ...types.completions import ChatCompletionResult
    from ...types.tool_call import ToolResult

_SPAN = "trace_span"


class TracingConfig(MiddlewareConfig):
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
        # Runtime only (not serialized): default is the global provider.
        self._provider = tracer_provider
        # Live spans can't be stored in the RunContext: they stay in this
        # process, keyed by run, and end with the run.
        self._runs: dict[str, Span] = {}

    @property
    def _tracer(self) -> trace.Tracer:
        return trace.get_tracer("max_ai", tracer_provider=self._provider)

    def _content(self, value: t.Any) -> str | None:
        if not self.capture_content or value is None:
            return None
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        return text if len(text) <= self.max_content_chars else text[: self.max_content_chars] + "…"

    def _child(self, mw: MiddlewareContext, name: str, attributes: dict[str, t.Any]) -> Span:
        parent = self._runs.get(mw.ctx.run_id)
        context = trace.set_span_in_context(parent) if parent is not None else None
        # On every span, so per-observation filters (user, session) find them.
        scope = {"user.id": mw.ctx.user_id, "session.id": mw.ctx.session_id}
        return self._tracer.start_span(
            name, context=context, kind=SpanKind.CLIENT, attributes=_clean({**attributes, **scope}),
        )

    # -------- RUN -----------------------------------------------------------
    async def on_run_start(self, mw: MiddlewareContext, task: list[CoreMessage] | None) -> None:
        ctx = mw.ctx
        task_text = "\n".join(m.text() for m in task) if task else None
        self._runs[ctx.run_id] = self._tracer.start_span(
            f"invoke_agent {mw.agent}",
            kind=SpanKind.INTERNAL,
            attributes=_clean({
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": mw.agent,
                "user.id": ctx.user_id,
                "session.id": ctx.session_id,
                "max_ai.run_id": ctx.run_id,
                "max_ai.resumed": task is None,
                "langfuse.trace.name": mw.agent,
                "langfuse.observation.type": "agent",
                "langfuse.trace.input": self._content(task_text),
                "langfuse.observation.input": self._content(task_text),
            }),
        )

    async def on_final_response(self, mw: MiddlewareContext, message: AssistantMessage) -> AssistantMessage:
        span = self._runs.get(mw.ctx.run_id)
        if span is not None:
            answer = message.structured_output.model_dump() if message.structured_output else message.text()
            span.set_attributes(_clean({
                "langfuse.trace.output": self._content(answer),
                "langfuse.observation.output": self._content(answer),
            }))
        return message

    async def on_run_end(self, mw: MiddlewareContext, response: AgentResponse) -> None:
        span = self._runs.pop(mw.ctx.run_id, None)
        if span is None:
            return
        usage = response.usage
        span.set_attributes(_clean({
            "max_ai.finish_reason": response.finish_reason,
            "max_ai.stop_message": response.stop_message,
            "gen_ai.usage.input_tokens": usage.tokens_input,
            "gen_ai.usage.output_tokens": usage.tokens_output,
            "max_ai.llm_calls": usage.llm_calls,
            "max_ai.tool_calls": usage.tool_calls,
        }))
        if response.finish_reason in ("error", "budget_exceeded", "output_limit", "stopped"):
            span.set_attributes({"langfuse.observation.level": "WARNING"})
        span.end()

    async def on_run_error(self, mw: MiddlewareContext, error: BaseException) -> None:
        span = self._runs.pop(mw.ctx.run_id, None)
        if span is not None:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, f"{type(error).__name__}: {error}"))
            span.end()

    # -------- MODEL -----------------------------------------------------------
    async def on_model_request(self, mw: MiddlewareContext, request: ModelRequest) -> ModelRequest:
        messages = [{"role": m.role, "content": m.text()} for m in request.messages]
        options = {k: v for k, v in request.options.items() if isinstance(v, (str, int, float, bool))}
        request.metadata[_SPAN] = self._child(mw, f"chat {request.model}", {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request.model,
            "gen_ai.request.tools": len(request.tools) or None,
            **{f"gen_ai.request.{key}": value for key, value in options.items()},
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": self._content(messages),
        })
        return request

    async def on_model_response(
        self, mw: MiddlewareContext, request: ModelRequest, result: ChatCompletionResult,
    ) -> ChatCompletionResult:
        span: Span | None = request.metadata.pop(_SPAN, None)
        if span is None:
            return result
        message, usage = result.message, result.usage
        output: t.Any = message.text()
        if message.tool_calls:
            calls = [{"name": c.tool_name, "arguments": c.parameters} for c in message.tool_calls]
            output = {"content": output, "tool_calls": calls} if output else {"tool_calls": calls}
        span.set_attributes(_clean({
            "gen_ai.response.model": result.model,
            "gen_ai.response.finish_reasons": [result.finish_reason] if result.finish_reason else None,
            "gen_ai.usage.input_tokens": usage.tokens_input,
            "gen_ai.usage.output_tokens": usage.tokens_output,
            "langfuse.observation.output": self._content(output),
            "langfuse.observation.cost_details": _cost(request, usage.tokens_input, usage.tokens_output),
        }))
        span.end()
        return result

    async def on_model_error(self, mw: MiddlewareContext, request: ModelRequest, error: Exception) -> None:
        span: Span | None = request.metadata.pop(_SPAN, None)
        if span is not None:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()
        return None

    # -------- TOOLS -----------------------------------------------------------
    async def on_tool_request(self, mw: MiddlewareContext, request: ToolRequest) -> None:
        request.metadata[_SPAN] = self._child(mw, f"execute_tool {request.tool_name}", {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": request.tool_name,
            "gen_ai.tool.call.id": request.record.id,
            "gen_ai.tool.call.arguments": self._content(request.parameters),
            "langfuse.observation.type": "tool",
        })
        return None

    async def on_tool_response(
        self, mw: MiddlewareContext, request: ToolRequest, result: ToolResult,
    ) -> ToolResult:
        span: Span | None = request.metadata.pop(_SPAN, None)
        if span is None:
            return result
        if result.success:
            span.set_attributes(_clean({"gen_ai.tool.call.result": self._content(result.result)}))
        else:
            span.set_status(Status(StatusCode.ERROR, result.error or "tool failed"))
            span.set_attributes({"langfuse.observation.level": "ERROR"})
        span.end()
        return result


def _clean(attributes: dict[str, t.Any]) -> dict[str, t.Any]:
    """OpenTelemetry attributes can't be None."""
    return {key: value for key, value in attributes.items() if value is not None}


def _cost(request: ModelRequest, tokens_in: int | None, tokens_out: int | None) -> str | None:
    config = request.model_config
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
