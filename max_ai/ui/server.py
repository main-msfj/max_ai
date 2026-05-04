"""Single-session Web UI for a pre-built MaxAI agent."""

from __future__ import annotations

import asyncio
import json
import typing as t
from pathlib import Path

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from max_ai.base.agent import Agent
from max_ai.core.event_type import (
    CompactionEvent,
    CoreEvent,
    ErrorEvent,
    ModelResponseEvent,
    ModelStreamChunkEvent,
    ReasoningCompleteEvent,
    ReasoningIterationEvent,
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)
from max_ai.types.agent_response import AgentResponse
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord


STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ApprovalDecision(BaseModel):
    tool_call_id: str
    approved: bool
    reason: str | None = None


class ApprovalRequest(BaseModel):
    decisions: list[ApprovalDecision] = Field(default_factory=list)


def create_app(agent: Agent) -> FastAPI:
    """Create a FastAPI app that renders and drives one agent session."""

    app = FastAPI(title="MaxAI WebUI")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.agent = agent
    app.state.ctx = RunContext()
    app.state.turn_lock = asyncio.Lock()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/info")
    async def info() -> dict[str, t.Any]:
        return _agent_info(app.state.agent)

    @app.post("/api/clear")
    async def clear() -> dict[str, str]:
        app.state.ctx = RunContext()
        return {"status": "ok"}

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        async def stream() -> t.AsyncIterator[str]:
            async for payload in _run_turn(app, req.message):
                yield _sse(payload)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/approve")
    async def approve(req: ApprovalRequest) -> StreamingResponse:
        async def stream() -> t.AsyncIterator[str]:
            async for payload in _resume_turn(app, req.decisions):
                yield _sse(payload)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def serve(
    agent: Agent,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the single-session WebUI for ``agent``."""

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "uvicorn is required to serve the MaxAI WebUI. "
            "Install the ui dependency group or add uvicorn."
        ) from exc

    uvicorn.run(create_app(agent), host=host, port=port, reload=reload)


server = serve


async def _run_turn(app: FastAPI, message: str) -> t.AsyncIterator[dict[str, t.Any]]:
    lock: asyncio.Lock = app.state.turn_lock
    if lock.locked():
        yield {"type": "error", "message": "Another turn is already running."}
        return

    async with lock:
        agent: Agent = app.state.agent
        ctx: RunContext = app.state.ctx
        yield {"type": "status", "status": "running"}
        try:
            async for payload in _stream_agent_events(
                agent.run_stream_events(
                    task=message,
                    run_context=ctx,
                    stream_tokens=True,
                )
            ):
                if payload.get("type") == "done" and payload.get("context") is not None:
                    app.state.ctx = payload.pop("context")
                yield payload
        except Exception as exc:  # noqa: BLE001
            yield {
                "type": "error",
                "message": str(exc),
                "error_type": type(exc).__name__,
            }


async def _resume_turn(
    app: FastAPI,
    decisions: list[ApprovalDecision],
) -> t.AsyncIterator[dict[str, t.Any]]:
    lock: asyncio.Lock = app.state.turn_lock
    if lock.locked():
        yield {"type": "error", "message": "Another turn is already running."}
        return

    async with lock:
        agent: Agent = app.state.agent
        ctx: RunContext = app.state.ctx
        try:
            for decision in decisions:
                ctx.tool_state.apply_approval(
                    decision.tool_call_id,
                    approved=decision.approved,
                    reason=decision.reason,
                )

            yield {"type": "status", "status": "resuming"}
            async for payload in _stream_agent_events(
                agent.resume_stream_events(run_context=ctx, stream_tokens=True)
            ):
                if payload.get("type") == "done" and payload.get("context") is not None:
                    app.state.ctx = payload.pop("context")
                yield payload
        except Exception as exc:  # noqa: BLE001
            yield {
                "type": "error",
                "message": str(exc),
                "error_type": type(exc).__name__,
            }


async def _stream_agent_events(
    stream: t.AsyncIterator[CoreEvent | AgentResponse | None],
) -> t.AsyncIterator[dict[str, t.Any]]:
    async for item in stream:
        if item is None:
            continue

        if isinstance(item, AgentResponse):
            yield {
                "type": "done",
                "finish_reason": item.finish_reason,
                "needs_approval": item.needs_approval,
                "pending_approvals": [
                    _approval_record(record) for record in item.pending_approvals
                ],
                "usage": jsonable_encoder(item.usage),
                "context": item.context,
            }
            continue

        yield _event_payload(item)


def _event_payload(event: CoreEvent) -> dict[str, t.Any]:
    if isinstance(event, ReasoningIterationEvent):
        return {
            "type": "reasoning",
            "phase": "iteration",
            "iteration": event.iteration,
            "max_iterations": event.max_iterations,
        }

    if isinstance(event, ReasoningCompleteEvent):
        return {
            "type": "reasoning",
            "phase": "complete",
            "finish_reason": event.finish_reason,
            "total_iterations": event.total_iterations,
        }

    if isinstance(event, CompactionEvent):
        return {
            "type": "compaction",
            "strategy": event.strategy,
            "old_message_count": event.old_message_count,
            "recent_message_count": event.recent_message_count,
            "old_token_count": event.old_token_count,
            "recent_token_count": event.recent_token_count,
            "total_token_count": event.total_token_count,
            "max_history_tokens": event.max_history_tokens,
        }

    if isinstance(event, ModelStreamChunkEvent):
        return {
            "type": "token",
            "text": event.chunk,
            "is_final": event.is_final,
        }

    if isinstance(event, ModelResponseEvent):
        return {
            "type": "assistant_text",
            "text": event.response,
            "has_tool_calls": event.has_tool_calls,
        }

    if isinstance(event, ToolCallEvent):
        return {
            "type": "tool_call",
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "parameters": jsonable_encoder(event.parameters),
        }

    if isinstance(event, ToolCallResponseEvent):
        result = event.tool_result
        return {
            "type": "tool_result",
            "tool_call_id": event.tool_call_id,
            "success": result.success if result else False,
            "result": jsonable_encoder(result.result) if result else None,
            "error": result.error if result else "Tool produced no result.",
        }

    if isinstance(event, ToolApprovalEvent):
        return {
            "type": "approval_item",
            "item": {
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "parameters": jsonable_encoder(event.parameters),
                "reason": event.reason_for_approval,
            },
        }

    if isinstance(event, ErrorEvent):
        return {
            "type": "error",
            "message": event.error_message,
            "error_type": event.error_type,
            "recoverable": event.is_recoverable,
        }

    return {
        "type": "event",
        "event_type": getattr(event, "event_type", type(event).__name__),
        "payload": jsonable_encoder(event),
    }


def _approval_record(record: ToolCallRecord) -> dict[str, t.Any]:
    return {
        "tool_call_id": record.id,
        "tool_name": record.tool_name,
        "parameters": jsonable_encoder(record.parameters),
        "reason": record.approval_reason,
        "status": getattr(record.status, "value", str(record.status)),
    }


def _agent_info(agent: Agent) -> dict[str, t.Any]:
    client = getattr(agent, "client", None)
    return {
        "name": agent.name,
        "description": agent.description,
        "model": getattr(client, "model", "unknown"),
        "prepared": agent.is_prepared,
    }


def _sse(payload: dict[str, t.Any]) -> str:
    encoded = json.dumps(jsonable_encoder(payload), ensure_ascii=False)
    return f"data: {encoded}\n\n"
