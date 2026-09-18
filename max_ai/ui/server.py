"""Web UI for one or more pre-built MaxAI agents."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import collections.abc as cabc
import base64
import hashlib
import json
import mimetypes
import re
import typing as t
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from max_ai.base.agent import Agent
from max_ai.config import setting
from max_ai.core.event_type import (
    CompactionEvent,
    CoreEvent,
    BashStartedEvent, BashFinishedEvent, BashFailedEvent, BashCancelledEvent,
    ErrorEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelStreamChunkEvent,
    PlanningEvent,
    ReasoningCompleteEvent,
    ReasoningIterationEvent,
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
    ToolProgressEvent,
    UserInputRequestEvent,
)
from max_ai.base.compaction import (
    TokenCounter,
    client_max_output_tokens,
    live_message_budget_tokens,
    live_message_capacity_tokens,
    live_message_threshold_tokens,
)
from max_ai.core.messages import ImagePart, TextPart, UserMessage
from max_ai.termination import CancellationToken
from max_ai.types.agent_response import AgentResponse
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord
from max_ai.workspace_copy.artifacts import ArtifactConflict


STATIC_DIR = Path(__file__).parent / "static"
WORKSPACE_EXCLUDES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "node_modules",
    "ui copy",
    "UI copy_",
}
MAX_WORKSPACE_FILES = 1000
MAX_PREVIEW_BYTES = 400_000
MAX_RAW_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
_INTERNAL_RUNTIME_PATTERNS = (
    re.compile(r"\$SKILLS_DIR\b"),
    re.compile(r"\$TOOLS_DIR\b"),
    re.compile(r"\$RUNTIME_DIR\b"),
    re.compile(r"\$WORKSPACE_DIR\b"),
    re.compile(r"/mnt/skills/[^\s\"'`]+"),
    re.compile(r"/mnt/tools/[^\s\"'`]+"),
    re.compile(r"/mnt/artifacts/[^\s\"'`]+"),
    re.compile(r"\bskills/[^\s\"'`]+/SKILL\.md\b"),
    re.compile(r"\bskills/[^\s\"'`]+/scripts/[^\s\"'`]+"),
    re.compile(r"\bread_skill\s+[A-Za-z0-9_-]+"),
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    agent_name: str | None = None
    images: list["ImageUpload"] = Field(default_factory=list)


class ChatStreamRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)
    agent_names: list[str] = Field(default_factory=list)
    images: list["ImageUpload"] = Field(default_factory=list)


class ImageUpload(BaseModel):
    name: str | None = None
    mime_type: str = "image/png"
    data_base64: str


class ApprovalDecision(BaseModel):
    tool_call_id: str
    approved: bool
    reason: str | None = None


class ApprovalRequest(BaseModel):
    decisions: list[ApprovalDecision] = Field(default_factory=list)
    agent_name: str | None = None


class ChatApproveRequest(BaseModel):
    session_id: str
    decisions: list[ApprovalDecision] = Field(default_factory=list)
    agent_name: str | None = None


class ChatCancelRequest(BaseModel):
    session_id: str
    agent_name: str | None = None


class ChatInputRequest(BaseModel):
    session_id: str
    answer: str
    agent_name: str | None = None
    tool_call_id: str | None = Field(
        default=None,
        description=(
            "Record id of the question being answered (from the "
            "user_input_request event). None answers the only pending "
            "question; an error is returned if several are pending."
        ),
    )


AgentInput = Agent | t.Sequence[Agent] | t.Mapping[str, Agent]


def create_app(
    agents: AgentInput,
    *,
    workspace_root: str | Path | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> FastAPI:
    """Create a FastAPI app that renders and drives the given agent(s)."""

    normalized_agents = _normalize_agents(agents)
    root = _resolve_workspace_root(
        workspace_root,
        next(iter(normalized_agents.values())).workspace.base_root,
    )
    if workspace_root is not None:
        for agent in normalized_agents.values():
            agent.workspace.base_root = root

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            managers = {agent.environment_manager for agent in normalized_agents.values()}
            results = await asyncio.gather(
                *(manager.close() for manager in managers), return_exceptions=True,
            )
            errors = [result for result in results if isinstance(result, Exception)]
            if errors:
                raise ExceptionGroup("Environment shutdown failed", errors)

    app = FastAPI(title="MaxAI WebUI", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.agents = normalized_agents
    app.state.agent_name = next(iter(app.state.agents))
    app.state.default_user_id = user_id or f"user_{uuid.uuid4().hex}"
    app.state.default_session_id = session_id
    initial_session_id = session_id or uuid.uuid4().hex
    app.state.contexts = _session_contexts(
        app.state.agents,
        user_id=app.state.default_user_id,
        session_id=initial_session_id,
    )
    app.state.sessions = {}
    app.state.workspace_root = root
    app.state.workspace_root.mkdir(parents=True, exist_ok=True)
    app.state.turn_lock = asyncio.Lock()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/info")
    async def info() -> dict[str, t.Any]:
        return _info_payload(app)

    @app.get("/api/agents")
    async def agents() -> list[dict[str, t.Any]]:
        return [
            _agent_summary(name, agent, default_selected=name == app.state.agent_name)
            for name, agent in app.state.agents.items()
        ]

    @app.get("/api/agents/{agent_name}")
    async def agent_detail(agent_name: str) -> dict[str, t.Any]:
        agent_name = _select_agent_name(app, agent_name)
        return _agent_info(app.state.agents[agent_name])

    @app.post("/api/sessions")
    async def create_session() -> dict[str, t.Any]:
        for agent in app.state.agents.values():
            await agent.prepare()
        return _create_session(app)

    @app.get("/api/chat/context")
    async def chat_context(
        session_id: str = Query(min_length=1),
        agent_name: str | None = None,
    ) -> dict[str, t.Any]:
        selected = _select_agent_name(app, agent_name)
        session = _get_session(app, session_id)
        ctx = session["contexts"][selected]
        agent = app.state.agents[selected]
        await agent.prepare()
        return _context_usage(ctx, agent)

    @app.get("/api/workspace")
    async def workspace(agent_name: str | None = None) -> dict[str, t.Any]:
        selected = _select_agent_name(app, agent_name)
        user_id = _workspace_user_id(app, selected)
        filesystem = _workspace_filesystem(app, selected)
        return {
            "root": ".",
            "files": await _workspace_files(
                app, selected=selected, user_id=user_id, filesystem=filesystem
            ),
        }

    @app.get("/api/workspace/files")
    async def workspace_files(agent_name: str | None = None) -> list[dict[str, t.Any]]:
        selected = _select_agent_name(app, agent_name)
        user_id = _workspace_user_id(app, selected)
        filesystem = _workspace_filesystem(app, selected)
        return await _workspace_files(
            app, selected=selected, user_id=user_id, filesystem=filesystem
        )

    @app.get("/api/workspace/sync")
    @app.post("/api/workspace/sync")
    async def workspace_sync(agent_name: str | None = None) -> dict[str, t.Any]:
        selected = _select_agent_name(app, agent_name)
        user_id = _workspace_user_id(app, selected)
        filesystem = _workspace_filesystem(app, selected)
        raise HTTPException(status_code=501, detail="Artifact synchronization is not enabled")

    @app.post("/api/workspace/upload")
    async def workspace_upload(
        request: Request,
        path: str = Query(min_length=1),
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, t.Any]:
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/octet-stream":
            raise HTTPException(status_code=400, detail="Expected application/octet-stream")
        selected = _select_agent_name(app, agent_name)
        filesystem = _workspace_filesystem(app, selected)
        if session_id is None:
            context = app.state.contexts[selected]
        else:
            session = _get_session(app, session_id)
            context = session["contexts"][selected]
            if context.user_id != app.state.contexts[selected].user_id:
                raise HTTPException(status_code=404, detail="Unknown session")
        declared_length = request.headers.get("content-length")
        if declared_length:
            try:
                if int(declared_length) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds 8 MiB limit")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid Content-Length") from None
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Upload exceeds 8 MiB limit")
            body.extend(chunk)
        try:
            result = await asyncio.to_thread(
                filesystem.write_bytes,
                context.user_id,
                f"{context.session_id}/{path}",
                bytes(body),
            )
        except (FileExistsError, ArtifactConflict):
            raise HTTPException(status_code=409, detail="File already exists") from None
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid workspace upload") from None
        except (OSError, RuntimeError):
            raise HTTPException(status_code=500, detail="Workspace upload failed") from None
        return result

    @app.get("/api/workspace/file")
    async def workspace_file(
        path: str = Query(min_length=1), agent_name: str | None = None
    ) -> dict[str, t.Any]:
        selected = _select_agent_name(app, agent_name)
        user_id = _workspace_user_id(app, selected)
        filesystem = _workspace_filesystem(app, selected)
        item = await _workspace_file_item(filesystem, user_id, path)
        name = PurePosixPath(path).name
        kind = _file_kind(PurePosixPath(name))
        payload: dict[str, t.Any] = {
            "name": name,
            "path": item["path"],
            "size": item["bytes"],
            "kind": kind,
            "mime": mimetypes.guess_type(name)[0],
            "sync_status": item["sync_status"],
        }
        if kind in {"text", "html"}:
            content = await _read_workspace_bytes(
                filesystem, user_id, path, MAX_PREVIEW_BYTES
            )
            text = content.decode("utf-8", errors="replace")
            if item["bytes"] > len(content):
                text += "\n\n[Preview truncated]"
            payload["content"] = text
        return payload

    @app.get("/api/workspace/raw")
    async def workspace_raw(
        path: str = Query(min_length=1), agent_name: str | None = None
    ) -> Response:
        selected = _select_agent_name(app, agent_name)
        user_id = _workspace_user_id(app, selected)
        filesystem = _workspace_filesystem(app, selected)
        item = await _workspace_file_item(filesystem, user_id, path)
        if item["bytes"] > MAX_RAW_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds download limit")
        content = await _read_workspace_bytes(filesystem, user_id, path, MAX_RAW_BYTES)
        name = PurePosixPath(path).name
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name, safe='')}",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post("/api/clear")
    async def clear(agent_name: str | None = None) -> dict[str, str]:
        agent_name = _select_agent_name(app, agent_name)
        previous = app.state.contexts[agent_name]
        app.state.contexts = dict(app.state.contexts)
        app.state.contexts[agent_name] = RunContext(
            user_id=previous.user_id,
            session_id=uuid.uuid4().hex,
        )
        return {"status": "ok"}

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        async def stream() -> t.AsyncIterator[str]:
            app.state.agent_name = _select_agent_name(app, req.agent_name)
            async for payload in _run_turn(app, req):
                yield _sse(payload)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatStreamRequest) -> StreamingResponse:
        async def stream() -> t.AsyncIterator[str]:
            agent_name = _select_agent_name(
                app,
                req.agent_names[0] if req.agent_names else None,
            )
            app.state.agent_name = agent_name
            session = _get_session(app, req.session_id)
            ctx = session["contexts"][agent_name]
            chat_req = ChatRequest(
                message=req.message,
                agent_name=agent_name,
                images=req.images,
            )
            async for payload in _run_turn_for_context(
                app,
                chat_req,
                agent_name,
                ctx,
                session=session,
            ):
                yield _sse(payload)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/approve")
    async def approve(req: ApprovalRequest) -> StreamingResponse:
        async def stream() -> t.AsyncIterator[str]:
            app.state.agent_name = _select_agent_name(app, req.agent_name)
            async for payload in _resume_turn(app, req.decisions):
                yield _sse(payload)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/chat/approve")
    async def chat_approve(req: ChatApproveRequest) -> StreamingResponse:
        async def stream() -> t.AsyncIterator[str]:
            agent_name = _select_agent_name(app, req.agent_name)
            app.state.agent_name = agent_name
            session = _get_session(app, req.session_id)
            ctx = session["contexts"][agent_name]
            async for payload in _resume_turn_for_context(
                app,
                req.decisions,
                agent_name,
                ctx,
                session=session,
            ):
                yield _sse(payload)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/chat/cancel")
    async def chat_cancel(req: ChatCancelRequest) -> dict[str, str]:
        agent_name = _select_agent_name(app, req.agent_name)
        session = _get_session(app, req.session_id)
        token = session.get("active_tokens", {}).get(agent_name)
        if token is None:
            return {"status": "idle"}
        token.cancel()
        return {"status": "cancelling"}

    @app.post("/api/chat/input")
    async def chat_input(req: ChatInputRequest) -> StreamingResponse:
        # Answer a pending question. Elicitation is durable record state
        # (like approvals): the previous stream already ended with
        # finish_reason='input_needed', so we apply the answer to the
        # session context and resume it as a fresh stream segment.
        async def stream() -> t.AsyncIterator[str]:
            agent_name = _select_agent_name(app, req.agent_name)
            app.state.agent_name = agent_name
            session = _get_session(app, req.session_id)
            ctx = session["contexts"][agent_name]

            pending = ctx.tool_state.pending_user_input
            tool_call_id = req.tool_call_id
            if tool_call_id is None:
                if len(pending) != 1:
                    yield _sse({
                        "type": "error",
                        "message": (
                            f"{len(pending)} question(s) pending; pass "
                            "tool_call_id to disambiguate."
                        ),
                    })
                    return
                tool_call_id = pending[0].id

            try:
                ctx.tool_state.apply_user_answer(tool_call_id, req.answer)
            except (KeyError, ValueError) as exc:
                yield _sse({"type": "error", "message": str(exc)})
                return

            async for payload in _resume_turn_for_context(
                app,
                decisions=[],
                agent_name=agent_name,
                ctx=ctx,
                session=session,
            ):
                yield _sse(payload)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def serve(
    agents: AgentInput,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    workspace_root: str | Path | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Run the Web UI for ``agents``.

    Typical usage from another file:

    ``from max_ai.ui import server``
    ``server(agent)``
    """

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "uvicorn is required to serve the MaxAI WebUI. "
            "Install the ui dependency group or add uvicorn."
        ) from exc

    uvicorn.run(
        create_app(
            agents,
            workspace_root=workspace_root,
            user_id=user_id,
            session_id=session_id,
        ),
        host=host,
        port=port,
        reload=reload,
    )


server = serve


async def _run_turn(app: FastAPI, req: ChatRequest) -> t.AsyncIterator[dict[str, t.Any]]:
    agent_name = app.state.agent_name
    ctx: RunContext = app.state.contexts[agent_name]
    async for payload in _run_turn_for_context(app, req, agent_name, ctx, legacy=True):
        yield payload


async def _run_turn_for_context(
    app: FastAPI,
    req: ChatRequest,
    agent_name: str,
    ctx: RunContext,
    *,
    legacy: bool = False,
    session: dict[str, t.Any] | None = None,
) -> t.AsyncIterator[dict[str, t.Any]]:
    lock: asyncio.Lock = app.state.turn_lock
    if lock.locked():
        yield {"type": "error", "message": "Another turn is already running."}
        return

    async with lock:
        agent: Agent = app.state.agents[agent_name]
        yield {"type": "status", "status": "running"}
        cancellation_token = CancellationToken()
        if session is not None:
            session.setdefault("active_tokens", {})[agent_name] = cancellation_token
        try:
            stream = agent.run_stream_events(
                task=_chat_task(req, await _persist_chat_uploads(agent, ctx, req.images)),
                run_context=ctx,
                cancellation_token=cancellation_token,
                stream_tokens=True,
            )
            if legacy:
                async for payload in _stream_agent_events(stream):
                    if payload.get("type") == "done" and payload.get("context") is not None:
                        app.state.contexts[agent_name] = payload.pop("context")
                    yield payload
            else:
                async for payload in _stream_ui_events(
                    stream, agent_name, agent, session=session
                ):
                    yield payload
        except asyncio.CancelledError:
            yield {"type": "cancelled", "message": "Turn cancelled."}
        except Exception as exc:  # noqa: BLE001
            yield {
                "type": "error",
                "message": str(exc),
                "error_type": type(exc).__name__,
            }
        finally:
            if session is not None:
                session.get("active_tokens", {}).pop(agent_name, None)


async def _resume_turn(
    app: FastAPI,
    decisions: list[ApprovalDecision],
) -> t.AsyncIterator[dict[str, t.Any]]:
    agent_name = app.state.agent_name
    ctx: RunContext = app.state.contexts[agent_name]
    async for payload in _resume_turn_for_context(
        app,
        decisions,
        agent_name,
        ctx,
        legacy=True,
    ):
        yield payload


async def _resume_turn_for_context(
    app: FastAPI,
    decisions: list[ApprovalDecision],
    agent_name: str,
    ctx: RunContext,
    *,
    legacy: bool = False,
    session: dict[str, t.Any] | None = None,
) -> t.AsyncIterator[dict[str, t.Any]]:
    lock: asyncio.Lock = app.state.turn_lock
    if lock.locked():
        yield {"type": "error", "message": "Another turn is already running."}
        return

    async with lock:
        agent: Agent = app.state.agents[agent_name]
        cancellation_token = CancellationToken()
        if session is not None:
            session.setdefault("active_tokens", {})[agent_name] = cancellation_token
        try:
            for decision in decisions:
                ctx.tool_state.apply_approval(
                    decision.tool_call_id,
                    approved=decision.approved,
                    reason=decision.reason,
                )

            pending_approvals = [
                _approval_record(record) for record in ctx.tool_state.pending_approvals
            ]
            if pending_approvals:
                yield {
                    "type": "status",
                    "status": "waiting_for_approval",
                    "message": f"{len(pending_approvals)} tool approval(s) still pending.",
                }
                yield {
                    "type": "session_state",
                    "agent_name": agent_name,
                    "messages": _display_messages(session, agent_name, ctx),
                    "pending_approvals": pending_approvals,
                    "context_usage": _context_usage(ctx, agent),
                }
                yield {
                    "type": "approval_required",
                    "agent_name": agent_name,
                    "pending_approvals": pending_approvals,
                }
                return

            pending_questions = [
                _question_record(record)
                for record in ctx.tool_state.pending_user_input
            ]
            if pending_questions:
                yield {
                    "type": "status",
                    "status": "waiting_for_input",
                    "message": f"{len(pending_questions)} question(s) still unanswered.",
                }
                yield {
                    "type": "input_required",
                    "agent_name": agent_name,
                    "pending_questions": pending_questions,
                }
                return

            yield {"type": "status", "status": "resuming"}
            stream = agent.resume_stream_events(
                run_context=ctx,
                cancellation_token=cancellation_token,
                stream_tokens=True,
            )
            if legacy:
                async for payload in _stream_agent_events(stream):
                    if payload.get("type") == "done" and payload.get("context") is not None:
                        app.state.contexts[agent_name] = payload.pop("context")
                    yield payload
            else:
                async for payload in _stream_ui_events(
                    stream, agent_name, agent, session=session
                ):
                    yield payload
        except asyncio.CancelledError:
            yield {"type": "cancelled", "message": "Turn cancelled."}
        except Exception as exc:  # noqa: BLE001
            yield {
                "type": "error",
                "message": str(exc),
                "error_type": type(exc).__name__,
            }
        finally:
            if session is not None:
                session.get("active_tokens", {}).pop(agent_name, None)


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
                "needs_input": item.needs_input,
                "pending_questions": [
                    _question_record(record) for record in item.pending_questions
                ],
                "usage": jsonable_encoder(item.usage),
                "context": item.context,
            }
            continue

        yield _event_payload(item)


async def _stream_ui_events(
    stream: t.AsyncIterator[CoreEvent | AgentResponse | None],
    agent_name: str,
    agent: Agent,
    session: dict[str, t.Any] | None = None,
) -> t.AsyncIterator[dict[str, t.Any]]:
    content_parts: list[str] = []
    thinking_parts: list[str] = []

    async for item in stream:
        if item is None:
            continue

        if isinstance(item, AgentResponse):
            context = item.context
            yield {
                "type": "agent_complete",
                "agent_name": agent_name,
                "assistant_message": _last_assistant_message(context),
                "finish_reason": item.finish_reason,
                "needs_approval": item.needs_approval,
                "needs_input": item.needs_input,
                "usage": jsonable_encoder(item.usage),
            }
            pending_approvals = [
                _approval_record(record) for record in item.pending_approvals
            ]
            pending_questions = [
                _question_record(record) for record in item.pending_questions
            ]
            yield {
                "type": "session_state",
                "agent_name": agent_name,
                "messages": _display_messages(session, agent_name, context),
                "pending_approvals": pending_approvals,
                "pending_questions": pending_questions,
                "context_usage": _context_usage(context, agent),
            }
            if item.needs_approval:
                yield {
                    "type": "approval_required",
                    "agent_name": agent_name,
                    "pending_approvals": pending_approvals,
                }
            if item.needs_input:
                yield {
                    "type": "input_required",
                    "agent_name": agent_name,
                    "pending_questions": pending_questions,
                }
            continue

        payload = _event_payload(item)
        if payload["type"] == "model_call":
            # New LLM call within the same turn (guard veto, plan step,
            # post-tool round): restart the delta accumulators so the new
            # answer streams fresh instead of being appended to the
            # previous one.
            content_parts = []
            thinking_parts = []
        if payload["type"] == "token":
            thinking = payload.get("thinking")
            text = payload.get("text")
            if thinking:
                thinking_parts.append(thinking)
                yield {
                    "type": "thinking_delta",
                    "agent_name": agent_name,
                    "content": "".join(thinking_parts),
                    "is_final": payload.get("is_final", False),
                }
            if text:
                content_parts.append(text)
                yield {
                    "type": "assistant_delta",
                    "agent_name": agent_name,
                    "content": "".join(content_parts),
                    "is_final": payload.get("is_final", False),
                }
            continue

        event = dict(payload)
        event.setdefault("event_type", payload["type"])
        yield {
            "type": "agent_event",
            "agent_name": agent_name,
            "event": event,
        }


def _event_payload(event: CoreEvent) -> dict[str, t.Any]:
    if isinstance(event, ModelCallEvent):
        return {
            "type": "model_call",
            "event_type": event.event_type,
            "model": event.model,
            "input_messages": _serialize_messages(event.input_messages),
        }

    if isinstance(event, ReasoningIterationEvent):
        return {
            "type": "reasoning",
            "event_type": event.event_type,
            "phase": "iteration",
            "iteration": event.iteration,
            "max_iterations": event.max_iterations,
        }

    if isinstance(event, ReasoningCompleteEvent):
        return {
            "type": "reasoning",
            "event_type": event.event_type,
            "phase": "complete",
            "finish_reason": event.finish_reason,
            "total_iterations": event.total_iterations,
        }

    if isinstance(event, CompactionEvent):
        return {
            "type": "compaction",
            "event_type": event.event_type,
            "phase": event.phase,
            "strategy": event.strategy,
            "changed": event.changed,
            "old_message_count": event.old_message_count,
            "recent_message_count": event.recent_message_count,
            "old_token_count": event.old_token_count,
            "recent_token_count": event.recent_token_count,
            "total_token_count": event.total_token_count,
            "live_message_threshold_tokens": event.live_message_threshold_tokens,
            "live_message_budget_tokens": event.live_message_budget_tokens,
            "summary": event.summary,
            "context_summary_persisted": event.context_summary_persisted,
            "context_summary_session_id": event.context_summary_session_id,
        }

    if isinstance(event, ModelStreamChunkEvent):
        return {
            "type": "token",
            "text": event.chunk,
            "thinking": event.thinking,
            "is_final": event.is_final,
        }

    if isinstance(event, ModelResponseEvent):
        return {
            "type": "assistant_text",
            "event_type": event.event_type,
            "text": event.response,
            "has_tool_calls": event.has_tool_calls,
            "usage": jsonable_encoder(event.usage),
        }

    if isinstance(event, (BashStartedEvent, BashFinishedEvent, BashFailedEvent, BashCancelledEvent)):
        return {**jsonable_encoder(event), "type": event.event_type}

    if isinstance(event, ToolCallEvent):
        parameters = jsonable_encoder(event.parameters)
        if event.tool_name == "bash":
            parameters = {"command": "[sandbox command hidden]"}
        return {
            "type": "tool_call",
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "parameters": parameters,
        }

    if isinstance(event, ToolProgressEvent):
        return {
            "type": "tool_progress",
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "content": event.content,
        }

    if isinstance(event, ToolCallResponseEvent):
        result = event.tool_result
        return {
            "type": "tool_result",
            "tool_call_id": event.tool_call_id,
            "success": result.success if result else False,
            "result": _redact_internal_runtime_details(
                jsonable_encoder(result.result) if result else None
            ),
            "error": _redact_internal_runtime_details(
                result.error if result else "Tool produced no result."
            ),
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

    if isinstance(event, PlanningEvent):
        return {
            "type": "planning",
            "event_type": event.event_type,
            "phase": event.phase,  # start | progress | complete | failed
            "plan": jsonable_encoder(event.plan) if event.plan else None,
        }

    if isinstance(event, UserInputRequestEvent):
        return {
            "type": "user_input_request",
            "event_type": event.event_type,
            "question": event.question,
            "options": event.options,
            "tool_call_id": event.tool_call_id,
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
        "parameters": (
            {"command": "[sandbox command hidden]"}
            if record.tool_name == "bash"
            else jsonable_encoder(record.parameters)
        ),
        "reason": record.approval_reason,
        "status": getattr(record.status, "value", str(record.status)),
    }


def _question_record(record: ToolCallRecord) -> dict[str, t.Any]:
    return {
        "tool_call_id": record.id,
        "question": record.input_question or "",
        "options": record.input_options,
        "status": getattr(record.status, "value", str(record.status)),
    }


def _redact_internal_runtime_details(value: t.Any) -> t.Any:
    if isinstance(value, str):
        redacted = value
        for pattern in _INTERNAL_RUNTIME_PATTERNS:
            redacted = pattern.sub("[internal runtime path]", redacted)
        return redacted
    if isinstance(value, list):
        return [_redact_internal_runtime_details(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                "[sandbox command hidden]"
                if key == "command"
                else _redact_internal_runtime_details(item)
            )
            for key, item in value.items()
        }
    return value


def _normalize_agents(agents: AgentInput) -> dict[str, Agent]:
    if isinstance(agents, Agent):
        return {agents.name: agents}

    if isinstance(agents, cabc.Mapping):
        normalized = dict(agents)
    else:
        normalized = {agent.name: agent for agent in agents}

    if not normalized:
        raise ValueError("server() requires at least one agent.")

    for name, agent in normalized.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Agent names must be non-empty strings.")
        if not isinstance(agent, Agent):
            raise TypeError(f"Expected Agent for '{name}', got {type(agent).__name__}.")
    return normalized




def _session_contexts(
    agents: cabc.Mapping[str, Agent],
    *,
    user_id: str,
    session_id: str,
) -> dict[str, RunContext]:
    return {
        name: RunContext(user_id=user_id, session_id=session_id)
        for name in agents
    }

def _create_session(app: FastAPI) -> dict[str, t.Any]:
    session_id = app.state.default_session_id or uuid.uuid4().hex
    user_id = app.state.default_user_id
    contexts = _session_contexts(
        app.state.agents,
        user_id=user_id,
        session_id=session_id,
    )
    # "display" is the server-side conversation log per agent: the agent
    # itself is stateless (compaction trims ctx.messages), so what the
    # user keeps seeing is owned here, by the UI layer.
    app.state.sessions[session_id] = {
        "contexts": contexts,
        "active_tokens": {},
        "display": {},
    }
    app.state.contexts = contexts
    return {
        "session_id": session_id,
        "user_id": user_id,
        "messages": [],
        "pending_approvals": [],
        "context_usage": _context_usage(
            next(iter(contexts.values())),
            next(iter(app.state.agents.values())),
        ),
    }


def _get_session(app: FastAPI, session_id: str) -> dict[str, t.Any]:
    try:
        return app.state.sessions[session_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session") from exc


def _resolve_workspace_root(
    workspace_root: str | Path | None,
    default_root: str | Path,
) -> Path:
    root = Path(workspace_root) if workspace_root is not None else Path(default_root)
    return root.expanduser().resolve()


def _workspace_filesystem(app: FastAPI, agent_name: str | None = None) -> t.Any:
    selected_agent = _select_agent_name(app, agent_name)
    return app.state.agents[selected_agent].workspace.get_filesystem()


def _workspace_user_id(app: FastAPI, agent_name: str | None = None) -> str:
    selected_agent = _select_agent_name(app, agent_name)
    ctx: RunContext = app.state.contexts[selected_agent]
    return ctx.user_id


def _select_agent_name(app: FastAPI, requested: str | None) -> str:
    if requested is None:
        return app.state.agent_name
    if requested not in app.state.agents:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {requested}")
    return requested


def _info_payload(app: FastAPI) -> dict[str, t.Any]:
    active_name = app.state.agent_name
    active = app.state.agents[active_name]
    payload = _agent_info(active)
    payload["active_agent"] = active_name
    payload["workspace_root"] = "."
    payload["agents"] = [
        _agent_summary(name, agent, default_selected=name == active_name)
        for name, agent in app.state.agents.items()
    ]
    return payload


def _agent_summary(
    name: str,
    agent: Agent,
    *,
    default_selected: bool = False,
) -> dict[str, t.Any]:
    client = getattr(agent, "client", None)
    config = getattr(client, "config", None)
    return {
        "name": name,
        "label": agent.name or name,
        "display_name": agent.name,
        "description": agent.description,
        "model": getattr(client, "model", "unknown"),
        "model_name": getattr(client, "model", "unknown"),
        "context_window": getattr(config, "max_context_window", None),
        "supports_vision": bool(getattr(config, "supports_vision", False)),
        "prepared": agent.is_prepared,
        "default_selected": default_selected,
        "invocation_hint": f"@{name}",
    }


def _agent_info(agent: Agent) -> dict[str, t.Any]:
    client = getattr(agent, "client", None)
    capabilities = getattr(agent, "capabilities", None)
    tools = []
    skills = []
    if capabilities is not None:
        try:
            tools = [
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                }
                for tool in capabilities.all_tools
            ]
        except Exception:  # noqa: BLE001 - info should not break chat
            tools = []

        try:
            skills = [
                {
                    "name": skill.name,
                    "description": skill.description,
                }
                for skill in capabilities.loaded_skill_blocks
            ]
        except Exception:  # noqa: BLE001 - info should not break chat
            skills = []

    config = getattr(client, "config", None)
    return {
        "name": agent.name,
        "description": agent.description,
        "model": getattr(client, "model", "unknown"),
        "model_name": getattr(client, "model", "unknown"),
        "prepared": agent.is_prepared,
        "max_context_tokens": getattr(config, "max_context_window", None),
        "context_window": getattr(config, "max_context_window", None),
        "tools": tools,
        "skills": skills,
    }


def _context_messages(ctx: RunContext) -> list[t.Any]:
    return [
        *ctx.message_history.iter_messages(),
        *ctx.messages,
    ]


def _display_key(data: dict[str, t.Any]) -> str:
    """Stable identity for a serialized message in the display log.

    Tool messages carry a unique ``tool_call_id``. Everything else is
    keyed by role/source/created_at — ``created_at`` has microsecond
    precision and survives ``model_copy`` updates, so a message whose
    ``interim`` flag flips upserts in place instead of duplicating.
    """
    tool_call_id = data.get("tool_call_id")
    if tool_call_id:
        return f"tool:{tool_call_id}"
    return f"{data.get('role')}|{data.get('source')}|{data.get('created_at')}"


def _display_messages(
    session: dict[str, t.Any] | None,
    agent_name: str,
    ctx: RunContext,
) -> list[dict[str, t.Any]]:
    """The conversation as the USER should see it.

    The agent is stateless by design: compaction evicts old messages from
    ``ctx.messages`` and the model works from a summary. What the user
    sees is the UI's job — so the server keeps an append-only display log
    per session/agent, upserting whatever the context currently holds.
    Messages compaction later evicts stay visible because they were
    logged before eviction.
    """
    current = _serialize_messages(_context_messages(ctx))
    if session is None:
        return current
    log: dict[str, dict[str, t.Any]] = session.setdefault(
        "display", {}
    ).setdefault(agent_name, {})
    for data in current:
        log[_display_key(data)] = data
    return list(log.values())


def _serialize_messages(messages: t.Sequence[t.Any]) -> list[dict[str, t.Any]]:
    return [_serialize_message(message) for message in messages]


def _serialize_message(message: t.Any) -> dict[str, t.Any]:
    if hasattr(message, "model_dump"):
        data = message.model_dump(mode="python")
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {"role": "assistant", "source": "assistant", "content": str(message)}

    content = data.get("content", "")
    if isinstance(content, list):
        text_parts: list[str] = []
        images: list[dict[str, str]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image" and part.get("data") is not None:
                raw_data = part.get("data")
                if isinstance(raw_data, str):
                    data_base64 = raw_data
                else:
                    data_base64 = base64.b64encode(raw_data).decode("utf-8")
                mime_type = part.get("mime_type") or "image/png"
                images.append(
                    {
                        "mime_type": mime_type,
                        "data_base64": data_base64,
                        "data_url": f"data:{mime_type};base64,{data_base64}",
                    }
                )
        data["content"] = "".join(text_parts)
        if images:
            data["images"] = images
    return data


def _last_assistant_message(ctx: RunContext) -> dict[str, t.Any] | None:
    # The user-facing final answer: skip guard-vetoed drafts (interim) and
    # prefer a message with text over a text-less tool-call shell (e.g. the
    # trailing update_plan call that closes out a plan).
    fallback = None
    for message in reversed(_context_messages(ctx)):
        if getattr(message, "role", None) != "assistant":
            continue
        if getattr(message, "interim", False):
            continue
        text_fn = getattr(message, "text", None)
        text = text_fn() if callable(text_fn) else str(message.content or "")
        if text.strip():
            return _serialize_message(message)
        if fallback is None:
            fallback = message
    return _serialize_message(fallback) if fallback is not None else None



def _context_usage(ctx: RunContext, agent: Agent) -> dict[str, t.Any]:
    client = getattr(agent, "client", None)
    config = getattr(client, "config", None)
    tokenizer_base = getattr(config, "tokenizer_base", "o200k_base")
    counter = TokenCounter(tokenizer_base=tokenizer_base)
    context_messages = _context_messages(ctx)
    live_tokens = counter.count_messages(context_messages)
    max_context = getattr(config, "max_context_window", None) or None
    prompt_tokens = 0
    prompt_layers: list[dict[str, t.Any]] = []
    if getattr(agent, "is_prepared", False):
        try:
            prompt_tokens = agent.prompt_tokens
            prompt_layers = [
                {
                    "name": name,
                    "tokens": usage.tokens,
                    "chars": usage.chars,
                }
                for name, usage in agent.rendered_layer_usage.items()
            ]
        except Exception:  # noqa: BLE001 - telemetry should not break chat
            prompt_tokens = 0
            prompt_layers = []

    summary = ctx.runtime_state.shared_state.get("compaction_summary")
    budgeted_prompt_tokens = setting.compaction_prompt_budget_tokens
    reserved_output = client_max_output_tokens(client)
    safety_margin = int(max_context * setting.compaction_safety_margin_ratio) if max_context else 0
    used = budgeted_prompt_tokens + live_tokens + reserved_output + safety_margin
    live_capacity = (
        live_message_capacity_tokens(
            max_context,
            max_output_tokens=reserved_output,
        )
        if max_context
        else None
    )
    live_threshold = (
        live_message_threshold_tokens(
            max_context,
            max_output_tokens=reserved_output,
        )
        if max_context
        else None
    )

    return {
        "used": used,
        "max": max_context,
        "live_message_tokens": live_tokens,
        "prompt_tokens": budgeted_prompt_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "summary_tokens": 0,
        "reserved_output_tokens": reserved_output,
        "safety_margin_tokens": safety_margin,
        "prompt_budget_tokens": setting.compaction_prompt_budget_tokens,
        "summary_budget_tokens": setting.compaction_summary_budget_tokens,
        "live_message_capacity_tokens": live_capacity,
        "live_message_threshold_tokens": live_threshold,
        "live_message_budget_tokens": (
            live_message_budget_tokens(live_capacity)
            if live_capacity is not None
            else None
        ),
        "message_count": len(context_messages),
        "summary": summary,
        "prompt_layers": prompt_layers,
    }


def _sse(payload: dict[str, t.Any]) -> str:
    encoded = json.dumps(jsonable_encoder(payload), ensure_ascii=False)
    return f"data: {encoded}\n\n"


async def _persist_chat_uploads(
    agent: Agent, ctx: RunContext, images: t.Sequence[ImageUpload]
) -> list[str]:
    """Persist chat images before the model run and return visible workspace paths."""
    if not images:
        return []
    filesystem = agent.workspace.get_filesystem()
    persisted: list[str] = []
    for image in images:
        encoded = image.data_base64
        if len(encoded) > ((MAX_UPLOAD_BYTES + 2) // 3) * 4 + 4:
            raise ValueError("Invalid uploaded image")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            raise ValueError("Invalid uploaded image") from None
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Invalid uploaded image")
        name = _safe_image_basename(image.name)
        digest = hashlib.sha256(data).hexdigest()
        path = f"uploads/{digest}/{name}"
        try:
            result = await asyncio.to_thread(
                filesystem.write_bytes, ctx.user_id, f"{ctx.session_id}/{path}", data
            )
            persisted_path = str(result["path"])
        except FileExistsError:
            # Same-content retries map to the same digest path and are safe.
            persisted_path = f"{ctx.session_id}/{path}"
            try:
                existing = await asyncio.to_thread(
                    filesystem.read_snapshot, ctx.user_id, persisted_path
                )
            except (OSError, ValueError):
                raise ValueError("Could not persist uploaded image") from None
            if existing != data:
                raise ValueError("Could not persist uploaded image")
        except (OSError, RuntimeError):
            raise ValueError("Could not persist uploaded image") from None
        persisted.append(persisted_path)
    ctx.runtime_state.shared_state["uploaded_images"] = list(persisted)
    return persisted


def _safe_image_basename(name: str | None) -> str:
    if name is None or not name:
        return "upload.bin"
    if (
        name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\0" in name
        or name.strip() != name
    ):
        raise ValueError("Invalid uploaded image name")
    safe = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    safe = safe[:200].rstrip(" .")
    if not safe or safe in {".", ".."}:
        raise ValueError("Invalid uploaded image name")
    return safe


def _chat_task(
    req: ChatRequest, uploaded_paths: t.Sequence[str] = ()
) -> str | UserMessage:
    if not req.images:
        return req.message

    note = ""
    if uploaded_paths:
        note = "\n\nUploaded images are available in the workspace at: " + ", ".join(
            uploaded_paths
        )
    parts = [TextPart(text=req.message + note)]
    for image in req.images:
        try:
            data = base64.b64decode(image.data_base64, validate=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid image payload") from exc
        parts.append(ImagePart(data=data, mime_type=image.mime_type))
    return UserMessage(source="user", content=parts)


async def _workspace_files(
    app: FastAPI,
    *,
    selected: str,
    user_id: str,
    filesystem: t.Any,
) -> list[dict[str, t.Any]]:
    try:
        items, _ = await asyncio.to_thread(
            filesystem.scan_files, user_id, path="", limit=MAX_WORKSPACE_FILES
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    files: list[dict[str, t.Any]] = []
    for item in items:
        path = item["path"]
        pure_path = PurePosixPath(path)
        if pure_path.name.startswith(".") or any(
            part in WORKSPACE_EXCLUDES for part in pure_path.parts
        ):
            continue
        name = pure_path.name
        sync_status = {"state": "disabled"}
        files.append(
            {
                "name": name,
                "path": path,
                "size": item["bytes"],
                "size_bytes": item["bytes"],
                "kind": _file_kind(pure_path),
                "url": f"/api/workspace/raw?{urlencode({'path': path})}",
                "sync_status": _public_sync_status(sync_status),
            }
        )
    return files


async def _workspace_file_item(
    filesystem: t.Any, user_id: str, path: str
) -> dict[str, t.Any]:
    try:
        items, _ = await asyncio.to_thread(
            filesystem.scan_files, user_id, path=path, limit=1
        )
        for item in items:
            if item["path"] == path:
                item["sync_status"] = _public_sync_status(
                    {"state": "disabled"}
                )
                return item
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    raise HTTPException(status_code=404, detail="File not found")


async def _read_workspace_bytes(
    filesystem: t.Any, user_id: str, path: str, max_bytes: int
) -> bytes:
    try:
        return await asyncio.to_thread(
            filesystem.read_bytes,
            user_id,
            path,
            max_bytes=max_bytes,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc


def _public_sync_status(status: t.Mapping[str, t.Any]) -> dict[str, t.Any]:
    state = status.get("state")
    if state not in {"synced", "pending", "conflict", "error", "disabled"}:
        state = "error"
    result: dict[str, t.Any] = {"state": state}
    for key in ("revision", "sha256"):
        value = status.get(key)
        if isinstance(value, str):
            result[key] = value
    if state == "error":
        result["error"] = "Sync failed"
    elif isinstance(status.get("error"), str) and status["error"]:
        result["error"] = "Sync issue"
    return result


def _file_kind(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    suffix = path.suffix.lower()
    if mime and mime.startswith("image/"):
        return "image"
    if mime == "application/pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".css",
        ".html",
        ".md",
        ".txt",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
        ".csv",
        ".sh",
        ".dockerfile",
    }:
        return "text"
    if mime and mime.startswith("text/"):
        return "text"
    return "binary"
