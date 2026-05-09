"""Web UI for one or more pre-built MaxAI agents."""

from __future__ import annotations

import asyncio
import collections.abc as cabc
import base64
import json
import mimetypes
import os
import typing as t
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from max_ai.base.agent import Agent
from max_ai.config import setting
from max_ai.core.event_type import (
    CompactionEvent,
    CoreEvent,
    ErrorEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelStreamChunkEvent,
    ReasoningCompleteEvent,
    ReasoningIterationEvent,
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)
from max_ai.core.compaction import TokenBudgetStrategy
from max_ai.core.messages import ImagePart, TextPart, UserMessage
from max_ai.types.agent_response import AgentResponse
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord


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
MAX_WORKSPACE_FILES = 250
MAX_PREVIEW_BYTES = 400_000


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


AgentInput = Agent | t.Sequence[Agent] | t.Mapping[str, Agent]


def create_app(
    agents: AgentInput,
    *,
    workspace_root: str | Path | None = None,
) -> FastAPI:
    """Create a FastAPI app that renders and drives the given agent(s)."""

    app = FastAPI(title="MaxAI WebUI")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.agents = _normalize_agents(agents)
    app.state.agent_name = next(iter(app.state.agents))
    app.state.contexts = {
        name: RunContext() for name in app.state.agents
    }
    app.state.sessions = {}
    app.state.workspace_root = _resolve_workspace_root(workspace_root)
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
        return _create_session(app)

    @app.get("/api/workspace")
    async def workspace() -> dict[str, t.Any]:
        root = _active_workspace_root(app)
        return {
            "root": str(root),
            "files": _workspace_files(root),
        }

    @app.get("/api/workspace/files")
    async def workspace_files() -> list[dict[str, t.Any]]:
        return _workspace_files(_active_workspace_root(app))

    @app.get("/api/workspace/file")
    async def workspace_file(path: str = Query(min_length=1)) -> dict[str, t.Any]:
        root = _active_workspace_root(app)
        file_path = _safe_workspace_path(root, path)
        stat = file_path.stat()
        kind = _file_kind(file_path)
        payload: dict[str, t.Any] = {
            "name": file_path.name,
            "path": _workspace_relpath(root, file_path),
            "size": stat.st_size,
            "kind": kind,
            "mime": mimetypes.guess_type(file_path.name)[0],
        }
        if kind == "text":
            payload["content"] = _read_preview_text(file_path)
        return payload

    @app.get("/api/workspace/raw")
    async def workspace_raw(path: str = Query(min_length=1)) -> FileResponse:
        return FileResponse(_safe_workspace_path(_active_workspace_root(app), path))

    @app.post("/api/clear")
    async def clear(agent_name: str | None = None) -> dict[str, str]:
        agent_name = _select_agent_name(app, agent_name)
        app.state.contexts[agent_name] = RunContext()
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
            async for payload in _run_turn_for_context(app, chat_req, agent_name, ctx):
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
            async for payload in _resume_turn_for_context(app, req.decisions, agent_name, ctx):
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
        create_app(agents, workspace_root=workspace_root),
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
) -> t.AsyncIterator[dict[str, t.Any]]:
    lock: asyncio.Lock = app.state.turn_lock
    if lock.locked():
        yield {"type": "error", "message": "Another turn is already running."}
        return

    async with lock:
        agent: Agent = app.state.agents[agent_name]
        yield {"type": "status", "status": "running"}
        try:
            stream = agent.run_stream_events(
                task=_chat_task(req),
                run_context=ctx,
                stream_tokens=True,
            )
            if legacy:
                async for payload in _stream_agent_events(stream):
                    if payload.get("type") == "done" and payload.get("context") is not None:
                        app.state.contexts[agent_name] = payload.pop("context")
                    yield payload
            else:
                async for payload in _stream_ui_events(stream, agent_name, agent):
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
) -> t.AsyncIterator[dict[str, t.Any]]:
    lock: asyncio.Lock = app.state.turn_lock
    if lock.locked():
        yield {"type": "error", "message": "Another turn is already running."}
        return

    async with lock:
        agent: Agent = app.state.agents[agent_name]
        try:
            for decision in decisions:
                ctx.tool_state.apply_approval(
                    decision.tool_call_id,
                    approved=decision.approved,
                    reason=decision.reason,
                )

            yield {"type": "status", "status": "resuming"}
            stream = agent.resume_stream_events(run_context=ctx, stream_tokens=True)
            if legacy:
                async for payload in _stream_agent_events(stream):
                    if payload.get("type") == "done" and payload.get("context") is not None:
                        app.state.contexts[agent_name] = payload.pop("context")
                    yield payload
            else:
                async for payload in _stream_ui_events(stream, agent_name, agent):
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


async def _stream_ui_events(
    stream: t.AsyncIterator[CoreEvent | AgentResponse | None],
    agent_name: str,
    agent: Agent,
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
                "usage": jsonable_encoder(item.usage),
            }
            pending_approvals = [
                _approval_record(record) for record in item.pending_approvals
            ]
            yield {
                "type": "session_state",
                "agent_name": agent_name,
                "messages": _serialize_messages(context.messages),
                "pending_approvals": pending_approvals,
                "context_usage": _context_usage(context, agent),
            }
            if item.needs_approval:
                yield {
                    "type": "approval_required",
                    "agent_name": agent_name,
                    "pending_approvals": pending_approvals,
                }
            continue

        payload = _event_payload(item)
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
            "thinking": event.thinking,
            "is_final": event.is_final,
        }

    if isinstance(event, ModelResponseEvent):
        return {
            "type": "assistant_text",
            "event_type": event.event_type,
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


def _create_session(app: FastAPI) -> dict[str, t.Any]:
    session_id = uuid.uuid4().hex
    contexts = {
        name: RunContext(session_id=session_id) for name in app.state.agents
    }
    app.state.sessions[session_id] = {"contexts": contexts}
    return {
        "session_id": session_id,
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


def _resolve_workspace_root(workspace_root: str | Path | None) -> Path:
    root = (
        Path(workspace_root)
        if workspace_root is not None
        else setting.get_or_create_server_tmp_dir()
    )
    return root.expanduser().resolve()


def _active_workspace_root(app: FastAPI, agent_name: str | None = None) -> Path:
    selected_agent = _select_agent_name(app, agent_name)
    ctx: RunContext = app.state.contexts[selected_agent]
    user_workspace = app.state.workspace_root / ctx.user_id / "workspace"
    user_workspace.mkdir(parents=True, exist_ok=True)
    return user_workspace.resolve()


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
    payload["workspace_root"] = str(_active_workspace_root(app, active_name))
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


def _serialize_messages(messages: t.Sequence[t.Any]) -> list[dict[str, t.Any]]:
    return [_serialize_message(message) for message in messages]


def _serialize_message(message: t.Any) -> dict[str, t.Any]:
    if hasattr(message, "model_dump"):
        data = message.model_dump(mode="json")
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {"role": "assistant", "source": "assistant", "content": str(message)}

    content = data.get("content", "")
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        data["content"] = "".join(text_parts)
    return data


def _last_assistant_message(ctx: RunContext) -> dict[str, t.Any] | None:
    for message in reversed(ctx.messages):
        if getattr(message, "role", None) == "assistant":
            return _serialize_message(message)
    return None


def _context_usage(ctx: RunContext, agent: Agent) -> dict[str, int | None]:
    client = getattr(agent, "client", None)
    config = getattr(client, "config", None)
    tokenizer_base = getattr(config, "tokenizer_base", "o200k_base")
    counter = TokenBudgetStrategy(tokenizer_base=tokenizer_base)
    used = counter.count_messages(ctx.messages)
    return {
        "used": used,
        "max": getattr(config, "max_context_window", None) or None,
    }


def _sse(payload: dict[str, t.Any]) -> str:
    encoded = json.dumps(jsonable_encoder(payload), ensure_ascii=False)
    return f"data: {encoded}\n\n"


def _chat_task(req: ChatRequest) -> str | UserMessage:
    if not req.images:
        return req.message

    parts = [TextPart(text=req.message)]
    for image in req.images:
        try:
            data = base64.b64decode(image.data_base64, validate=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid image payload") from exc
        parts.append(ImagePart(data=data, mime_type=image.mime_type))
    return UserMessage(source="user", content=parts)


def _workspace_files(root: Path) -> list[dict[str, t.Any]]:
    files: list[dict[str, t.Any]] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in WORKSPACE_EXCLUDES and not name.startswith(".")
        ]
        current = Path(current_root)
        if _is_excluded_path(current):
            dirnames[:] = []
            continue
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            path = current / filename
            if _is_excluded_path(path) or not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append(
                {
                    "name": path.name,
                    "path": _workspace_relpath(root, path),
                    "size": stat.st_size,
                    "size_bytes": stat.st_size,
                    "kind": _file_kind(path),
                    "url": f"/api/workspace/raw?path={_workspace_relpath(root, path)}",
                }
            )
            if len(files) >= MAX_WORKSPACE_FILES:
                return files
    return files


def _safe_workspace_path(root: Path, path: str) -> Path:
    workspace_root = root.resolve()
    candidate = (workspace_root / path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if _is_excluded_path(candidate) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate


def _workspace_relpath(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_excluded_path(path: Path) -> bool:
    parts = set(path.parts)
    return any(part in WORKSPACE_EXCLUDES for part in parts)


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


def _read_preview_text(path: Path) -> str:
    with path.open("rb") as handle:
        data = handle.read(MAX_PREVIEW_BYTES + 1)
    truncated = len(data) > MAX_PREVIEW_BYTES
    text = data[:MAX_PREVIEW_BYTES].decode("utf-8", errors="replace")
    if truncated:
        text += "\n\n[Preview truncated]"
    return text
