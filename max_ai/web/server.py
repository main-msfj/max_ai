"""The web UI's server: one page per conversation, its live fragments by SSE,
and the few POSTs the page sends (a message, stop, approvals, answers)."""

from __future__ import annotations

import asyncio
import mimetypes
import posixpath
import typing as t
import webbrowser
from datetime import UTC, datetime
from urllib.parse import quote

from markupsafe import escape
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..base.session_store import CoreSessionStore
from ..core.ids import short_id
from ..types.run_context import RunContext
from .commands import Commands
from .conversation import Conversation
from .view import HERE, Files, History, Html, Requests, Trace
from .workspace import Workspace

if t.TYPE_CHECKING:
    from ..agents.agent import Agent


class WebHost:
    """The agent, where conversations are saved, and the ones open in memory."""

    PING_EVERY = 15  # Seconds between SSE keep-alive comments

    def __init__(self, agent: Agent, store: CoreSessionStore | None, user_id: str) -> None:
        self.agent = agent
        self.store = store
        self.user_id = user_id
        self.html = Html()
        self.conversations: dict[str, Conversation] = {}  # Open ones, by session id
        self.workspace = Workspace(agent, user_id)
        self.commands = Commands(self)
        self.app = Starlette(routes=[
            Route("/", self.index),
            Route("/new", self.new),
            Route("/sessions", self.sessions),
            Route("/s/{sid}", self.page),
            Route("/s/{sid}/events", self.events),
            Route("/s/{sid}/send", self.send, methods=["POST"]),
            Route("/s/{sid}/stop", self.stop, methods=["POST"]),
            Route("/s/{sid}/approve/{rid}", self.approve, methods=["POST"]),
            Route("/s/{sid}/answer/{rid}", self.answer, methods=["POST"]),
            Route("/s/{sid}/trace", self.trace),
            Route("/s/{sid}/span/{span_id}", self.span),
            Route("/s/{sid}/menu", self.menu),
            Route("/s/{sid}/command", self.command),
            Route("/s/{sid}/files", self.files),
            Route("/s/{sid}/file", self.file),
            Route("/s/{sid}/download", self.download),
            Route("/s/{sid}/rules", self.rules, methods=["GET", "POST"]),
            Mount("/static", StaticFiles(directory=HERE / "static"), name="static"),
        ])

    # -------- CONVERSATIONS -----------------------------------------------------------
    async def conversation(self, sid: str) -> Conversation:
        """The open conversation, the saved one, or a new one with that id."""
        if sid in self.conversations:
            return self.conversations[sid]
        ctx = await self.store.load(self.user_id, sid) if self.store is not None else None
        conversation = Conversation(self, ctx or RunContext(user_id=self.user_id, session_id=sid))
        self.conversations[sid] = conversation
        return conversation

    async def save(self, ctx: RunContext) -> None:
        if self.store is None:
            return
        try:
            await self.store.save(ctx)
        except Exception as error:  # noqa: BLE001 — a failed save must not end the turn
            conversation = self.conversations.get(str(ctx.session_id))
            if conversation is not None:
                conversation.publish(conversation.html.append(str(
                    conversation.html.parts.note(f"Could not save the session: {error}", "error")
                )))

    # -------- PAGES -----------------------------------------------------------
    async def index(self, _request: Request) -> Response:
        """The latest conversation, or a new one."""
        if self.store is not None:
            saved = await self.store.list_sessions(self.user_id, limit=1)
            if saved:
                return RedirectResponse(f"/s/{saved[0].session_id}")
        return RedirectResponse(f"/s/{short_id()}")

    async def new(self, _request: Request) -> Response:
        return RedirectResponse(f"/s/{short_id()}")

    async def page(self, request: Request) -> Response:
        conversation = await self.conversation(request.path_params["sid"])
        ctx = conversation.context
        history = History.render(self.html, ctx, self.workspace.folder(conversation.sid))
        if not conversation.busy:
            history += Requests.cards(self.html, conversation.sid, ctx, self.agent.sandbox is not None)
        first = next((m.text() for m in ctx.messages if m.role == "user"), "")
        return HTMLResponse(self.html.page(
            sid=conversation.sid, agent_name=self.agent.name, model=self.agent.client.model,
            description=getattr(self.agent, "description", "") or "",
            title=first.strip().splitlines()[0][:80] if first.strip() else "",
            history=history, status=conversation.status(), usage=conversation.usage(),
            busy=conversation.busy, sandbox=self.sandbox_label(),
            todos=str(self.html.parts.todos(ctx.todos, conversation.busy)), folder=self.workspace.folder(conversation.sid),
        ))

    def sandbox_label(self) -> str:
        sandbox = self.agent.sandbox
        if sandbox is None:
            return "No sandbox"
        return f"{type(sandbox).__name__.removesuffix('Sandbox')} sandbox"

    async def sessions(self, request: Request) -> Response:
        """The sidebar: saved conversations, the newest first."""
        found = await self.store.list_sessions(self.user_id) if self.store is not None else []
        groups: dict[str, list[dict[str, str]]] = {}
        for info in found:
            groups.setdefault(self.day(info.updated_at), []).append({
                "id": info.session_id, "title": info.title or "(untitled)",
                "meta": f"{self.ago(info.updated_at)} · {info.message_count} msgs",
            })
        current = request.query_params.get("current", "")
        return HTMLResponse(str(self.html.parts.sessions(list(groups.items()), current)))

    async def events(self, request: Request) -> Response:
        """The page's live feed: htmx fragments, and the names that refresh panels."""
        conversation = await self.conversation(request.path_params["sid"])
        queue = conversation.listen()

        async def stream() -> t.AsyncGenerator[str]:
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        name, html = await asyncio.wait_for(queue.get(), self.PING_EVERY)
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    data = "".join(f"data: {line}\n" for line in (html.splitlines() or [""]))
                    yield f"event: {name}\n{data}\n"
            finally:
                conversation.unlisten(queue)

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    async def trace(self, request: Request) -> Response:
        conversation = await self.conversation(request.path_params["sid"])
        folder = self.workspace.folder(conversation.sid)
        return HTMLResponse(Trace.render(self.html, conversation.sid, conversation.spans, folder))

    async def span(self, request: Request) -> Response:
        conversation = await self.conversation(request.path_params["sid"])
        span = conversation.spans.spans.get(request.path_params["span_id"])
        if span is None:
            return HTMLResponse("")
        return HTMLResponse(Trace.detail(self.html, span, self.workspace.folder(conversation.sid)))

    # -------- COMMANDS AND FILES -----------------------------------------------------------
    async def menu(self, _request: Request) -> Response:
        """The ``/`` menu: commands and skills."""
        return JSONResponse(await self.commands.menu())

    async def command(self, request: Request) -> Response:
        """A command's panel."""
        conversation = await self.conversation(request.path_params["sid"])
        name = request.query_params.get("name", "").split(" ")[0].lower()
        title, rows = await self.commands.panel(conversation, name)
        return HTMLResponse(str(self.html.parts.panel(title, rows)))

    async def files(self, request: Request) -> Response:
        """The Files pane's tree; ``?format=json`` gives the paths for the ``@`` menu."""
        sid = request.path_params["sid"]
        try:
            found = await self.workspace.files(sid)
        except Exception as error:  # noqa: BLE001 — the sandbox may be down
            if request.query_params.get("format") == "json":
                return JSONResponse([])
            return HTMLResponse(f'<div class="empty">Could not list the files: {escape(str(error))}</div>')
        if request.query_params.get("format") == "json":
            return JSONResponse([path for path, size in found if size >= 0])
        return HTMLResponse(str(self.html.parts.tree(sid, Files.tree(found), len(found))))

    async def file(self, request: Request) -> Response:
        """One file of the workspace, colored, with line numbers."""
        sid, path = request.path_params["sid"], request.query_params.get("path", "")
        try:
            content = await self.workspace.read(sid, path)
        except Exception as error:  # noqa: BLE001 — the sandbox may be down
            content = {"error": str(error)}
        return HTMLResponse(Files.view(self.html, sid, path, content))

    async def download(self, request: Request) -> Response:
        """One file of the workspace, as an attachment."""
        sid, path = request.path_params["sid"], request.query_params.get("path", "")
        try:
            data = await self.workspace.download(sid, path)
        except Exception as error:  # noqa: BLE001 — not a file, too big, or the sandbox is down
            return Response(f"Could not download {path}: {error}", status_code=404, media_type="text/plain")
        name = posixpath.basename(path)
        kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
        disposition = f"attachment; filename*=UTF-8''{quote(name)}"
        return Response(data, media_type=kind, headers={"Content-Disposition": disposition})

    async def rules(self, request: Request) -> Response:
        """The Permissions tab; a POST adds or takes out one rule first."""
        error = ""
        if request.method == "POST":
            form = await request.form()
            error = await self.commands.change_rule(
                str(form.get("action") or "add"), str(form.get("kind") or ""), str(form.get("rule") or ""),
            )
        lists: dict[str, list[str]] = {}
        if self.agent.sandbox is not None:
            try:
                lists = await self.commands.permissions()
            except Exception as failed:  # noqa: BLE001 — the sandbox may be down
                error = error or f"Could not read the rules: {failed}"
        has_rules = self.agent.sandbox is not None
        return HTMLResponse(str(self.html.parts.rules(request.path_params["sid"], lists, has_rules, error)))

    # -------- WHAT THE PAGE SENDS -----------------------------------------------------------
    async def send(self, request: Request) -> Response:
        conversation = await self.conversation(request.path_params["sid"])
        form = await request.form()
        await conversation.send(str(form.get("text") or ""))
        return Response(status_code=204)

    async def stop(self, request: Request) -> Response:
        (await self.conversation(request.path_params["sid"])).stop()
        return Response(status_code=204)

    async def approve(self, request: Request) -> Response:
        conversation = await self.conversation(request.path_params["sid"])
        form = await request.form()
        await conversation.approve(request.path_params["rid"], str(form.get("choice") or "deny"))
        return Response(status_code=204)

    async def answer(self, request: Request) -> Response:
        """``q0``, ``q1``… are the picked options; ``q0_other`` a written answer wins."""
        conversation = await self.conversation(request.path_params["sid"])
        rid = request.path_params["rid"]
        record = conversation.context.tool_state.records.get(rid)
        if record is None:
            return Response(status_code=204)
        form = await request.form()
        answers: dict[str, str | list[str]] = {}
        for index, question in enumerate(Requests.questions(record)):
            other = str(form.get(f"q{index}_other") or "").strip()
            picked = [str(v) for v in form.getlist(f"q{index}")]
            if other:
                answers[question["text"]] = [*picked, other] if question["multi"] else other
            elif picked:
                answers[question["text"]] = picked if question["multi"] else picked[0]
        await conversation.answer(rid, answers)
        return Response(status_code=204)

    @staticmethod
    def day(moment: datetime) -> str:
        """The sidebar group of a conversation: Today, Yesterday, This week or Older."""
        days = (datetime.now(UTC).astimezone().date() - moment.astimezone().date()).days
        return "Today" if days <= 0 else "Yesterday" if days == 1 else "This week" if days < 7 else "Older"

    @staticmethod
    def ago(moment: datetime) -> str:
        seconds = int((datetime.now(UTC) - moment).total_seconds())
        for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
            if seconds >= size:
                return f"{seconds // size}{unit} ago"
        return "just now"


async def run_web(
    agent: Agent,
    *,
    store: CoreSessionStore | None = None,
    user_id: str = "web-user",
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Serve the web UI for an already configured agent until Ctrl+C.

    Like ``run_cli``: with a ``store`` every turn is saved and the sidebar
    lists the saved conversations. The agent is closed when the server ends.
    """
    import uvicorn

    web = WebHost(agent, store, user_id)
    server = uvicorn.Server(uvicorn.Config(web.app, host=host, port=port, log_level="warning"))
    url = f"http://{host}:{port}"
    print(f"MaxAI web UI on {url}  (Ctrl+C to stop)")
    if open_browser:
        asyncio.get_running_loop().call_later(1.0, webbrowser.open, url)
    try:
        await server.serve()
    finally:
        await agent.close()
