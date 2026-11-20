"""One conversation of the web UI: its context, the turn that runs and the
pages that follow it. The same steps as the CLI's turn, with the browser
answering approvals and questions by HTTP instead of a prompt."""

from __future__ import annotations

import asyncio
import re
import time
import typing as t
from datetime import UTC, datetime

from markupsafe import escape

from ..base.loop import LoopState
from ..capabilities.loop import ReactLoop
from ..capabilities.tools.ask_user import AskUserTool
from ..capabilities.tools.todos import WriteTodosTool
from ..core.event_type import (
    CoreEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelStreamChunkEvent,
    ToolCallResponseEvent,
)
from ..core.harness import messages as harness
from ..core.harness.permissions import Permissions
from ..core.messages import HARNESS_SOURCE, HARNESS_TAG, UserMessage
from ..core.tool import ToolDispatcher
from ..core.tool.display import BASH
from ..types.run_context import RunContext
from ..types.tool_call import ToolResult
from .spans import SpanProjector
from .view import PLAN_OPTIONS, WRITE_TODOS, Html, Requests, ToolCard, TurnView
from .workspace import Workspace

if t.TYPE_CHECKING:
    from .server import WebHost

Fragment = tuple[str, str]  # (SSE event name, HTML)
MENTION = re.compile(r"(?<![\w@])@([\w./-]+[\w/])")  # @path, without a trailing dot


class Conversation:
    """A session as the browser sees it. Every page open on it gets the same
    fragments; the turn goes on when the page that started it is closed."""

    def __init__(self, host: WebHost, ctx: RunContext) -> None:
        self.host = host
        self.context = ctx
        self.html: Html = host.html
        self.view = TurnView(host.html, self.sid)
        self.view.folder = host.workspace.folder(self.sid)
        self.spans = SpanProjector()
        self.listeners: set[asyncio.Queue[Fragment]] = set()  # One queue per open page
        self.task: asyncio.Task[None] | None = None  # The turn that runs
        self.started_at = 0.0  # time.time() the turn started, for the page's timer
        self.tokens_in = 0
        self.tokens_out = 0

    @property
    def sid(self) -> str:
        return str(self.context.session_id)

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def waiting(self) -> bool:
        """Calls wait for the user: approvals or questions."""
        return any(
            r.is_pending_approval or r.is_awaiting_input
            for r in self.context.tool_state.records.values()
        )

    # -------- PAGES -----------------------------------------------------------
    def listen(self) -> asyncio.Queue[Fragment]:
        queue: asyncio.Queue[Fragment] = asyncio.Queue()
        self.listeners.add(queue)
        return queue

    def unlisten(self, queue: asyncio.Queue[Fragment]) -> None:
        self.listeners.discard(queue)

    def publish(self, *html: str, event: str = "message") -> None:
        body = "".join(html)
        if not body and event == "message":
            return
        for queue in self.listeners:
            queue.put_nowait((event, body))

    def status(self) -> str:
        mode = "busy" if self.busy else "waiting" if self.waiting else ""
        return str(self.html.parts.status(mode, int(self.started_at * 1000)))

    def usage(self) -> str:
        """Tokens spent, and the model's window part by part as the last call left it."""
        window = self.host.agent.client.config.max_context_window
        parts = self.context.context
        if parts is not None and parts.window:
            window = parts.window
        sizes = [
            ("System prompt", parts.prompt if parts else 0, "c-prompt"),
            ("Tools", parts.tools if parts else 0, "c-tools"),
            ("Messages", parts.messages if parts else 0, "c-messages"),
        ]
        used = sum(size for _, size, _ in sizes)
        ratio = min(1.0, used / window) if window else 0.0

        def share(size: int) -> float:
            return round(min(100.0, size / window * 100), 1) if window else 0.0

        return str(self.html.parts.usage({
            "tokens_in": self.k(self.tokens_in), "tokens_out": self.k(self.tokens_out),
            "used": self.k(used), "window": self.k(window), "percent": round(ratio * 100),
            "level": "high" if ratio >= 0.85 else "mid" if ratio >= 0.6 else "low",
            "parts": [
                {"name": name, "tokens": self.k(size), "share": share(size), "kind": kind}
                for name, size, kind in sizes
            ],
            "free": self.k(max(window - used, 0)), "free_share": round(max(0.0, 100 - ratio * 100), 1),
        }))

    def _show_state(self) -> None:
        self.publish(self.html.replace("status", self.status()), self.html.replace("usage", self.usage()))

    # -------- WHAT THE USER DOES -----------------------------------------------------------
    async def send(self, text: str) -> None:
        """A message: a new turn, or read by the running one at its next model call.
        ``!command`` runs in the workspace, ``/skill`` asks for that skill and
        each ``@path`` attaches that file of the workspace."""
        text = text.strip()
        if not text:
            return
        if text.startswith("!"):
            await self.shell(text[1:].strip())
            return
        # The first message names the conversation.
        if not any(m.role == "user" for m in self.context.messages):
            self.publish(self.html.replace("title", str(escape(text.splitlines()[0][:80]))))
        self.publish(self.html.delete("hero"), self.html.append(str(self.html.parts.user(text))))
        task = await self._task(text)
        if self.busy:
            self.context.inbox.append(task)
            return
        # Calls still waiting are closed by the agent: the message answers instead.
        for record in self.context.tool_state.records.values():
            if not record.is_consumed:
                self.publish(self.html.delete(f"req-{record.id}"))
        self.publish(*self.view.cancel_cards())
        self.start(task)

    async def _task(self, text: str) -> str:
        """What the model reads: a skill's command spelled out, mentions attached."""
        command, _, rest = text.partition(" ")
        skills = await self.host.commands.load_skills()
        if command.startswith("/") and command[1:] in {name for name, _ in skills}:
            text = f'Use the "{command[1:]}" skill.' + (f" {rest.strip()}" if rest.strip() else "")
        found = [m for m in MENTION.findall(text) if Workspace.relative(m)]
        return await self.host.workspace.mentions(self.sid, text, found) if found else text

    async def shell(self, command: str) -> None:
        """A ``!command`` of the user: run in the workspace, shown, not sent to the model."""
        if not command or self.busy:
            return
        self.publish(self.html.delete("hero"))
        card = ToolCard.start(f"sh-{int(time.time() * 1000)}", BASH, {"command": command})
        card.summary, card.kind = "your command", "plain"
        self.publish(self.html.append(str(self.html.parts.tool(card))))
        started = datetime.now(UTC)
        try:
            ran = await self.host.workspace.shell(self.sid, command)
            output = {"stdout": ran.stdout, "stderr": ran.stderr, "exit_code": ran.exit_code}
            result = ToolResult.success_result(card.id, output)
        except Exception as error:  # noqa: BLE001 — shown in the card
            result = ToolResult(success=False, error=str(error), tool_call_id=card.id)
        result = result.model_copy(update={"started_at": started, "completed_at": datetime.now(UTC)})
        self.publish(self.html.swap(str(self.html.parts.tool(card.finish(result), True))))
        self.publish("", event="files")

    def stop(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()

    async def approve(self, record_id: str, choice: str) -> None:
        """Allow once, always, or deny one call; the turn goes on once none waits."""
        agent, ctx = self.host.agent, self.context
        record = ctx.tool_state.records.get(record_id)
        if record is None or record.is_consumed or self.busy:
            return
        self.publish(self.html.delete(f"req-{record_id}"))
        # The card's diamond says how it went: no note in the chat, unless the rule was lost.
        if choice == "deny":
            record.reject()
        else:
            always = choice == "always"
            try:
                await agent.approve(ctx, record_id, always=always)
            except Exception as error:  # noqa: BLE001 — the rule could not be written
                self.publish(self._note(f"Approved once; the rule was not kept: {error}", "warn"))
            else:
                kept = Permissions.rules_for(record.tool_name, record.parameters) if always else []
                if kept:
                    self.host.commands.rules = None  # the Permissions tab shows the new rule
                    await self._approve_covered(kept)
        self._go_on()

    async def answer(self, record_id: str, answers: dict[str, str | list[str]]) -> None:
        """The answers of a call that waits, by question text."""
        record = self.context.tool_state.records.get(record_id)
        if record is None or record.is_consumed or self.busy:
            return
        if record.tool_name == WRITE_TODOS:
            # A plan: accepted, or not, with what the user typed instead.
            said = next(iter(answers.values()), None)
            comment = None if said in (None, *PLAN_OPTIONS) else str(said)
            accepted = said == PLAN_OPTIONS[0]
            record.store_result(WriteTodosTool.answer(record, self.context, accepted, comment))
            if accepted:
                todos = self.html.parts.todos(self.context.todos, False)
                self.publish(self.html.replace("todos", str(todos)))
        else:
            record.store_result(AskUserTool.answer(record, answers))
        self.publish(self.html.delete(f"req-{record_id}"))
        self._go_on()

    async def _approve_covered(self, kept: list[str]) -> None:
        """Other calls of the batch the rule just kept also covers."""
        rules = Permissions({"allow": kept})
        for record in list(self.context.tool_state.records.values()):
            if record.is_pending_approval and rules.decide(record.tool_name, record.parameters) is True:
                await self.host.agent.approve(self.context, record.id)
                self.publish(self.html.delete(f"req-{record.id}"))

    def _go_on(self) -> None:
        if not self.waiting and not self.busy:
            self.start(None)
        else:
            self._show_state()

    # -------- THE TURN -----------------------------------------------------------
    def start(self, task: str | None) -> None:
        self.task = asyncio.create_task(self._turn(task))

    async def _turn(self, task: str | None) -> None:
        """One turn of the user, until it ends or waits for them."""
        self.started_at = time.time()
        self.spans.start_turn(task or "↳ after your answer")
        self._show_state()
        try:
            while True:
                state = await self._run_once(task)
                if state.finish_reason == ReactLoop.WAITING:
                    # Written while the turn ran: the agent closes the calls and reads it.
                    if self.context.inbox:
                        task = None
                        continue
                    can_keep = self.host.agent.sandbox is not None
                    self.publish(self.html.append(Requests.cards(self.html, self.sid, self.context, can_keep)))
                    break
                self.publish(self.html.append(self._turn_end(state)))
                # Written after the loop's last look: one more run reads it.
                if not self.context.inbox:
                    break
                task = None
        except asyncio.CancelledError:
            self._close_interrupted()
        except Exception as error:  # noqa: BLE001 — a failed turn must not kill the server
            self.publish(*self.view.finish(), self._note(f"Failed: {error}", "error"))
            self.spans.end_turn("error")
        finally:
            self.publish(*self.view.finish())
            # Tasks left when the turn ends fold to one line; all done, they go.
            self.publish(self.html.replace("todos", str(self.html.parts.todos(self.context.todos, False))))
            self.task = None
            await self.host.save(self.context)
            self._show_state()
            self.publish("", event="spans")
            self.publish("", event="saved")

    async def _run_once(self, task: str | None) -> LoopState:
        state = LoopState()
        answered = False  # Tools answered since the last save
        events = self.host.agent.run_stream_events(task, self.context, state=state, stream=True)
        async for event in events:
            # Saved at the model call after tools: by then every answer is in ctx.
            if isinstance(event, ToolCallResponseEvent):
                answered = True
            elif isinstance(event, ModelCallEvent) and answered:
                answered = False
                await self.host.save(self.context)
            self._draw(event)
        return state

    def _draw(self, event: CoreEvent) -> None:
        """The event on the page, in the trace and in the token count."""
        self.publish(*self.view.event(event))
        if isinstance(event, ToolCallResponseEvent):
            self.publish("", event="files")  # The tool may have changed the files
        changed = self.spans.feed(event)
        if changed is not None and not isinstance(event, ModelStreamChunkEvent):
            self.publish("", event="spans")
        if isinstance(event, ModelResponseEvent) and event.usage is not None:
            self.tokens_in += event.usage.tokens_input
            self.tokens_out += event.usage.tokens_output
            self.publish(self.html.replace("usage", self.usage()))

    def _close_interrupted(self) -> None:
        """Stopped: every open call gets its answer and the model is told."""
        ToolDispatcher.close_interrupted(self.context)
        self.context.messages.append(
            UserMessage(source=HARNESS_SOURCE, content=f"{HARNESS_TAG} {harness.TURN_INTERRUPTED}")
        )
        self.spans.end_turn("cancelled")
        self.publish(
            *self.view.finish(), *self.view.cancel_cards(),
            self._note("Stopped · what ran so far stays in the conversation.", "warn"),
        )

    def _turn_end(self, state: LoopState) -> str:
        elapsed = int(time.time() - self.started_at)
        calls = f"{state.iteration} model call{'s' if state.iteration != 1 else ''}"
        details = f"· {calls}" + (f" · compacted {state.compactions}x" if state.compactions else "")
        if state.finish_reason == "stop":
            return str(self.html.parts.turn_end(True, f"Done in {elapsed}s", details))
        reason = state.finish_reason.replace("_", " ")
        return str(self.html.parts.turn_end(False, f"Stopped after {elapsed}s", f"· {reason} {details}"))

    def _note(self, text: str, kind: str = "") -> str:
        return self.html.append(str(self.html.parts.note(text, kind)))

    @staticmethod
    def k(tokens: int) -> str:
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:g}M"
        if tokens >= 100_000:
            return f"{tokens / 1000:.0f}k"
        return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)
