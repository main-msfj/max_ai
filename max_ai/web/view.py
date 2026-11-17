"""What the browser sees: events, spans and saved messages turned into HTML.

Every fragment that changes the page while a turn runs is an htmx
out-of-band swap, so one SSE message can append a block, replace another
and update the status line at once.
"""

from __future__ import annotations

import json
import time
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt
from markupsafe import escape
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import Lexer
from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
from pygments.util import ClassNotFound

from ..core.event_type import (
    CompactionEvent,
    CoreEvent,
    ErrorEvent,
    FatalErrorEvent,
    LoopNoticeEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelRetryEvent,
    ModelStreamChunkEvent,
    TodosEvent,
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)
from ..core.harness.permissions import Permissions
from ..core.messages import HARNESS_SOURCE, AssistantMessage, ToolMessage
from ..core.spans import Span, SpanProjector
from ..core.tool.display import BASH, exit_code, result_text, tool_summary
from ..types.run_context import RunContext
from ..types.tool_call import ToolCallRecord, ToolResult
from .workspace import Workspace

HERE = Path(__file__).parent
ASK_USER = "ask_user"
WRITE_TODOS = "write_todos"


class Html:
    """The templates, the markdown renderer and the out-of-band wrappers."""

    CODE_STYLE = "one-dark"
    MAX_OUTPUT = 8000

    def __init__(self) -> None:
        self.env = Environment(
            loader=FileSystemLoader(HERE / "templates"), autoescape=select_autoescape(["html"])
        )
        self.parts = t.cast(t.Any, self.env.get_template("partials.html").module)
        self.markdown = MarkdownIt("commonmark", {"html": False})
        self.markdown.enable(["table", "strikethrough"])
        self.markdown.add_render_rule("fence", self._fence)
        self.markdown.add_render_rule("code_block", self._fence)
        self.code_css = HtmlFormatter(style=self.CODE_STYLE).get_style_defs(".code")

    def page(self, **values: t.Any) -> str:
        # The files' times in their links: a changed file is never taken from the cache.
        version = max(int(f.stat().st_mtime) for f in (HERE / "static").iterdir())
        return self.env.get_template("page.html").render(code_css=self.code_css, version=version, **values)

    def md(self, text: str) -> str:
        return self.markdown.render(text)

    @staticmethod
    def append(html: str, target: str = "#messages") -> str:
        """``html`` added at the end of ``target``."""
        return f'<div hx-swap-oob="beforeend:{target}">{html}</div>'

    @staticmethod
    def replace(element_id: str, inner: str) -> str:
        """The inside of ``#element_id`` replaced by ``inner``."""
        return f'<div id="{element_id}" hx-swap-oob="innerHTML">{inner}</div>'

    @staticmethod
    def swap(html: str) -> str:
        """``html`` takes the place of the element with its id (its root tag carries it)."""
        head, _, rest = html.partition(">")
        return f'{head} hx-swap-oob="outerHTML">{rest}'

    @staticmethod
    def delete(element_id: str) -> str:
        return f'<div id="{element_id}" hx-swap-oob="delete"></div>'

    @staticmethod
    def _fence(_renderer: t.Any, tokens: t.Any, index: int, _options: t.Any, _env: t.Any) -> str:
        """A code block of an answer."""
        token = tokens[index]
        language = str(token.info or "").strip().split(" ")[0]
        try:
            lexer = get_lexer_by_name(language or "text")
        except ClassNotFound:
            lexer = get_lexer_by_name("text")
        return Html.code(str(token.content), lexer, language or "text") + "\n"

    @staticmethod
    def code(text: str, lexer: Lexer, label: str) -> str:
        """Code colored, with a head (its ``label`` and a copy button) and line
        numbers when it has more than one line."""
        colored = highlight(text, lexer, HtmlFormatter(nowrap=True)).rstrip("\n").split("\n")
        if len(colored) > 1:
            width = len(str(len(colored)))
            colored = [
                f'<span class="ln">{n:>{width}}</span>{line}' for n, line in enumerate(colored, 1)
            ]
        head = (
            f'<div class="code-head"><span>{escape(label)}</span>'
            '<button type="button" class="copy">Copy</button></div>'
        )
        return f'<div class="codeblock">{head}<pre class="code"><code>' + "\n".join(colored) + "</code></pre></div>"


@dataclass
class ToolCard:
    """A tool call as one collapsible card."""

    id: str
    name: str
    summary: str
    kind: str  # Colors the argument: path, command, search or plain
    state: str = "running"  # running, ok, error, cancelled
    meta: str = ""
    command: str = ""
    output: str = ""

    @classmethod
    def start(cls, call_id: str, name: str, parameters: dict[str, t.Any]) -> ToolCard:
        is_bash = name == BASH
        return cls(
            id=call_id, name="Bash" if is_bash else name,
            summary=tool_summary(name, parameters), kind=cls.kind_of(parameters),
            command=str(parameters.get("command") or "") if is_bash else "",
        )

    def finish(self, result: ToolResult | None) -> ToolCard:
        if result is None:
            self.state, self.meta = "cancelled", "cancelled"
            return self
        text = result_text(result)
        code = exit_code(result)
        ok = result.success and code in (None, 0)
        self.state = "ok" if ok else "error"
        lines = text.count("\n") + 1
        parts = [f"{lines} line{'s' if lines != 1 else ''}"]
        if code not in (None, 0):
            parts.insert(0, f"exit {code}")
        if result.duration_ms is not None:
            parts.append(f"{result.duration_ms}ms")
        self.meta = " · ".join(parts) if ok or code is not None else "failed"
        if len(text) > Html.MAX_OUTPUT:
            text = text[: Html.MAX_OUTPUT] + f"\n… ({len(text) - Html.MAX_OUTPUT:,} more chars)"
        self.output = text
        return self

    @staticmethod
    def kind_of(parameters: dict[str, t.Any]) -> str:
        if any(key in parameters for key in ("file_path", "path")):
            return "path"
        if "command" in parameters:
            return "command"
        if any(key in parameters for key in ("pattern", "query", "url")):
            return "search"
        return "plain"


class TurnView:
    """The fragments of one conversation as its events come; it remembers the
    answer and the thinking being written, so they can be redrawn whole."""

    FLUSH_EVERY = 0.08  # Seconds between redraws of a streaming answer

    def __init__(self, html: Html, sid: str) -> None:
        self.html, self.sid = html, sid
        self.count = 0  # Blocks made, for unique ids
        self.answer_id: str | None = None  # The answer being written
        self.answer = ""
        self.thought_id: str | None = None  # The thinking being written
        self.thought = ""
        self.thought_started = 0.0
        self.dirty = False  # Text arrived that is not drawn yet
        self.drawn_at = 0.0
        self.cards: dict[str, ToolCard] = {}  # Running tools by call id

    def event(self, event: CoreEvent) -> list[str]:
        """The fragments one event of the run adds or changes."""
        parts = self.html.parts

        # 1. The model writes: thinking and answer are redrawn whole, at most every 80ms.
        if isinstance(event, ModelStreamChunkEvent):
            out: list[str] = []
            if event.thinking:
                if self.thought_id is None:
                    self.thought_id, self.thought = self._id("th"), ""
                    self.thought_started = time.monotonic()
                    out.append(self.html.append(parts.thinking(self.thought_id, "", False, "")))
                self.thought += event.thinking
                self.dirty = True
            # A bare "\n\n" before a tool call would open an empty answer.
            if event.chunk and not event.is_final and (self.answer_id or event.chunk.strip()):
                out += self._close_thought()
                if self.answer_id is None:
                    self.answer_id, self.answer = self._id("a"), ""
                    out.append(self.html.append(parts.assistant(self.answer_id, "")))
                self.answer += event.chunk
                self.dirty = True
            return out + self.flush()
        if isinstance(event, ModelRetryEvent):
            note = f"Model call failed ({event.reason}); retrying in {event.delay:.0f}s."
            out = self.flush(force=True)
            if event.discard_partial and self.answer_id is not None:
                out.append(self.html.delete(self.answer_id))
                self.answer_id, self.answer = None, ""
            return out + [self.html.append(parts.note(note, "warn"))]
        if isinstance(event, ModelCallEvent):
            return self.finish()
        if isinstance(event, ModelResponseEvent):
            out = self.flush(force=True)
            if event.response and self.answer_id is None and self.answer == "":
                self.answer_id = self._id("a")
                self.answer = event.response
                out.append(self.html.append(parts.assistant(self.answer_id, self.html.md(event.response))))
            return out + self.finish()

        # 2. Tools: a card that runs, then shows how it ended.
        if isinstance(event, ToolCallEvent):
            out = self.finish()
            if event.tool_name in (ASK_USER, WRITE_TODOS):
                return out
            card = ToolCard.start(event.tool_call_id, event.tool_name, event.parameters)
            # A call that waited for approval is announced again: its card is reused.
            seen = card.id in self.cards
            self.cards[card.id] = card
            drawn = str(parts.tool(card))
            return out + [self.html.swap(drawn) if seen else self.html.append(drawn)]
        if isinstance(event, ToolApprovalEvent):
            card = self.cards.get(event.tool_call_id)
            if card is None:
                return []
            card.state, card.meta = "waiting", "waiting for you"
            return [self.html.swap(str(parts.tool(card)))]
        if isinstance(event, TodosEvent):
            return [self.html.replace("todos", str(parts.todos(event.todos)))]
        if isinstance(event, ToolCallResponseEvent):
            card = self.cards.pop(event.tool_call_id, None)
            if card is None:
                return []
            return [self.html.swap(str(parts.tool(card.finish(event.tool_result))))]

        # 3. Compaction, harness rules and errors, as notes.
        if isinstance(event, CompactionEvent) and event.phase == "end" and event.changed:
            note = f"Context compacted: {event.tokens_before:,} → {event.tokens_after:,} tokens"
            return self.finish() + [self.html.append(parts.note(note))]
        if isinstance(event, LoopNoticeEvent) and event.message:
            return [self.html.append(parts.note(f"harness · {event.kind}", "warn"))]
        if isinstance(event, (ErrorEvent, FatalErrorEvent)):
            return self.finish() + [self.html.append(parts.note(event.error_message, "error"))]
        return []

    def flush(self, force: bool = False) -> list[str]:
        """Redraw the thinking and the answer if text arrived since the last time."""
        if not self.dirty or (not force and time.monotonic() - self.drawn_at < self.FLUSH_EVERY):
            return []
        self.dirty, self.drawn_at = False, time.monotonic()
        out: list[str] = []
        if self.thought_id is not None:
            thinking = self.html.parts.thinking(self.thought_id, self.thought, False, "")
            out.append(self.html.swap(str(thinking)))
        if self.answer_id is not None:
            out.append(self.html.replace(self.answer_id, self.html.md(self.answer)))
        return out

    def finish(self) -> list[str]:
        """The model stopped writing: last redraw, the thinking folds."""
        out = self.flush(force=True) + self._close_thought()
        self.answer_id, self.answer = None, ""
        return out

    def cancel_cards(self) -> list[str]:
        """Tools cut by a stop end as cancelled."""
        out = [self.html.swap(str(self.html.parts.tool(c.finish(None)))) for c in self.cards.values()]
        self.cards.clear()
        return out

    def _close_thought(self) -> list[str]:
        if self.thought_id is None:
            return []
        took = time.monotonic() - self.thought_started
        took_text = f"{took:.0f}s" if took >= 1 else "<1s"
        html = self.html.parts.thinking(self.thought_id, self.thought.strip(), True, took_text)
        self.thought_id, self.thought = None, ""
        return [self.html.swap(str(html))]

    def _id(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}-{int(time.time() * 1000) % 10_000_000}-{self.count}"


class Requests:
    """The approval and question cards of the calls that wait for the user."""

    @staticmethod
    def cards(html: Html, sid: str, ctx: RunContext, can_keep: bool) -> str:
        out: list[str] = []
        for record in ctx.tool_state.records.values():
            if record.is_consumed:
                continue
            if record.is_pending_approval:
                out.append(str(html.parts.approval(sid, Requests.approval(record, can_keep))))
            elif record.is_awaiting_input:
                out.append(str(html.parts.questions(sid, record.id, Requests.questions(record))))
        return "".join(out)

    @staticmethod
    def approval(record: ToolCallRecord, can_keep: bool) -> dict[str, t.Any]:
        params = record.parameters
        if record.tool_name == BASH:
            shown = str(params.get("command") or "")
        else:
            shown = json.dumps(params, indent=2, ensure_ascii=False, default=str)
        keep = can_keep and Permissions.can_be_kept(record.tool_name, params)
        return {
            "id": record.id, "name": record.tool_name, "params": shown,
            "bash": record.tool_name == BASH, "reason": str(params.get("description") or ""),
            "rules": Permissions.rules_for(record.tool_name, params) if keep else [],
        }

    @staticmethod
    def questions(record: ToolCallRecord) -> list[dict[str, t.Any]]:
        items = t.cast("list[dict[str, t.Any]]", record.parameters.get("questions") or [])
        return [
            {
                "text": item["question"], "header": item.get("header") or "",
                "multi": bool(item.get("multi_select")),
                "options": [
                    {"label": o["label"], "description": o.get("description", "")}
                    for o in t.cast("list[dict[str, str]]", item.get("options") or [])
                ],
            }
            for item in items
        ]


class History:
    """A saved conversation drawn again from its messages (no model calls)."""

    @staticmethod
    def render(html: Html, ctx: RunContext) -> str:
        parts = html.parts
        results = {m.tool_call_id: m for m in ctx.messages if isinstance(m, ToolMessage)}
        out: list[str] = []
        if ctx.summary is not None:
            out.append(str(parts.note("Earlier messages were compacted into a summary.")))
        for message in ctx.messages:
            if message.source == HARNESS_SOURCE or message.role in ("system", "tool"):
                continue
            if message.role == "user":
                out.append(str(parts.user(message.text())))
                continue
            if not isinstance(message, AssistantMessage):
                continue
            if message.text().strip():
                out.append(str(parts.assistant(f"h-{len(out)}", html.md(message.text()))))
            for call in message.tool_calls:
                if call.tool_name in (ASK_USER, WRITE_TODOS):
                    continue
                card = ToolCard.start(call.id, call.tool_name, call.parameters)
                done = results.get(call.id)
                if done is None:
                    card.state, card.meta = "cancelled", "no result"
                else:
                    card.state = "ok" if done.success else "error"
                    card.output = done.text()[: Html.MAX_OUTPUT]
                    card.meta = "done" if done.success else "failed"
                out.append(str(parts.tool(card)))
        return "".join(out)


class Files:
    """The Files pane: the workspace as a tree, and one file open."""

    @staticmethod
    def tree(found: list[tuple[str, int]]) -> list[dict[str, t.Any]]:
        """Nested nodes (``name``, ``path``, ``size``, ``children``), folders first."""
        root: dict[str, t.Any] = {"children": {}}
        for path, size in found:
            node = root
            names = path.split("/")
            for depth, name in enumerate(names[:-1], 1):
                folder = {"children": {}, "size": -1, "name": name, "path": "/".join(names[:depth])}
                node = node["children"].setdefault(name, folder)
            leaf = node["children"].setdefault(names[-1], {"children": {}})
            leaf.update(name=names[-1], path=path, size=size)

        def ordered(node: dict[str, t.Any]) -> list[dict[str, t.Any]]:
            children = list(node["children"].values())
            for child in children:
                child["children"] = ordered(child)
                child["label"] = "" if child["size"] < 0 else Files.size(child["size"])
            return sorted(children, key=lambda c: (c["size"] >= 0, c.get("name", "").lower()))

        return ordered(root)

    @staticmethod
    def view(html: Html, path: str, content: dict[str, t.Any]) -> str:
        if "error" in content:
            body = f'<div class="empty">{escape(content["error"])}</div>'
        elif content.get("binary"):
            body = '<div class="empty">A binary file: it is not shown.</div>'
        else:
            try:
                lexer = get_lexer_for_filename(path, stripnl=False)
            except ClassNotFound:
                lexer = get_lexer_by_name("text")
            body = Html.code(str(content["text"]) or " ", lexer, lexer.name)
            if content.get("cut"):
                body += f'<div class="empty">Only the first {Workspace.MAX_READ // 1000} kB are shown.</div>'
        size = Files.size(int(content.get("size", 0)))
        return str(html.parts.file_view(path, size, body))

    @staticmethod
    def size(size: int) -> str:
        if size < 1000:
            return f"{size} B"
        return f"{size / 1000:.1f} kB" if size < 1_000_000 else f"{size / 1_000_000:.1f} MB"


class Trace:
    """The Trajectory tab: a timeline of the spans and a ledger, one row each."""

    LANES = {"model": 6, "compaction": 6, "tool": 32}  # y of each kind in the overview

    @staticmethod
    def render(html: Html, sid: str, spans: SpanProjector, turn_id: str | None) -> str:
        return str(html.parts.trace(sid, Trace.data(spans, turn_id)))

    @staticmethod
    def data(spans: SpanProjector, turn_id: str | None) -> dict[str, t.Any]:
        turns = spans.turns()
        chosen = [s for s in turns if s.id == turn_id] or turns
        rows: list[tuple[Span, int]] = []
        for turn in chosen:
            Trace._walk(spans, turn, 0, rows)
        now = datetime.now(timezone.utc)
        if not rows:
            return {"turns": turns, "turn": turn_id, "rows": [], "marks": [], "total": "", "stats": ""}

        # 1. The time the rows cover, left to right.
        start = min(span.started_at for span, _ in rows)
        end = max(span.ended_at or now for span, _ in rows)
        total = max((end - start).total_seconds(), 0.001)

        def at(moment: datetime) -> float:
            return (moment - start).total_seconds() / total * 100

        # 2. One row per span, with its bar and, for the model, the wait to its first token.
        drawn: list[dict[str, t.Any]] = []
        for span, depth in rows:
            left = at(span.started_at)
            width = max(at(span.ended_at or now) - left, 0.4)
            ttft = span.ttft_ms
            seconds = (span.duration_ms or 0) / 1000 or (now - span.started_at).total_seconds()
            ttft_pct = min(100.0, ttft / 1000 / seconds * 100) if ttft and seconds else 0.0
            drawn.append({
                "id": span.id, "kind": span.kind, "name": Trace._label(span), "depth": depth,
                "status": span.status, "dur": Trace._seconds(span.duration_ms, span.status),
                "left": round(left, 3), "width": round(width, 3), "ttft_pct": round(ttft_pct, 1),
                "lane": Trace.LANES.get(span.kind), "x": round(left * 10, 2),
                "w": round(width * 10, 2), "ttft_w": round(width * 10 * ttft_pct / 100, 2),
            })

        # 3. Where each turn starts, and what the scope cost.
        marks = [round(at(turn.started_at) * 10, 2) for turn in chosen]
        models = [s for s, _ in rows if s.kind == "model"]
        tokens_in = sum(s.usage.tokens_input for s in models if s.usage)
        tokens_out = sum(s.usage.tokens_output for s in models if s.usage)
        tools = sum(1 for s, _ in rows if s.kind == "tool")
        stats = f"{len(models)} model calls · {tools} tools · ↑ {tokens_in:,} ↓ {tokens_out:,} tokens"
        return {
            "turns": turns, "turn": turn_id if turn_id in {s.id for s in turns} else None,
            "rows": drawn, "marks": marks, "total": Trace._seconds(int(total * 1000), "ok"),
            "stats": stats,
        }

    @staticmethod
    def inspect(html: Html, span: Span) -> str:
        facts: list[tuple[str, str]] = [
            ("started", span.started_at.astimezone().strftime("%H:%M:%S.%f")[:-3]),
            ("duration", Trace._seconds(span.duration_ms, span.status)),
        ]
        if span.ttft_ms is not None:
            facts.append(("first token", Trace._seconds(span.ttft_ms, "ok")))
        if span.retries:
            facts.append(("retries", str(span.retries)))
        if span.usage is not None:
            facts.append(("tokens in", f"{span.usage.tokens_input:,}"))
            facts.append(("tokens out", f"{span.usage.tokens_output:,}"))
            if span.usage.tokens_cached:
                facts.append(("cached", f"{span.usage.tokens_cached:,}"))
        if span.kind == "model" and isinstance(span.input, int):
            facts.append(("messages sent", str(span.input)))
        shown_input = "" if span.kind == "model" else Trace._text(span.input)
        return str(html.parts.inspector({
            "kind": span.kind, "name": span.name, "status": span.status, "facts": facts,
            "input": shown_input, "output": Trace._text(span.output)[: Html.MAX_OUTPUT],
        }))

    @staticmethod
    def _walk(spans: SpanProjector, span: Span, depth: int, rows: list[tuple[Span, int]]) -> None:
        rows.append((span, depth))
        for child in spans.children(span.id):
            Trace._walk(spans, child, depth + 1, rows)

    @staticmethod
    def _label(span: Span) -> str:
        if span.kind == "turn":
            name = span.name.strip().splitlines()[0] if span.name.strip() else "turn"
            return name if len(name) <= 60 else name[:59] + "…"
        return span.name

    @staticmethod
    def _seconds(ms: int | None, status: str) -> str:
        if ms is None:
            return "running" if status == "running" else "—"
        return f"{ms}ms" if ms < 1000 else f"{ms / 1000:.1f}s"

    @staticmethod
    def _text(value: t.Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)

