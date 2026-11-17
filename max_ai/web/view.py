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
from ..core.tool.display import BASH, exit_code, result_text, tool_summary
from ..types.run_context import RunContext
from ..types.tool_call import ToolCallRecord, ToolResult
from .spans import Span, SpanProjector
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
    """The Trajectory tab: one card per turn, newest first, with its model
    calls and tools on a timeline; a row opens to show what went in and out."""

    MAX_DETAIL = 4000  # Characters of input or output shown in a row

    @staticmethod
    def render(html: Html, sid: str, spans: SpanProjector) -> str:
        return str(html.parts.trace(sid, Trace.data(spans)))

    @staticmethod
    def data(spans: SpanProjector) -> list[dict[str, t.Any]]:
        now = datetime.now(timezone.utc)
        turns: list[dict[str, t.Any]] = []
        for turn in reversed(spans.turns()):
            # 1. The calls of the turn in the order they started; steps are only grouping.
            calls = sorted(
                (s for s in spans.children(turn.id, deep=True) if s.kind != "step"),
                key=lambda s: s.started_at,
            )
            total = max(((turn.ended_at or now) - turn.started_at).total_seconds(), 0.001)
            rows = [Trace._row(span, turn.started_at, total, now) for span in calls]

            # 2. What the turn cost.
            models = [s for s in calls if s.kind == "model"]
            tools = sum(1 for s in calls if s.kind == "tool")
            tokens = sum(s.usage.tokens_input + s.usage.tokens_output for s in models if s.usage)
            stats = [f"{len(models)} model call{'s' if len(models) != 1 else ''}",
                     f"{tools} tool{'s' if tools != 1 else ''}"]
            if tokens:
                stats.append(f"{Trace._k(tokens)} tokens")
            turns.append({
                "id": turn.id, "title": Trace._title(turn.name), "status": turn.status,
                "dur": Trace._seconds(turn.duration_ms, turn.status), "stats": " · ".join(stats),
                "rows": rows,
            })
        return turns

    @staticmethod
    def _row(span: Span, start: datetime, total: float, now: datetime) -> dict[str, t.Any]:
        """One call: its name, a line under it, and its bar inside the turn."""
        left = (span.started_at - start).total_seconds() / total * 100
        seconds = ((span.ended_at or now) - span.started_at).total_seconds()
        width = max(seconds / total * 100, 1.5)
        ttft = span.ttft_ms / 1000 / seconds * 100 if span.ttft_ms and seconds else 0.0
        if span.kind == "model":
            name, sub = span.name.split("/")[-1], []
            if span.ttft_ms is not None:
                sub.append(f"first token {Trace._seconds(span.ttft_ms, 'ok')}")
            if span.usage is not None:
                sub.append(f"↑ {Trace._k(span.usage.tokens_input)} ↓ {Trace._k(span.usage.tokens_output)}")
            line, shown_input = " · ".join(sub), ""
        elif span.kind == "tool":
            params = span.input if isinstance(span.input, dict) else {}
            name = "Bash" if span.name == BASH else span.name
            line = tool_summary(span.name, t.cast("dict[str, t.Any]", params))
            shown_input = Trace._text(span.input)
        else:
            name, line, shown_input = "Compaction", f"{Trace._text(span.input)} tokens before", ""
        return {
            "id": span.id, "kind": span.kind, "status": span.status, "name": name, "line": line,
            "dur": Trace._seconds(span.duration_ms, span.status),
            "left": round(min(left, 98.5), 2), "width": round(min(width, 100 - min(left, 98.5)), 2),
            "ttft": round(min(ttft, 100.0), 1),
            "input": shown_input[: Trace.MAX_DETAIL],
            "output": Trace._text(span.output)[: Trace.MAX_DETAIL],
        }

    @staticmethod
    def _title(name: str) -> str:
        first = name.strip().splitlines()[0] if name.strip() else "turn"
        return first if len(first) <= 80 else first[:79] + "…"

    @staticmethod
    def _k(tokens: int) -> str:
        return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)

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

