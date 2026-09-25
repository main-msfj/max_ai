"""Transcript blocks: each turn piece is a widget, so it can collapse and update."""

from __future__ import annotations

import json
import re
import time
import typing as t
from pathlib import Path

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Markdown, Static
from textual.widgets._markdown import MarkdownStream

from ..types.tool_call import ToolResult

# MaxAI look: neutral text and surfaces; lilac only for accents (glyphs,
# spinner, focus/hover, command names).
SPINNER = "◇◈◆◈"
ACCENT = "#b794f6"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


class WelcomeBox(Static):
    """Session banner: who is running, on what model, in which workspace."""

    DEFAULT_CSS = f"""
    WelcomeBox {{
        border: round {ACCENT}; padding: 0 1; margin: 1 0 1 0; height: auto; width: auto;
        max-width: 90;
    }}
    """

    def __init__(self, agent_name: str, model: str, workspace: Path) -> None:
        body = Text()
        body.append("◆ ", style=f"bold {ACCENT}")
        body.append("M A X · A I", style=f"bold {ACCENT}")
        body.append("   agent workspace", style="#71717a")
        body.append(f"\n\n  agent      {agent_name}", style="#a1a1aa")
        body.append(f"\n  model      {model}", style="#a1a1aa")
        body.append(f"\n  workspace  {workspace}", style="#a1a1aa")
        body.append("\n\n  /help commands · esc interrupt · ctrl+t thinking", style="dim")
        super().__init__(body)


class UserBlock(Static):
    DEFAULT_CSS = """
    UserBlock {
        background: #1c1c1f; color: #e4e4e7; padding: 0 1; margin: 1 0 0 0; height: auto;
        border-left: outer #b794f6;
    }
    """

    def __init__(self, text: str) -> None:
        body = Text("❯ ", style="bold #b794f6")
        body.append(text)
        super().__init__(body)


class NoteLine(Static):
    """One dim harness line (gate retries, done marker, errors, verbose events)."""

    DEFAULT_CSS = """
    NoteLine { height: auto; padding: 0 0 0 2; color: #71717a; }
    """

    def __init__(self, text: str | Text, style: str = "") -> None:
        super().__init__(Text(text, style=style) if isinstance(text, str) else text)


class AssistantBlock(Horizontal):
    """``✦`` + streamed Markdown. Streaming goes through MarkdownStream so a long
    answer isn't re-parsed from scratch on every chunk."""

    DEFAULT_CSS = """
    AssistantBlock { height: auto; margin: 1 0 0 0; }
    AssistantBlock > .bullet { width: 2; color: #b794f6; }
    AssistantBlock > Markdown { width: 1fr; margin: 0; padding: 0; background: transparent; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.text = ""
        self._stream: MarkdownStream | None = None

    def compose(self):
        yield Static("✦", classes="bullet")
        yield Markdown("")

    async def write(self, chunk: str) -> None:
        self.text += chunk
        if self._stream is None:
            self._stream = Markdown.get_stream(self.query_one(Markdown))
        await self._stream.write(chunk)

    async def finish(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None


class FoldBlock(Vertical):
    """A box that folds to its one-line header; click toggles it.

    Subclasses render ``_header()`` and ``_body()``; ``foldable`` is False
    while the content is still live (streaming thinking, pending question).
    """

    DEFAULT_CSS = """
    FoldBlock {
        height: auto; border: round #3f3f46; padding: 0 1; margin: 1 0 0 0;
    }
    FoldBlock:hover { border: round #b794f6; }
    FoldBlock > .header { color: #a1a1aa; }
    FoldBlock > .body { color: #d4d4d8; max-height: 16; overflow-y: auto; }
    FoldBlock.-collapsed { width: auto; }
    FoldBlock.-collapsed > .header { width: auto; }
    FoldBlock.-collapsed > .body { display: none; }
    """

    foldable = True

    def compose(self):
        yield Static(self._header(), classes="header")
        yield Static(self._body(), classes="body")

    def _header(self) -> Text:
        raise NotImplementedError

    def _body(self) -> Text:
        raise NotImplementedError

    def _fold_hint(self, header: Text) -> Text:
        if self.foldable:
            arrow = "▸ expand" if self.has_class("-collapsed") else "▾ collapse"
            header.append(f"   {arrow}", style="dim")
        return header

    def on_mount(self) -> None:
        self.refresh_block()

    def refresh_block(self) -> None:
        headers, bodies = self.query(".header"), self.query(".body")
        if not headers or not bodies:
            return
        headers.first(Static).update(self._header())
        bodies.first(Static).update(self._body())

    def collapse(self, collapsed: bool = True) -> None:
        self.set_class(collapsed, "-collapsed")
        self.refresh_block()

    def on_click(self) -> None:
        if self.foldable:
            self.collapse(not self.has_class("-collapsed"))


class ThinkingBlock(FoldBlock):
    """Reasoning in a box: live while streaming, then folds to one line.

    Hidden entirely when thinking is off, but still recorded, so turning it
    back on reveals past thoughts.
    """

    DEFAULT_CSS = """
    ThinkingBlock > .body { color: #8b8b94; text-style: italic; max-height: 12; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.text = ""
        self.done = False
        self._started = time.monotonic()
        self._elapsed = 0.0

    @property
    def foldable(self) -> bool:  # type: ignore[override]
        return self.done

    def _header(self) -> Text:
        header = Text("◈ ", style=ACCENT)
        if not self.done:
            header.append("Thinking…", style="italic")
            return header
        words = len(self.text.split())
        took = f"{self._elapsed:.0f}s" if self._elapsed >= 1 else "<1s"
        header.append(f"Thought for {took}", style="bold #a1a1aa")
        header.append(f" · {_plural(words, 'word')}", style="dim")
        return self._fold_hint(header)

    def _body(self) -> Text:
        return Text(self.text.strip())

    def append(self, chunk: str) -> None:
        self.text += chunk
        self.refresh_block()
        bodies = self.query(".body")
        if bodies:
            bodies.first(Static).scroll_end(animate=False)

    def finish(self) -> None:
        if self.done:
            return
        self.done = True
        self._elapsed = time.monotonic() - self._started
        self.collapse()


_PLAN_STYLES = {
    "done": ("☒", "#71717a strike"),
    "active": ("▶", f"bold {ACCENT}"),
    "failed": ("✗", "#f87171"),
    "pending": ("☐", "#d4d4d8"),
}


class PlanBlock(FoldBlock):
    """The agent's plan as a checklist; folds to progress + current step."""

    def __init__(self, plan: t.Any) -> None:
        super().__init__()
        self.plan = plan

    def update_plan(self, plan: t.Any) -> None:
        self.plan = plan
        self.refresh_block()

    def _header(self) -> Text:
        steps = list(self.plan.steps)
        done = sum(step.status == "done" for step in steps)
        header = Text("◇ ", style=f"bold {ACCENT}")
        header.append(f"Plan {done}/{len(steps)}", style="bold #e4e4e7")
        current = next((s for s in steps if s.status == "active"), None)
        current = current or next((s for s in steps if s.status == "pending"), None)
        if done == len(steps) and steps:
            header.append(" · all done", style="#4ade80")
        elif current is not None:
            header.append(f" · {current.description[:60]}", style="#a1a1aa")
        return self._fold_hint(header)

    def _body(self) -> Text:
        body = Text()
        for index, step in enumerate(self.plan.steps):
            icon, style = _PLAN_STYLES.get(step.status, ("·", ""))
            if index:
                body.append("\n")
            body.append(f"{icon} {step.description}", style=style)
        return body


def context_bar(used: int, maximum: int, cells: int = 10) -> Text:
    """``▓▓▓▓░░░░░░ 52k / 128k``: green, amber from 60% used, red from 85%."""
    ratio = min(1.0, used / maximum) if maximum > 0 else 0.0
    color = "#f87171" if ratio >= 0.85 else "#fbbf24" if ratio >= 0.6 else "#4ade80"
    filled = round(ratio * cells)
    bar = Text("▓" * filled, style=color)
    bar.append("░" * (cells - filled), style="#3f3f46")
    bar.append(f" {_k(used)} / {_k(maximum)}", style=color)
    return bar


def _k(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:g}M"
    if tokens >= 100_000:
        return f"{tokens / 1000:.0f}k"
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


class CompactionBlock(FoldBlock):
    """Context compaction: live while running, then one folded line whose
    body shows what the model now sees of the past (e.g. the summary)."""

    DEFAULT_CSS = """
    CompactionBlock > .body { color: #a1a1aa; max-height: 14; }
    """

    def __init__(self, strategy: str) -> None:
        super().__init__()
        self.strategy = strategy
        self.done = False
        self.failed: str | None = None
        self.event: t.Any = None

    @property
    def foldable(self) -> bool:  # type: ignore[override]
        return self.done and bool(self._details())

    def finish(self, event: t.Any) -> None:
        self.done, self.event = True, event
        self.collapse()

    def fail(self, message: str) -> None:
        self.done, self.failed = True, message
        self.collapse()

    def _header(self) -> Text:
        header = Text("◇ ", style=f"bold {ACCENT}")
        if not self.done:
            header.append("Compacting context…", style="italic #a1a1aa")
            return header
        if self.failed is not None:
            header.append("Compaction failed", style="bold #f87171")
            header.append(f" · {self.failed[:80]} · continuing uncompacted", style="#a1a1aa")
            return header
        event = self.event
        left = len(event.old_messages)
        if event.pruned_only:
            header.append("Trimmed old tool output", style="bold #e4e4e7")
        elif event.summary:
            header.append(f"Compacted · {_plural(left, 'message')} → summary", style="bold #e4e4e7")
        else:
            header.append(f"Window slid · {_plural(left, 'message')} out", style="bold #e4e4e7")
        header.append(f" · {_k(event.tokens_before)} → {_k(event.tokens_after)} tokens", style="#a1a1aa")
        return self._fold_hint(header)

    def _details(self) -> str:
        if self.event is None:
            return ""
        return (self.event.summary or "").strip()

    def _body(self) -> Text:
        return Text(self._details())


class PastSummaryBlock(FoldBlock):
    """On resume: what the model remembers of messages compacted away."""

    DEFAULT_CSS = """
    PastSummaryBlock > .body { color: #a1a1aa; max-height: 14; }
    """

    def __init__(self, archived: int, summary: str | None) -> None:
        super().__init__()
        self.archived, self.summary = archived, (summary or "").strip()

    @property
    def foldable(self) -> bool:  # type: ignore[override]
        return bool(self.summary)

    def _header(self) -> Text:
        header = Text("◇ ", style=f"bold {ACCENT}")
        header.append(f"Earlier conversation summarized · {_plural(self.archived, 'message')}",
                      style="bold #a1a1aa")
        return self._fold_hint(header)

    def _body(self) -> Text:
        return Text(self.summary)


def time_ago(moment: t.Any) -> str:
    """``just now`` / ``5 min ago`` / ``3 h ago`` / ``yesterday`` / ``22 Sep``."""
    from datetime import datetime, timezone

    seconds = (datetime.now(timezone.utc) - moment).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86_400:
        return f"{int(seconds // 3600)} h ago"
    if seconds < 172_800:
        return "yesterday"
    return moment.strftime("%d %b")


_FILLER = frozenset(
    "what which who whom whose when where why how would could should do does did "
    "is are was were can will you your yours me my i we us our the a an to of for "
    "in on at about be like want prefer today please it this that".split()
)


class Question(t.NamedTuple):
    """One pending question, ready to render and answer."""

    record_id: str
    text: str
    # (label, description) shown to the user.
    options: list[tuple[str, str]]
    # What gets sent back for each option (records carry "label — description").
    values: list[str]
    header: str | None = None
    # True when the record holds several questions (answers go back as a dict).
    grouped: bool = False


class OptionRow(Static):
    """An option line of the form (the last one is "Other"); click selects it."""

    def __init__(self, question: int, index: int) -> None:
        super().__init__(classes="option")
        self.question = question
        self.index = index

    def on_click(self, event) -> None:
        event.stop()
        form = self.parent
        if isinstance(form, QuestionForm):
            form.pick(self.question, self.index)


class QuestionForm(Vertical):
    """All pending questions in one box: tabs across the top (←→), options
    for the current one (↑↓ + enter or click) plus an "Other" row for a
    free answer typed in the prompt, and — with several questions — a
    Submit tab to review and send them together. Once sent it folds to
    ``? Answered N questions`` (click expands).
    """

    DEFAULT_CSS = """
    QuestionForm { height: auto; border: round #b794f6; padding: 0 1; margin: 1 0 0 0; }
    QuestionForm > .tabs { margin-bottom: 1; }
    QuestionForm > .question { color: #f4f4f5; }
    QuestionForm > .option { height: 1; display: none; }
    QuestionForm > .option.-current { display: block; }
    QuestionForm > .hint { color: #71717a; margin-top: 1; }
    QuestionForm.-done { border: round #3f3f46; width: auto; }
    QuestionForm.-done:hover { border: round #b794f6; }
    QuestionForm.-done > .option.-current, QuestionForm.-done > .question,
    QuestionForm.-done > .hint { display: none; }
    QuestionForm > .summary { display: none; }
    QuestionForm.-done > .summary { display: block; width: auto; }
    """

    class Completed(Message):
        def __init__(self, form: "QuestionForm") -> None:
            super().__init__()
            self.form = form

    class TypingRequested(Message):
        """The user picked "Other": the prompt should ask for their answer."""

        def __init__(self, form: "QuestionForm", question: str) -> None:
            super().__init__()
            self.form = form
            self.question = question

    def __init__(self, questions: list[Question]) -> None:
        super().__init__()
        self.questions = questions
        self.current = 0
        # A question without options starts on its "Other" row.
        self.highlight = [0 if q.values else len(q.values) for q in questions]
        self.answers: dict[int, str] = {}
        self.typing = False
        self.done = False
        self.cancelled = False
        self.expanded = False

    # -------- layout
    def compose(self):
        yield Static(classes="tabs")
        yield Static(classes="question")
        for q, question in enumerate(self.questions):
            for index in range(len(question.options) + 1):
                yield OptionRow(q, index)
        yield Static(classes="hint")
        yield Static(classes="summary")

    def on_mount(self) -> None:
        self.redraw()

    @property
    def multi(self) -> bool:
        return len(self.questions) > 1

    @property
    def on_submit_tab(self) -> bool:
        return self.multi and self.current == len(self.questions)

    def _is_other(self, q: int, index: int) -> bool:
        return index == len(self.questions[q].values)

    def _tab_title(self, question: Question) -> str:
        if question.header:
            return question.header[:14]
        # Drop question filler so "What would you like me to call you?"
        # becomes "call" instead of "What would you".
        words = [w for w in re.findall(r"[\w'-]+", question.text) if w.lower() not in _FILLER]
        title = " ".join(words[:2]) or question.text
        return title if len(title) <= 22 else title[:21] + "…"

    def _display(self, q: int) -> str | None:
        answer = self.answers.get(q)
        if answer is None:
            return None
        for (label, _), value in zip(self.questions[q].options, self.questions[q].values):
            if answer == value:
                return label
        return answer

    def _is_custom(self, q: int) -> bool:
        return q in self.answers and self.answers[q] not in self.questions[q].values

    # -------- drawing
    def redraw(self) -> None:
        parts = {name: self.query(f".{name}") for name in ("tabs", "question", "hint", "summary")}
        if not all(parts.values()):
            return
        tabs, question, hint, summary = (parts[n].first(Static) for n in parts)
        tabs.display = self.multi and not self.done
        tabs.update(self._tabs())
        summary.update(self._summary())
        for row in self.query(OptionRow):
            current = not self.on_submit_tab and row.question == self.current
            row.set_class(current, "-current")
            if current:
                row.update(self._option_line(row.question, row.index))
        if self.on_submit_tab:
            question.update(self._review())
            hint.update(Text("enter submit all · ← back to change an answer", style="#71717a"))
            return
        q = self.questions[self.current]
        title = Text()
        if self.multi:
            title.append(f"{self.current + 1}/{len(self.questions)}  ", style="#71717a")
        title.append(q.text, style="bold")
        question.update(title)
        if self.typing:
            keys = "type your answer in the prompt · enter to confirm"
        else:
            keys = "↑↓ choose · enter select"
            if self.multi:
                keys += " · ←→ switch question"
            keys += " · esc cancel"
        hint.update(Text(keys, style="#71717a"))

    def _tabs(self) -> Text:
        line = Text()
        for q, question in enumerate(self.questions):
            box = "☒" if q in self.answers else "☐"
            style = f"bold reverse {ACCENT}" if q == self.current else ("#4ade80" if q in self.answers else "#a1a1aa")
            line.append(f" {box} {self._tab_title(question)} ", style=style)
            line.append(" ")
        ready = len(self.answers) == len(self.questions)
        style = f"bold reverse {ACCENT}" if self.on_submit_tab else ("#4ade80" if ready else "#71717a")
        line.append(" ✓ Submit ", style=style)
        return line

    def _option_line(self, q: int, index: int) -> Text:
        highlighted = index == self.highlight[q]
        line = Text("❯ " if highlighted else "  ", style=f"bold {ACCENT}")
        if self._is_other(q, index):
            if self._is_custom(q):
                line.append(f"{index + 1}. ✎ {self.answers[q]}", style="bold #4ade80")
                line.append(" ✓", style="#4ade80")
            elif self.typing and highlighted:
                line.append(f"{index + 1}. ✎ Other", style=f"bold {ACCENT}")
                line.append(" — typing in the prompt below…", style=ACCENT)
            else:
                line.append(f"{index + 1}. ✎ Other", style="bold #f4f4f5" if highlighted else "#d4d4d8")
                line.append(" — type your own answer", style="#71717a")
            return line
        label, description = self.questions[q].options[index]
        chosen = self.answers.get(q) == self.questions[q].values[index]
        style = "bold #4ade80" if chosen else ("bold #f4f4f5" if highlighted else "#d4d4d8")
        line.append(f"{index + 1}. {label}", style=style)
        if chosen:
            line.append(" ✓", style="#4ade80")
        if description:
            line.append(f" — {description}", style="#71717a")
        return line

    def _review(self) -> Text:
        body = Text("Review your answers\n", style="bold #f4f4f5")
        for q, question in enumerate(self.questions):
            shown = self._display(q)
            body.append(f"\n  {question.text}\n", style="#d4d4d8")
            body.append("   → ", style="#71717a")
            body.append(shown or "not answered yet", style="bold #4ade80" if shown else "#fbbf24")
        return body

    def _summary(self) -> Text:
        count = len(self.questions)
        header = Text("? ", style=f"bold {ACCENT}")
        if self.cancelled:
            header.append(f"{_plural(count, 'question')} · cancelled", style="#a1a1aa")
            return header
        if count == 1:
            header.append(self.questions[0].text[:70], style="bold #e4e4e7")
            header.append("  → ", style="#71717a")
            header.append((self._display(0) or "")[:50], style="bold #4ade80")
        else:
            header.append(f"Answered {count} questions", style="bold #e4e4e7")
        header.append("   " + ("▾ collapse" if self.expanded else "▸ expand"), style="dim")
        if self.expanded:
            for q, question in enumerate(self.questions):
                header.append(f"\n  {question.text}", style="#a1a1aa")
                header.append("  → ", style="#71717a")
                header.append(self._display(q) or "", style="bold #4ade80")
        return header

    # -------- interaction (the app forwards keys from the prompt)
    def move(self, delta: int) -> None:
        if self.done or self.on_submit_tab:
            return
        self.typing = False
        count = len(self.questions[self.current].values) + 1
        self.highlight[self.current] = (self.highlight[self.current] + delta) % count
        self.redraw()

    def switch(self, delta: int) -> None:
        if self.done or not self.multi:
            return
        self.typing = False
        self.current = (self.current + delta) % (len(self.questions) + 1)
        self.redraw()

    def enter(self, text: str | None) -> None:
        """Enter in the prompt: ``text`` is a free answer or an option
        number; empty picks the highlighted row (or submits)."""
        if self.done:
            return
        if self.on_submit_tab:
            missing = [q for q in range(len(self.questions)) if q not in self.answers]
            if missing:
                self.current = missing[0]
                self.redraw()
            else:
                self._complete()
            return
        q = self.questions[self.current]
        if text:
            if not self.typing and text.isdigit() and 1 <= int(text) <= len(q.values):
                self._answer(self.current, q.values[int(text) - 1])
            else:
                self._answer(self.current, text)
            return
        self.pick(self.current, self.highlight[self.current])

    def pick(self, q: int, index: int) -> None:
        if self.done:
            return
        self.current = q
        self.highlight[q] = index
        if self._is_other(q, index):
            self.typing = True
            self.redraw()
            self.post_message(self.TypingRequested(self, self.questions[q].text))
            return
        self._answer(q, self.questions[q].values[index])

    def _answer(self, q: int, value: str) -> None:
        self.typing = False
        self.answers[q] = value
        if not self.multi:
            self._complete()
            return
        missing = [i for i in range(len(self.questions)) if i not in self.answers]
        self.current = missing[0] if missing else len(self.questions)
        self.redraw()

    def cancel(self) -> None:
        self.done = self.cancelled = True
        self.add_class("-done")
        self.redraw()

    def _complete(self) -> None:
        self.done = True
        self.add_class("-done")
        self.redraw()
        self.post_message(self.Completed(self))

    def result(self) -> dict[str, str | dict[str, str]]:
        """Answers per record: a string, or {question: answer} when the
        record grouped several questions in one ask_user call."""
        out: dict[str, str | dict[str, str]] = {}
        for q, question in enumerate(self.questions):
            answer = self.answers.get(q, "")
            if question.grouped:
                group = out.setdefault(question.record_id, {})
                assert isinstance(group, dict)
                group[question.text] = answer
            else:
                out[question.record_id] = answer
        return out

    def on_click(self) -> None:
        if self.done and not self.cancelled:
            self.expanded = not self.expanded
            self.redraw()


def tool_summary(tool_name: str, parameters: dict[str, t.Any]) -> str:
    """The one argument that says what the call does, not the whole payload."""
    for key in ("command", "file_name", "path", "file_path", "query", "url", "city", "to"):
        if key in parameters:
            return _first_line(str(parameters[key]))
    parts = [f"{k}={v}" for k, v in parameters.items() if k != "description"]
    return _first_line(", ".join(parts))


def _first_line(text: str, limit: int = 100) -> str:
    """A whole script becomes ``cat > x.py << 'EOF' … (+214 lines)``."""
    lines = text.strip().splitlines() or [""]
    first = lines[0] if len(lines[0]) <= limit else lines[0][: limit - 1] + "…"
    return f"{first} … (+{len(lines) - 1} lines)" if len(lines) > 1 else first


def result_text(result: ToolResult) -> str:
    if not result.success:
        return result.error or "failed"
    value = result.result
    if isinstance(value, dict) and ("stdout" in value or "stderr" in value):
        out = str(value.get("stdout") or "").rstrip()
        err = str(value.get("stderr") or "").rstrip()
        return "\n".join(part for part in (out, err) if part) or "(no output)"
    if isinstance(value, str):
        return value or "(empty)"
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


_INTERRUPTED = "interrupted"


class ToolBlock(Vertical):
    """``◆ name(arg)`` then ``╰─ summary``. Output folds; click to expand.

    State lives on the block and ``_redraw`` redraws from it, so updates
    that land before the children are mounted (the spinner tick, a fast
    result) are kept and drawn on mount instead of failing.
    """

    DEFAULT_CSS = """
    ToolBlock { height: auto; margin: 1 0 0 0; }
    ToolBlock:hover > .summary { color: #d4d4d8; }
    ToolBlock > .summary { color: #a1a1aa; padding-left: 2; }
    ToolBlock > .output {
        display: none; color: #d4d4d8; padding: 0 1; margin-left: 5;
        border-left: solid #3f3f46; max-height: 20; overflow-y: auto;
    }
    ToolBlock.-expanded > .output { display: block; }
    """

    MAX_OUTPUT = 6000

    def __init__(self, tool_name: str, parameters: dict[str, t.Any]) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.parameters = parameters
        self.result: ToolResult | None = None
        self._waiting = False
        self._summary = Text()
        self._output = ""
        self._frame = 0

    def compose(self):
        yield Static(classes="title")
        yield Static(classes="summary")
        yield Static(classes="output")

    def on_mount(self) -> None:
        self._redraw()

    @property
    def running(self) -> bool:
        return self.result is None

    def _part(self, name: str) -> Static | None:
        found = self.query(f".{name}")
        return found.first(Static) if found else None

    def _title(self) -> Text:
        if self.result is None:
            dot = Text(SPINNER[self._frame % len(SPINNER)] + " ", style=f"bold {ACCENT}")
        elif self._interrupted:
            dot = Text("◇ ", style="bold #71717a")
        elif self.result.success and not self._nonzero_exit():
            dot = Text("◆ ", style="bold #4ade80")
        else:
            dot = Text("◆ ", style="bold #f87171")
        dot.append(self.tool_name, style="bold")
        summary = tool_summary(self.tool_name, self.parameters)
        if summary:
            if len(summary) > 90:
                summary = summary[:87] + "…"
            dot.append(f"({summary})", style="#a1a1aa")
        return dot

    def _summary_line(self) -> Text:
        if self.result is None:
            if self._waiting:
                return Text("╰─ Waiting for approval…", style="#fbbf24")
            return Text("╰─ Running…", style="dim")
        summary = self._summary.copy()
        hint = "▾ click to collapse" if self.has_class("-expanded") else "▸ click to expand"
        summary.append(f"   {hint}", style="dim")
        return summary

    def _redraw(self) -> None:
        title, summary, output = self._part("title"), self._part("summary"), self._part("output")
        if title is None or summary is None or output is None:
            return
        title.update(self._title())
        summary.update(self._summary_line())
        output.update(Text(self._output))

    def _nonzero_exit(self) -> bool:
        value = self.result.result if self.result else None
        return isinstance(value, dict) and value.get("exit_code") not in (None, 0)

    def set_waiting(self, waiting: bool) -> None:
        self._waiting = waiting
        self._redraw()

    def tick(self) -> None:
        if self.running:
            self._frame += 1
            title = self._part("title")
            if title is not None:
                title.update(self._title())

    @property
    def _interrupted(self) -> bool:
        return self.result is not None and self.result.error == _INTERRUPTED

    def complete(self, result: ToolResult | None) -> None:
        """``None`` means the turn was cancelled before the tool finished."""
        self.result = result or ToolResult(success=False, error=_INTERRUPTED, tool_call_id="")
        text = result_text(self.result)
        lines = text.count("\n") + 1
        ms = self.result.duration_ms
        summary = Text("╰─ ", style="dim")
        if self._interrupted:
            summary.append("cancelled", style="#a1a1aa")
        elif not self.result.success:
            summary.append(f"Error: {text.splitlines()[0][:120] if text else 'failed'}", style="#f87171")
        else:
            code = self.result.result.get("exit_code") if isinstance(self.result.result, dict) else None
            if code not in (None, 0):
                summary.append(f"exit {code} · ", style="#f87171")
            summary.append(_plural(lines, "line"))
            if ms is not None:
                summary.append(f" · {ms}ms", style="dim")
        self._summary = summary
        if len(text) > self.MAX_OUTPUT:
            text = text[: self.MAX_OUTPUT] + f"\n… ({len(text) - self.MAX_OUTPUT:,} more chars)"
        self._output = text
        self._redraw()

    def set_expanded(self, expanded: bool = True) -> None:
        self.set_class(expanded, "-expanded")
        self._redraw()

    def on_click(self) -> None:
        if not self.running:
            self.set_expanded(not self.has_class("-expanded"))
