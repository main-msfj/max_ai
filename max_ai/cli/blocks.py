"""Transcript blocks: each turn piece is a widget, so it can collapse and update."""

from __future__ import annotations

import re
import time
import typing as t

from rich.text import Text
from textual import events
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.widgets import Input, Markdown, Static
from textual.widgets._markdown import MarkdownFence, MarkdownStream

from ..core.tool.display import (  # noqa: F401 — the CLI's app imports them from here
    BASH,
    _first_line,  # pyright: ignore[reportPrivateUsage]
    changes,
    command_lines,
    exit_code,
    first_line,
    result_text,
    tool_label,
    tool_summary,
)
from ..types.todos import Todo
from ..types.tool_call import ToolResult
from .theme import (
    BORDER,
    BORDER_SUBTLE,
    CODE,
    COMMAND,
    DIM,
    ELEMENT,
    ERROR,
    LOGO,
    MUTED,
    PATH,
    PRIMARY,
    SUCCESS,
    SYSTEM,
    TEXT,
    WARNING,
    gradient,
)

# Neutral text and surfaces; PRIMARY only for accents (glyphs, spinner,
# focus/hover, command names). Colors live in theme.py.
SPINNER = "◇◈◆◈"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


class WelcomeBox(Static):
    """Session banner: the MAX-AI logo and the agent. The model is in the line at the bottom."""

    DEFAULT_CSS = """
    WelcomeBox { width: 1fr; height: auto; margin: 2 0 1 0; content-align: center top; text-align: center; }
    """

    def __init__(self, agent_name: str) -> None:
        body = Text(justify="center")
        for left, right in LOGO:
            body.append_text(gradient(left + right))
            body.append("\n")
        body.append(f"\n{agent_name}\n\n", style=f"bold {PRIMARY}")
        for key, what in (("/help", "commands"), ("esc", "interrupt"), ("ctrl+t", "thinking"),
                          ("ctrl+b", "workspace")):
            body.append(key, style=f"bold {PRIMARY}")
            body.append(f" {what}   ", style=DIM)
        super().__init__(body)


class UserBlock(Static):
    DEFAULT_CSS = f"""
    UserBlock {{
        background: {ELEMENT}; color: {TEXT}; padding: 0 1; margin: 1 0 0 0; height: auto;
        border-left: outer {PRIMARY};
    }}
    """

    def __init__(self, text: str) -> None:
        body = Text()
        body.append("❯ ", style=f"bold {PRIMARY}")
        body.append(text, style=f"bold {TEXT}")
        super().__init__(body)


class NoteLine(Static):
    """One dim harness line (gate retries, done marker, errors, verbose events)."""

    DEFAULT_CSS = f"""
    NoteLine {{ height: auto; padding: 0 0 0 2; color: {DIM}; }}
    """

    def __init__(self, text: str | Text, style: str = "") -> None:
        super().__init__(Text(text, style=style) if isinstance(text, str) else text)


class NumberedFence(MarkdownFence):
    """A code block with its line numbers in a dim gutter (one-liners stay bare)."""

    @classmethod
    def highlight(
        cls, code: str, language: str, ansi: bool = False, dark: bool = False
    ) -> Content:
        highlighted = super().highlight(code, language, ansi, dark)
        lines = highlighted.split("\n")
        # The fence's closing newline is not a line of code.
        if len(lines) > 1 and not lines[-1].plain:
            lines = lines[:-1]
        if len(lines) < 2:
            return highlighted
        width = len(str(len(lines)))
        numbered = [
            Content.assemble((f"{number:>{width}}  ", DIM), line)
            for number, line in enumerate(lines, 1)
        ]
        return Content("\n").join(numbered)


class CodeMarkdown(Markdown):
    """Markdown whose code blocks show line numbers."""

    BLOCKS = {**Markdown.BLOCKS, "fence": NumberedFence, "code_block": NumberedFence}


class AssistantBlock(Horizontal):
    """``✦`` + streamed Markdown. Streaming goes through MarkdownStream so a long
    answer isn't re-parsed from scratch on every chunk."""

    DEFAULT_CSS = f"""
    AssistantBlock {{ height: auto; margin: 1 0 0 0; }}
    AssistantBlock > .bullet {{ width: 2; color: {PRIMARY}; }}
    AssistantBlock > Markdown {{ width: 1fr; margin: 0; padding: 0; background: transparent; }}
    AssistantBlock Markdown MarkdownBlock:dark > .code_inline {{ background: transparent; color: {CODE}; }}
    AssistantBlock Markdown MarkdownBlock > .strong {{ color: {TEXT}; text-style: bold; }}
    AssistantBlock MarkdownBullet {{ color: {PRIMARY}; }}
    AssistantBlock Markdown > MarkdownBlock:first-child {{ margin-top: 0; }}
    AssistantBlock MarkdownH1, AssistantBlock MarkdownH2 {{ color: {PRIMARY}; background: transparent; }}
    AssistantBlock MarkdownH3, AssistantBlock MarkdownH4 {{ color: {SYSTEM}; background: transparent; }}
    """

    def __init__(self) -> None:
        super().__init__()
        self.text = ""
        self._stream: MarkdownStream | None = None

    def compose(self):
        yield Static("✦", classes="bullet")
        yield CodeMarkdown("")

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

    DEFAULT_CSS = f"""
    FoldBlock {{
        height: auto; border: round {BORDER}; padding: 0 1; margin: 1 0 0 0;
    }}
    FoldBlock:hover {{ border: round {PRIMARY}; }}
    FoldBlock > .header {{ color: {MUTED}; }}
    FoldBlock > .body {{ color: {TEXT}; max-height: 16; overflow-y: auto; }}
    FoldBlock.-collapsed {{ width: auto; }}
    FoldBlock.-collapsed > .header {{ width: auto; }}
    FoldBlock.-collapsed > .body {{ display: none; }}
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

    DEFAULT_CSS = f"""
    ThinkingBlock {{ border: none; border-left: outer {BORDER_SUBTLE}; padding: 0 1; }}
    ThinkingBlock:hover {{ border: none; border-left: outer {PRIMARY}; }}
    ThinkingBlock > .body {{ color: {MUTED}; text-style: italic; max-height: 12; }}
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
        header = Text()
        header.append("◈ ", style=PRIMARY)
        if not self.done:
            header.append("Thinking…", style="italic")
            return header
        words = len(self.text.split())
        took = f"{self._elapsed:.0f}s" if self._elapsed >= 1 else "<1s"
        header.append(f"Thought for {took}", style=f"italic {MUTED}")
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


_TODO_STYLES = {
    "done": ("[x]", f"{DIM} strike"),
    "in_progress": ("[>]", f"bold {PRIMARY}"),
    "pending": ("[ ]", TEXT),
}


class TodoBlock(FoldBlock):
    """The agent's list of things to do (``write_todos``) under ``⎿``. An
    older list shrinks to its progress line when a newer one comes."""

    DEFAULT_CSS = """
    TodoBlock, TodoBlock:hover { border: none; padding: 0; }
    TodoBlock > .body { max-height: 100; }
    """
    foldable = False

    def __init__(self, todos: list[Todo]) -> None:
        super().__init__()
        self.todos = list(todos)

    def update_todos(self, todos: list[Todo]) -> None:
        self.todos = list(todos)
        self.refresh_block()

    def _header(self) -> Text:
        done = sum(todo.status == "done" for todo in self.todos)
        header = Text("● ", style=f"bold {PRIMARY}")
        header.append(f"Todos {done}/{len(self.todos)}", style=f"bold {TEXT}")
        # The item in progress, else the first one still pending.
        current = next((x for x in self.todos if x.status == "in_progress"), None)
        current = current or next((x for x in self.todos if x.status == "pending"), None)
        if self.todos and done == len(self.todos):
            header.append(" · all done", style=SUCCESS)
        elif current is not None:
            header.append(f" · {current.content[:60]}", style=MUTED)
        return self._fold_hint(header)

    def _body(self) -> Text:
        body = Text()
        for index, todo in enumerate(self.todos):
            icon, style = _TODO_STYLES.get(todo.status, ("·", ""))
            body.append("  ⎿  " if index == 0 else "\n     ", style=DIM)
            body.append(f"{icon} {todo.content}", style=style)
        return body


def context_bar(used: int, maximum: int, cells: int = 10) -> Text:
    """``▓▓▓▓░░░░░░ 52k / 128k``: green, amber from 60% used, red from 85%."""
    ratio = min(1.0, used / maximum) if maximum > 0 else 0.0
    color = ERROR if ratio >= 0.85 else WARNING if ratio >= 0.6 else SUCCESS
    filled = round(ratio * cells)
    bar = Text("━" * filled, style=color)
    bar.append("━" * (cells - filled), style=BORDER_SUBTLE)
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

    DEFAULT_CSS = f"""
    CompactionBlock > .body {{ color: {MUTED}; max-height: 14; }}
    """

    def __init__(self, strategy: str) -> None:
        super().__init__()
        self.strategy = strategy
        self.done = False
        self.event: t.Any = None

    @property
    def foldable(self) -> bool:  # type: ignore[override]
        return self.done and bool(self._details())

    def finish(self, event: t.Any) -> None:
        self.done, self.event = True, event
        self.collapse()

    def _header(self) -> Text:
        header = Text("◇ ", style=f"bold {PRIMARY}")
        if not self.done:
            header.append("Compacting context…", style=f"italic {MUTED}")
            return header
        event = self.event
        kept = _plural(event.kept_message_count, "message")
        if event.pruned_only:
            header.append("Trimmed old tool output", style=f"bold {TEXT}")
        elif event.summary:
            header.append(f"Compacted · {kept} kept, the rest → summary", style=f"bold {TEXT}")
        else:
            header.append(f"Window slid · {kept} kept", style=f"bold {TEXT}")
        header.append(f" · {_k(event.tokens_before)} → {_k(event.tokens_after)} tokens", style=MUTED)
        # The messages left with nothing kept of them: the user must know.
        if event.summary_failed:
            lost = _plural(event.removed, "message")
            header.append(f" · the summary failed: {lost} left without one", style=WARNING)
        return self._fold_hint(header)

    def _details(self) -> str:
        if self.event is None:
            return ""
        return (self.event.summary or "").strip()

    def _body(self) -> Text:
        return Text(self._details())


class PastSummaryBlock(FoldBlock):
    """On resume: what the model remembers of messages compacted away."""

    DEFAULT_CSS = f"""
    PastSummaryBlock > .body {{ color: {MUTED}; max-height: 14; }}
    """

    def __init__(self, summary: str | None) -> None:
        super().__init__()
        self.summary = (summary or "").strip()

    @property
    def foldable(self) -> bool:  # type: ignore[override]
        return bool(self.summary)

    def _header(self) -> Text:
        header = Text("◇ ", style=f"bold {PRIMARY}")
        header.append("Earlier messages were summarized", style=f"bold {MUTED}")
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
    # What gets sent back for each option.
    values: list[str]
    header: str | None = None
    # Several options can be chosen; the answer is then a list.
    multi_select: bool = False


class OptionRow(Static):
    """An option line of the form (the last one is "Other"); click selects it."""

    def __init__(self, question: int, index: int) -> None:
        super().__init__(classes="option")
        self.question = question
        self.index = index

    def on_click(self, event: events.Click) -> None:
        event.stop()
        form = self.parent
        if isinstance(form, QuestionForm):
            form.pick(self.question, self.index)


class QuestionForm(Vertical):
    """All pending questions in one box: tabs across the top (←→), options
    for the current one (↑↓ + enter, a number key or click) plus an "Other"
    row that opens a text field right in the box, and — with several
    questions — a Submit tab to review and send them together. Once sent
    the box gives way to one ``question → answer`` line for each.
    """

    DEFAULT_CSS = f"""
    QuestionForm {{ height: auto; border: round {PRIMARY}; padding: 0 1; margin: 1 0 0 0; }}
    QuestionForm > .tabs {{ margin-bottom: 1; }}
    QuestionForm > .question {{ color: {TEXT}; }}
    QuestionForm > .option {{ height: 1; display: none; }}
    QuestionForm > .option.-current {{ display: block; }}
    QuestionForm > .hint {{ color: {DIM}; margin-top: 1; }}
    QuestionForm > Input, QuestionForm > Input:focus {{
        display: none; height: 1; border: none; padding: 0 1; margin-left: 5;
        background: {ELEMENT}; color: {TEXT};
    }}
    QuestionForm.-typing > Input {{ display: block; }}
    QuestionForm.-done {{ border: none; padding: 0; }}
    QuestionForm.-done > .option.-current, QuestionForm.-done > .question,
    QuestionForm.-done > .hint, QuestionForm.-done > Input {{ display: none; }}
    QuestionForm > .summary {{ display: none; }}
    QuestionForm.-done > .summary {{ display: block; }}
    """

    class Completed(Message):
        def __init__(self, form: "QuestionForm") -> None:
            super().__init__()
            self.form = form

    class TypingFinished(Message):
        """The "Other" field closed: the prompt takes the keys again."""

        def __init__(self, form: "QuestionForm") -> None:
            super().__init__()
            self.form = form

    def __init__(self, questions: list[Question]) -> None:
        super().__init__()
        self.questions = questions
        self.current = 0
        # A question without options starts on its "Other" row.
        self.highlight = [0 if q.values else len(q.values) for q in questions]
        self.answers: dict[int, str | list[str]] = {}
        self.typing = False
        self.done = False
        self.cancelled = False

    # -------- layout
    def compose(self):
        yield Static(classes="tabs")
        yield Static(classes="question")
        for q, question in enumerate(self.questions):
            for index in range(len(question.options) + 1):
                yield OptionRow(q, index)
        yield Input(placeholder="Type your answer · enter to send · esc to go back")
        yield Static(classes="hint")
        yield Static(classes="summary")

    def on_mount(self) -> None:
        self.redraw()
        if not self.questions[0].values:
            self.pick(0, 0)  # nothing to choose: straight to the text field

    @property
    def several_questions(self) -> bool:
        return len(self.questions) > 1

    @property
    def has_submit(self) -> bool:
        """Several questions, or one with several answers, need confirming."""
        return self.several_questions or self.questions[0].multi_select

    @property
    def on_submit_tab(self) -> bool:
        return self.has_submit and self.current == len(self.questions)

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

    def _chosen(self, q: int) -> list[str]:
        answer = self.answers.get(q)
        if answer is None:
            return []
        return answer if isinstance(answer, list) else [answer]

    def _display(self, q: int) -> str | None:
        labels = dict(zip(self.questions[q].values, (o[0] for o in self.questions[q].options)))
        return ", ".join(labels.get(value, value) for value in self._chosen(q)) or None

    def _custom(self, q: int) -> str | None:
        """What the user typed for this question, if anything."""
        return next((v for v in self._chosen(q) if v not in self.questions[q].values), None)

    # -------- drawing
    def redraw(self) -> None:
        parts = {name: self.query(f".{name}") for name in ("tabs", "question", "hint", "summary")}
        if not all(parts.values()):
            return
        tabs, question, hint, summary = (parts[n].first(Static) for n in parts)
        tabs.display = self.has_submit and not self.done
        tabs.update(self._tabs())
        summary.update(self._summary())
        for row in self.query(OptionRow):
            current = not self.on_submit_tab and row.question == self.current
            row.set_class(current, "-current")
            if current:
                row.update(self._option_line(row.question, row.index))
        if self.on_submit_tab:
            question.update(self._review())
            hint.update(Text("enter submit all · ← back to change an answer", style=DIM))
            return
        q = self.questions[self.current]
        title = Text()
        if self.several_questions:
            title.append(f"{self.current + 1}/{len(self.questions)}  ", style=DIM)
        title.append(q.text, style="bold")
        if q.multi_select:
            title.append(" (select all that apply)", style=DIM)
        question.update(title)
        if self.typing:
            keys = "enter send · esc back to the options"
        else:
            keys = "↑↓ choose · enter toggles" if q.multi_select else "↑↓ choose · enter or 1-9 select"
            if self.has_submit:
                keys += " · ←→ switch question"
            keys += " · esc cancel"
        hint.update(Text(keys, style=DIM))

    def _tabs(self) -> Text:
        line = Text()
        for q, question in enumerate(self.questions):
            style = f"bold reverse {PRIMARY}" if q == self.current else (SUCCESS if q in self.answers else MUTED)
            line.append(f" {self._tab_title(question)} ", style=style)
            line.append(" ")
        ready = len(self.answers) == len(self.questions)
        style = f"bold reverse {PRIMARY}" if self.on_submit_tab else (SUCCESS if ready else TEXT)
        line.append(" Submit ", style=style)
        return line

    def _option_line(self, q: int, index: int) -> Text:
        highlighted = index == self.highlight[q]
        line = Text("❯ " if highlighted else "  ", style=f"bold {PRIMARY}")
        several = self.questions[q].multi_select
        if self._is_other(q, index):
            custom = self._custom(q)
            mark = ("[x] " if custom else "[ ] ") if several else "✎ "
            if custom:
                line.append(f"{index + 1}. {mark}{custom}", style=f"bold {SUCCESS}")
            elif self.typing and highlighted:
                line.append(f"{index + 1}. {mark}Other", style=f"bold {PRIMARY}")
            else:
                line.append(f"{index + 1}. {mark}Other", style=f"bold {TEXT}" if highlighted else TEXT)
                line.append(" — type your own answer", style=DIM)
            return line
        label, description = self.questions[q].options[index]
        chosen = self.questions[q].values[index] in self._chosen(q)
        style = f"bold {SUCCESS}" if chosen else (f"bold {TEXT}" if highlighted else TEXT)
        mark = ("[x] " if chosen else "[ ] ") if several else ""
        line.append(f"{index + 1}. {mark}{label}", style=style)
        if description:
            line.append(f" — {description}", style=DIM)
        return line

    def _review(self) -> Text:
        body = Text("Review your answers\n", style=f"bold {TEXT}")
        for q, question in enumerate(self.questions):
            shown = self._display(q)
            body.append(f"\n  {question.text} → ", style=TEXT)
            body.append(shown or "not answered yet", style=f"bold {SUCCESS}" if shown else WARNING)
        return body

    def _summary(self) -> Text:
        """One ``question → answer`` line for every question."""
        body = Text()
        if self.cancelled:
            body.append("? ", style=f"bold {PRIMARY}")
            body.append(f"{_plural(len(self.questions), 'question')} · cancelled", style=MUTED)
            return body
        for q, question in enumerate(self.questions):
            if q:
                body.append("\n")
            body.append(f"{question.text} → ", style=f"bold {TEXT}")
            body.append(self._display(q) or "(no answer)", style=SUCCESS)
        return body

    # -------- interaction (the app forwards keys from the prompt)
    def move(self, delta: int) -> None:
        if self.done or self.on_submit_tab:
            return
        self._close_field()
        count = len(self.questions[self.current].values) + 1
        self.highlight[self.current] = (self.highlight[self.current] + delta) % count
        self.redraw()

    def switch(self, delta: int) -> None:
        if self.done or not self.has_submit:
            return
        self._close_field()
        self.current = (self.current + delta) % (len(self.questions) + 1)
        self.redraw()

    def enter(self, text: str | None) -> None:
        """Enter in the prompt: ``text`` is a free answer or an option
        number; empty picks the highlighted row (or submits)."""
        if self.done:
            return
        if self.on_submit_tab:
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
            custom = self._custom(q)
            if custom and self.questions[q].multi_select:
                self._toggle(q, custom)  # a second press takes it back
                return
            self.typing = True
            self.set_class(True, "-typing")
            self.redraw()
            field = self.query_one(Input)
            field.value = custom or ""
            field.focus()
            return
        self._answer(q, self.questions[q].values[index])

    def pick_number(self, number: int) -> bool:
        """A number key: pick that option of the current question."""
        if self.done or self.typing or self.on_submit_tab:
            return False
        question = self.questions[self.current]
        if not question.values or not 1 <= number <= len(question.values) + 1:
            return False
        self.pick(self.current, number - 1)
        return True

    def stop_typing(self) -> None:
        """Esc in the "Other" field: back to the options."""
        self._close_field()
        self.redraw()

    def _close_field(self) -> None:
        """Hide the "Other" field; the prompt takes the keys again."""
        if self.typing:
            self.typing = False
            self.set_class(False, "-typing")
            self.post_message(self.TypingFinished(self))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if event.value.strip():
            self._answer(self.current, event.value.strip())

    def _toggle(self, q: int, value: str) -> None:
        """Mark or unmark ``value``; one typed answer per question at most."""
        values = self.questions[q].values
        chosen = self._chosen(q)
        if value in chosen:
            chosen.remove(value)
        else:
            if value not in values:
                chosen = [v for v in chosen if v in values]
            chosen.append(value)
        if chosen:
            self.answers[q] = chosen
        else:
            self.answers.pop(q, None)
        self.redraw()

    def _answer(self, q: int, value: str) -> None:
        self._close_field()
        if self.questions[q].multi_select:
            self._toggle(q, value)
            return
        self.answers[q] = value
        if not self.has_submit:
            self._complete()
            return
        missing = [i for i in range(len(self.questions)) if i not in self.answers]
        self.current = missing[0] if missing else len(self.questions)
        self.redraw()
        if not self.on_submit_tab and not self.questions[self.current].values:
            self.pick(self.current, 0)

    def cancel(self) -> None:
        self.done = self.cancelled = True
        self.add_class("-done")
        self.redraw()

    def _complete(self) -> None:
        self.done = True
        self.add_class("-done")
        self.redraw()
        self.post_message(self.Completed(self))

    def result(self) -> dict[str, dict[str, str | list[str]]]:
        """Answers per record: {record_id: {question: answer}}."""
        out: dict[str, dict[str, str | list[str]]] = {}
        for q, question in enumerate(self.questions):
            group = out.setdefault(question.record_id, {})
            if q in self.answers:  # unanswered ones stay out: the tool says so
                group[question.text] = self.answers[q]
        return out


_INTERRUPTED = "interrupted"


class ToolBlock(Vertical):
    """``◆ Name  what it does``, then a short card: ``IN`` with what goes in
    (the command, or the lines a file tool changes) and ``OUT`` with how it
    went. Folded it shows a few lines of each; a click opens it whole.

    State lives on the block and ``_redraw`` draws from it, so updates that
    land before the children are mounted (the spinner tick, a fast result)
    are drawn on mount.
    """

    DEFAULT_CSS = f"""
    ToolBlock {{ height: auto; margin: 1 0 0 0; }}
    ToolBlock > .card {{
        height: auto; margin: 0 0 0 2; padding: 0 1; border: round {BORDER_SUBTLE};
    }}
    ToolBlock:hover > .card {{ border: round {BORDER}; }}
    ToolBlock .input, ToolBlock .output {{
        height: auto; color: {MUTED}; text-wrap: nowrap; text-overflow: ellipsis;
    }}
    /* A line between what goes in and what comes out, when there are both. */
    ToolBlock .output.-below {{ border-top: solid {BORDER_SUBTLE}; }}
    ToolBlock.-expanded .input, ToolBlock.-expanded .output {{
        max-height: 20; overflow-y: auto; text-wrap: wrap;
    }}
    """

    MAX_OUTPUT = 6000
    PREVIEW_LINES = 3  # of a command and of what it printed, before the block is opened
    DIFF_LINES = 4  # of a change, before the block is opened
    CARD_LINES = 30  # lines of a change shown when it asks to run
    QUIET: t.ClassVar[frozenset[str]] = frozenset({"read_file"})  # output counted, not shown
    OPEN: t.ClassVar[str] = " · click to open"

    def __init__(self, tool_name: str, parameters: dict[str, t.Any]) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.parameters = parameters
        self.changed = changes(tool_name, parameters)  # the lines a file tool changes
        self.result: ToolResult | None = None
        self._waiting = False
        self._output = ""
        self._frame = 0

    def compose(self):
        yield Static(classes="title")
        with Vertical(classes="card"):
            yield Static(classes="input")
            yield Static(classes="output")

    def on_mount(self) -> None:
        self._redraw()

    @property
    def running(self) -> bool:
        return self.result is None

    @property
    def is_bash(self) -> bool:
        return self.tool_name == BASH

    def _part(self, name: str) -> Static | None:
        found = self.query(f".{name}")
        return found.first(Static) if found else None

    def _title(self) -> Text:
        dot = Text()
        if self.result is None:
            dot.append(SPINNER[self._frame % len(SPINNER)] + " ", style=f"bold {PRIMARY}")
        elif self._interrupted:
            dot.append("◇ ", style=f"bold {DIM}")
        elif self.result.success and exit_code(self.result) in (None, 0):
            dot.append("◆ ", style=f"bold {SUCCESS}")
        else:
            dot.append("◆ ", style=f"bold {ERROR}")
        # A command left running says so beside its name, where it is seen.
        background = self.is_bash and bool(self.parameters.get("background"))
        summary = tool_summary(self.tool_name, {**self.parameters, "background": False})
        if len(summary) > 90:
            summary = summary[:87] + "…"
        # Bash: its description is the title; the command goes on the IN line.
        if self.is_bash:
            dot.append("Bash", style=f"bold {TEXT}")
            if background:
                dot.append("  ↗ background", style=f"bold {PRIMARY}")
            if summary:
                dot.append(f"  {summary}", style=MUTED)
            return dot.append_text(self._meta())
        dot.append(tool_label(self.tool_name), style=f"bold {TEXT}")
        if summary:
            dot.append("  ")
            dot.append(summary, style=self._argument_color(self.parameters))

        # A file tool says how many lines it puts in and takes out.
        marks = [mark for mark, _ in self.changed]
        if marks:
            dot.append(f"  +{marks.count('+')}", style=SUCCESS)
            if "-" in marks:
                dot.append(f" −{marks.count('-')}", style=ERROR)
        return dot.append_text(self._meta())

    def _meta(self) -> Text:
        """How a call that ended went, at the end of its title: ``exit 1 · 27ms``."""
        meta = Text()
        if self.result is None or self._interrupted or not self.result.success:
            return meta
        code, ms = exit_code(self.result), self.result.duration_ms
        if code not in (None, 0):
            meta.append(f"  exit {code}", style=ERROR)
        if ms is not None:
            meta.append(f"  {ms}ms" if ms < 1000 else f"  {ms / 1000:.1f}s", style=DIM)
        return meta

    @staticmethod
    def _argument_color(parameters: dict[str, t.Any]) -> str:
        """Paths and commands in plain text, searches blue: the argument says what it is."""
        if any(key in parameters for key in ("file_path", "path")):
            return PATH
        if "command" in parameters:
            return COMMAND
        if any(key in parameters for key in ("pattern", "query", "url")):
            return SYSTEM
        return MUTED

    def _diff(self, opened: bool) -> tuple[list[Text], int]:
        """The lines this call changes and how many more are hidden. Folded,
        a few of them, from just before the first one that changes."""
        lines = list(self.diff(self.changed, first_line(self.tool_name, self.result)).split("\n"))
        if opened:
            return lines, 0
        first = next((n for n, (mark, _) in enumerate(self.changed) if mark != " "), 0)
        shown = lines[max(0, first - 1) :][: self.DIFF_LINES]
        return shown, len(lines) - len(shown)

    @staticmethod
    def diff(changed: list[tuple[str, str]], start: int | None, limit: int | None = None) -> Text:
        """The lines a file tool changes: red out, green in, each with its
        line of the file when ``start`` is known. At most ``limit`` of them."""
        lines = changed if limit is None else changed[:limit]
        # A line that leaves is counted in the old file; the others in the new one.
        old = new = start or 0
        shown = Text()
        for index, (mark, line) in enumerate(lines):
            shown.append("\n" if index else "")
            if start is not None:
                shown.append(f"{old if mark == '-' else new:>4} ", style=DIM)
            old, new = old + (mark != "+"), new + (mark != "-")
            style = {"+": SUCCESS, "-": ERROR}.get(mark, MUTED)
            shown.append(f"{'−' if mark == '-' else mark} {line}", style=style)
        if len(lines) < len(changed):
            shown.append(f"\n… (+{len(changed) - len(lines)} lines)", style=DIM)
        return shown

    @classmethod
    def approval(cls, tool_name: str, parameters: dict[str, t.Any], why: str | None) -> Text:
        """What the user sees when a call asks to run: what it is, what
        goes in, and why it is asked."""
        # 1. The same title as the tool's block: its name and what it does.
        card = Text("◆ ", style=f"bold {PRIMARY}")
        card.append(tool_label(tool_name), style=f"bold {TEXT}")
        if summary := tool_summary(tool_name, parameters):
            card.append(f"  {summary}", style=TEXT if tool_name != BASH else MUTED)

        # 2. What goes in: the lines of a file tool, the whole command, or the arguments.
        changed = changes(tool_name, parameters)
        if changed:
            given = cls.diff(changed, first_line(tool_name, None), cls.CARD_LINES)
        elif tool_name == BASH:
            given = Text("\n".join(command_lines(str(parameters.get("command") or ""))), style=TEXT)
        else:
            given = Text("\n".join(f"{key}: {value}" for key, value in parameters.items()), style=TEXT)
        card.append_text(cls.labeled("IN", given))

        # 3. Why it is asked, when the harness said it.
        if why:
            card.append_text(cls.labeled("WHY", Text(why, style=WARNING)))
        return card

    @staticmethod
    def labeled(label: str, body: Text) -> Text:
        """``body`` under a short label, its other lines aligned with the first."""
        shown = Text("\n\n")
        shown.append(f"  {label:<5}", style=f"bold {DIM}")
        for index, line in enumerate(body.split("\n")):
            shown.append("\n       " if index else "")
            shown.append_text(line)
        return shown

    @classmethod
    def rows(cls, label: str, lines: list[Text], hidden: int = 0) -> Text:
        """``lines`` under a short label, the rest aligned with the first;
        ``hidden`` more are only counted, with how to see them."""
        shown = Text()
        shown.append(f"{label:<5}", style=f"bold {DIM}")
        for index, line in enumerate(lines):
            shown.append("\n     " if index else "")
            shown.append_text(line)
        if hidden:
            shown.append(f"\n     … +{_plural(hidden, 'line')}{cls.OPEN}", style=DIM)
        return shown

    def _input(self) -> Text:
        """``IN``: the command, or the lines a file tool changes. A few of
        them folded, all of them once opened."""
        opened = self.has_class("-expanded")
        if self.changed:
            lines, hidden = self._diff(opened)
            return self.rows("IN", lines, hidden)
        if not self.is_bash:
            return Text()
        command = str(self.parameters.get("command") or "").strip()
        lines = [Text(line, style=COMMAND) for line in command.splitlines()] or [Text("")]
        shown = lines if opened else lines[: self.PREVIEW_LINES]
        return self.rows("IN", shown, len(lines) - len(shown))

    def _out(self) -> Text:
        """``OUT``: how it went. A few lines of what it printed folded, all
        of it once opened; a failure in one line."""
        opened = self.has_class("-expanded")
        # 1. Not ended yet, or stopped by the user.
        if self.result is None:
            if self._waiting:
                return Text("OUT  waiting for approval…", style=WARNING)
            return Text("OUT  running…", style=f"italic {DIM}")
        if self._interrupted:
            return self.rows("OUT", [Text("cancelled", style=MUTED)])
        # 2. A failure: what went wrong, in red.
        printed = self._output.splitlines() or [""]
        if not self.result.success:
            lines = [Text(line, style=ERROR) for line in printed]
            lines[0] = Text(f"Error: {printed[0] or 'failed'}", style=ERROR)
            shown = lines if opened else lines[:1]
            return self.rows("OUT", shown, len(lines) - len(shown))
        # 3. A tool whose output is the file itself: counted until opened.
        if self.tool_name in self.QUIET and not opened:
            return Text.assemble(
                ("OUT  ", f"bold {DIM}"), (_plural(len(printed), "line"), MUTED), (self.OPEN, DIM)
            )
        # 4. What it printed.
        lines = [Text(line, style=TEXT if opened else MUTED) for line in printed]
        shown = lines if opened else lines[: self.PREVIEW_LINES]
        return self.rows("OUT", shown, len(lines) - len(shown))

    def _redraw(self) -> None:
        title, given, output = self._part("title"), self._part("input"), self._part("output")
        if title is None or given is None or output is None:
            return
        title.update(self._title())
        given.display = self.is_bash or bool(self.changed)
        given.update(self._input())
        output.set_class(bool(given.display), "-below")
        output.update(self._out())

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
