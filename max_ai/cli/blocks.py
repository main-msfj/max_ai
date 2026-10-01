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
from textual.widgets import Input, Markdown, Static
from textual.widgets._markdown import MarkdownStream

from ..types.tool_call import ToolResult
from .theme import (
    BORDER,
    DIM,
    ELEMENT,
    ERROR,
    LOGO,
    MUTED,
    PRIMARY,
    SUCCESS,
    TEXT,
    WARNING,
)

# Neutral text and surfaces; PRIMARY only for accents (glyphs, spinner,
# focus/hover, command names). Colors live in theme.py.
SPINNER = "◇◈◆◈"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


class WelcomeBox(Static):
    """Session banner: the MAX-AI logo, then agent, model and workspace."""

    DEFAULT_CSS = """
    WelcomeBox { width: 1fr; height: auto; margin: 2 0 1 0; content-align: center top; text-align: center; }
    """

    def __init__(self, agent_name: str, model: str, workspace: Path) -> None:
        body = Text(justify="center")
        for left, right in LOGO:
            body.append(left, style=MUTED)
            body.append(right + "\n", style=f"bold {TEXT}")
        body.append(f"\n{agent_name}", style=f"bold {PRIMARY}")
        body.append(f" · {model}\n", style=MUTED)
        body.append(f"{workspace}\n\n", style=DIM)
        body.append("/help", style=f"bold {TEXT}")
        body.append(" commands · ", style=DIM)
        body.append("esc", style=f"bold {TEXT}")
        body.append(" interrupt · ", style=DIM)
        body.append("ctrl+t", style=f"bold {TEXT}")
        body.append(" thinking", style=DIM)
        super().__init__(body)


class UserBlock(Static):
    DEFAULT_CSS = f"""
    UserBlock {{
        background: {ELEMENT}; color: {TEXT}; padding: 0 1; margin: 1 0 0 0; height: auto;
        border-left: outer {PRIMARY};
    }}
    """

    def __init__(self, text: str) -> None:
        body = Text("❯ ", style=f"bold {PRIMARY}")
        body.append(text)
        super().__init__(body)


class NoteLine(Static):
    """One dim harness line (gate retries, done marker, errors, verbose events)."""

    DEFAULT_CSS = f"""
    NoteLine {{ height: auto; padding: 0 0 0 2; color: {DIM}; }}
    """

    def __init__(self, text: str | Text, style: str = "") -> None:
        super().__init__(Text(text, style=style) if isinstance(text, str) else text)


class AssistantBlock(Horizontal):
    """``✦`` + streamed Markdown. Streaming goes through MarkdownStream so a long
    answer isn't re-parsed from scratch on every chunk."""

    DEFAULT_CSS = f"""
    AssistantBlock {{ height: auto; margin: 1 0 0 0; }}
    AssistantBlock > .bullet {{ width: 2; color: {PRIMARY}; }}
    AssistantBlock > Markdown {{ width: 1fr; margin: 0; padding: 0; background: transparent; }}
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
        header = Text("◈ ", style=PRIMARY)
        if not self.done:
            header.append("Thinking…", style="italic")
            return header
        words = len(self.text.split())
        took = f"{self._elapsed:.0f}s" if self._elapsed >= 1 else "<1s"
        header.append(f"Thought for {took}", style=f"bold {MUTED}")
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
    "done": ("☒", f"{DIM} strike"),
    "active": ("▶", f"bold {PRIMARY}"),
    "failed": ("✗", ERROR),
    "pending": ("☐", TEXT),
}


class PlanBlock(FoldBlock):
    """The agent's plan as a checklist under ``⎿``. An older plan shrinks to
    its progress line when a newer one comes."""

    DEFAULT_CSS = """
    PlanBlock, PlanBlock:hover { border: none; padding: 0; }
    PlanBlock > .body { max-height: 100; }
    """
    foldable = False

    def __init__(self, plan: t.Any) -> None:
        super().__init__()
        self.plan = plan

    def update_plan(self, plan: t.Any) -> None:
        self.plan = plan
        self.refresh_block()

    def _header(self) -> Text:
        steps = list(self.plan.steps)
        done = sum(step.status == "done" for step in steps)
        header = Text("● ", style=f"bold {PRIMARY}")
        header.append(f"Plan {done}/{len(steps)}", style=f"bold {TEXT}")
        current = next((s for s in steps if s.status == "active"), None)
        current = current or next((s for s in steps if s.status == "pending"), None)
        if done == len(steps) and steps:
            header.append(" · all done", style=SUCCESS)
        elif current is not None:
            header.append(f" · {current.description[:60]}", style=MUTED)
        return self._fold_hint(header)

    def _body(self) -> Text:
        body = Text()
        for index, step in enumerate(self.plan.steps):
            icon, style = _PLAN_STYLES.get(step.status, ("·", ""))
            body.append("  ⎿  " if index == 0 else "\n     ", style=DIM)
            body.append(f"{icon} {step.description}", style=style)
        return body


def context_bar(used: int, maximum: int, cells: int = 10) -> Text:
    """``▓▓▓▓░░░░░░ 52k / 128k``: green, amber from 60% used, red from 85%."""
    ratio = min(1.0, used / maximum) if maximum > 0 else 0.0
    color = ERROR if ratio >= 0.85 else WARNING if ratio >= 0.6 else SUCCESS
    filled = round(ratio * cells)
    bar = Text("▓" * filled, style=color)
    bar.append("░" * (cells - filled), style=BORDER)
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
        header = Text("◇ ", style=f"bold {PRIMARY}")
        if not self.done:
            header.append("Compacting context…", style=f"italic {MUTED}")
            return header
        if self.failed is not None:
            header.append("Compaction failed", style=f"bold {ERROR}")
            header.append(f" · {self.failed[:80]} · continuing uncompacted", style=MUTED)
            return header
        event = self.event
        left = len(event.old_messages)
        if event.pruned_only:
            header.append("Trimmed old tool output", style=f"bold {TEXT}")
        elif event.summary:
            header.append(f"Compacted · {_plural(left, 'message')} → summary", style=f"bold {TEXT}")
        else:
            header.append(f"Window slid · {_plural(left, 'message')} out", style=f"bold {TEXT}")
        header.append(f" · {_k(event.tokens_before)} → {_k(event.tokens_after)} tokens", style=MUTED)
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

    def __init__(self, archived: int, summary: str | None) -> None:
        super().__init__()
        self.archived, self.summary = archived, (summary or "").strip()

    @property
    def foldable(self) -> bool:  # type: ignore[override]
        return bool(self.summary)

    def _header(self) -> Text:
        header = Text("◇ ", style=f"bold {PRIMARY}")
        header.append(f"Earlier conversation summarized · {_plural(self.archived, 'message')}",
                      style=f"bold {MUTED}")
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
    for the current one (↑↓ + enter, a number key or click) plus an "Other"
    row that opens a text field right in the box, and — with several
    questions — a Submit tab to review and send them together. Once sent
    the box gives way to each question with its answer under ``⎿``.
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
        self.answers: dict[int, str] = {}
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
            hint.update(Text("enter submit all · ← back to change an answer", style=DIM))
            return
        q = self.questions[self.current]
        title = Text()
        if self.multi:
            title.append(f"{self.current + 1}/{len(self.questions)}  ", style=DIM)
        title.append(q.text, style="bold")
        question.update(title)
        if self.typing:
            keys = "enter send · esc back to the options"
        else:
            keys = "↑↓ choose · enter or 1-9 select"
            if self.multi:
                keys += " · ←→ switch question"
            keys += " · esc cancel"
        hint.update(Text(keys, style=DIM))

    def _tabs(self) -> Text:
        line = Text()
        for q, question in enumerate(self.questions):
            box = "☒" if q in self.answers else "☐"
            style = f"bold reverse {PRIMARY}" if q == self.current else (SUCCESS if q in self.answers else MUTED)
            line.append(f" {box} {self._tab_title(question)} ", style=style)
            line.append(" ")
        ready = len(self.answers) == len(self.questions)
        style = f"bold reverse {PRIMARY}" if self.on_submit_tab else (SUCCESS if ready else DIM)
        line.append(" ✓ Submit ", style=style)
        return line

    def _option_line(self, q: int, index: int) -> Text:
        highlighted = index == self.highlight[q]
        line = Text("❯ " if highlighted else "  ", style=f"bold {PRIMARY}")
        if self._is_other(q, index):
            if self._is_custom(q):
                line.append(f"{index + 1}. ✎ {self.answers[q]}", style=f"bold {SUCCESS}")
                line.append(" ✓", style=SUCCESS)
            elif self.typing and highlighted:
                line.append(f"{index + 1}. ✎ Other", style=f"bold {PRIMARY}")
            else:
                line.append(f"{index + 1}. ✎ Other", style=f"bold {TEXT}" if highlighted else TEXT)
                line.append(" — type your own answer", style=DIM)
            return line
        label, description = self.questions[q].options[index]
        chosen = self.answers.get(q) == self.questions[q].values[index]
        style = f"bold {SUCCESS}" if chosen else (f"bold {TEXT}" if highlighted else TEXT)
        line.append(f"{index + 1}. {label}", style=style)
        if chosen:
            line.append(" ✓", style=SUCCESS)
        if description:
            line.append(f" — {description}", style=DIM)
        return line

    def _review(self) -> Text:
        body = Text("Review your answers\n", style=f"bold {TEXT}")
        for q, question in enumerate(self.questions):
            shown = self._display(q)
            body.append(f"\n  {question.text}\n", style=TEXT)
            body.append("   → ", style=DIM)
            body.append(shown or "not answered yet", style=f"bold {SUCCESS}" if shown else WARNING)
        return body

    def _summary(self) -> Text:
        """``? question`` then ``⎿ answer`` under it, for every question."""
        body = Text()
        if self.cancelled:
            body.append("? ", style=f"bold {PRIMARY}")
            body.append(f"{_plural(len(self.questions), 'question')} · cancelled", style=MUTED)
            return body
        for q, question in enumerate(self.questions):
            if q:
                body.append("\n")
            body.append("? ", style=f"bold {PRIMARY}")
            body.append(question.text, style=f"bold {TEXT}")
            body.append("\n  ⎿  ", style=DIM)
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
        if self.done or not self.multi:
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
            self.set_class(True, "-typing")
            self.redraw()
            field = self.query_one(Input)
            field.value = self.answers[q] if self._is_custom(q) else ""
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

    def _answer(self, q: int, value: str) -> None:
        self._close_field()
        self.answers[q] = value
        if not self.multi:
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


def tool_summary(tool_name: str, parameters: dict[str, t.Any]) -> str:
    """The one argument that says what the call does, not the whole payload."""
    for key in ("command", "file_path", "path", "pattern", "query", "url", "city", "to"):
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
    if isinstance(value, dict) and "output" in value and "exit_code" in value:
        return str(value["output"]).rstrip() or "(no output)"
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

    DEFAULT_CSS = f"""
    ToolBlock {{ height: auto; margin: 1 0 0 0; }}
    ToolBlock:hover > .summary {{ color: {TEXT}; }}
    ToolBlock > .summary {{ color: {MUTED}; padding-left: 2; }}
    ToolBlock > .output {{
        display: none; color: {TEXT}; padding: 0 1; margin-left: 5;
        border-left: solid {BORDER}; max-height: 20; overflow-y: auto;
    }}
    ToolBlock.-expanded > .output {{ display: block; }}
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
            dot = Text(SPINNER[self._frame % len(SPINNER)] + " ", style=f"bold {PRIMARY}")
        elif self._interrupted:
            dot = Text("◇ ", style=f"bold {DIM}")
        elif self.result.success and not self._nonzero_exit():
            dot = Text("◆ ", style=f"bold {SUCCESS}")
        else:
            dot = Text("◆ ", style=f"bold {ERROR}")
        dot.append(self.tool_name, style="bold")
        summary = tool_summary(self.tool_name, self.parameters)
        if summary:
            if len(summary) > 90:
                summary = summary[:87] + "…"
            dot.append(f"({summary})", style=MUTED)
        return dot

    def _summary_line(self) -> Text:
        if self.result is None:
            if self._waiting:
                return Text("╰─ Waiting for approval…", style=WARNING)
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
            summary.append("cancelled", style=MUTED)
        elif not self.result.success:
            summary.append(f"Error: {text.splitlines()[0][:120] if text else 'failed'}", style=ERROR)
        else:
            code = self.result.result.get("exit_code") if isinstance(self.result.result, dict) else None
            if code not in (None, 0):
                summary.append(f"exit {code} · ", style=ERROR)
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
