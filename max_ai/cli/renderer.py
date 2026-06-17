"""Render an agent's event stream to the terminal with Rich.

``CliRenderer`` is a small state machine: feed it ``CoreEvent``s as they
arrive from ``agent.run_stream_events`` and it prints them as a live,
readable transcript — streaming assistant text, dimmed reasoning, a plan
panel, tool-call lines, and a "thinking" spinner between model turns.

It owns no agent logic and no input: the REPL drives it. That split keeps
the renderer reusable (one-shot, tests) and the loop logic in one place.
"""

from __future__ import annotations

import typing as t

from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

from ..core.event_type import (
    CoreEvent,
    ErrorEvent,
    FatalErrorEvent,
    PlanningEvent,
    EvalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
    ModelStreamChunkEvent,
    UserInputRequestEvent,
    ReasoningCompleteEvent,
)
from ..reasoning.plan import AgentPlan


# Status glyphs for plan steps — kept ASCII-ish so they render anywhere.
_STEP_GLYPH = {
    "pending": "[ ]",
    "active": "[~]",
    "done": "[x]",
    "failed": "[!]",
}
_STEP_STYLE = {
    "pending": "dim",
    "active": "bold yellow",
    "done": "green",
    "failed": "bold red",
}


class CliRenderer:
    """Translate a stream of ``CoreEvent``s into terminal output.

    One instance per REPL session (it keeps a single ``Console``); call
    ``begin_turn`` before feeding a turn's events and ``handle`` for each
    event. State that must not leak across turns (streaming flags, the
    spinner) is reset in ``begin_turn``.
    """

    def __init__(self, console: Console | None = None, *, show_thinking: bool = True) -> None:
        self.console = console or Console()
        self.show_thinking = show_thinking

        # Per-turn streaming state.
        self._streaming_text = False  # have we printed any answer chunk yet?
        self._streaming_thinking = False
        self._spinner: Live | None = None

    # -------- TURN LIFECYCLE --------------------------------------------------
    def begin_turn(self) -> None:
        """Reset per-turn state and show the agent label + thinking spinner."""
        self._streaming_text = False
        self._streaming_thinking = False
        self.console.print()
        self.console.print(Text("agent", style="bold cyan"), end="  ")
        self._start_spinner()

    def end_turn(self) -> None:
        """Finish the turn: stop the spinner and close the streamed line."""
        self._stop_spinner()
        if self._streaming_text or self._streaming_thinking:
            self.console.print()  # newline after the streamed answer

    # -------- EVENT DISPATCH --------------------------------------------------
    def handle(self, event: CoreEvent) -> None:
        """Render a single event. Unknown event types are ignored."""
        if isinstance(event, ModelStreamChunkEvent):
            self._on_stream_chunk(event)
        elif isinstance(event, PlanningEvent):
            self._on_planning(event)
        elif isinstance(event, ToolCallEvent):
            self._on_tool_call(event)
        elif isinstance(event, ToolCallResponseEvent):
            self._on_tool_response(event)
        elif isinstance(event, EvalEvent):
            self._on_eval(event)
        elif isinstance(event, UserInputRequestEvent):
            self._on_user_input_request(event)
        elif isinstance(event, (ErrorEvent, FatalErrorEvent)):
            self._on_error(event)
        elif isinstance(event, ReasoningCompleteEvent):
            # Terminal marker for the loop; nothing to draw, but stop the
            # spinner in case no text streamed (e.g. a pure tool turn).
            self._stop_spinner()

    # -------- STREAMING TEXT / THINKING --------------------------------------
    def _on_stream_chunk(self, ev: ModelStreamChunkEvent) -> None:
        # Reasoning tokens (dimmed). Only when enabled and present.
        if ev.thinking and self.show_thinking:
            self._stop_spinner()
            if not self._streaming_thinking:
                self.console.print(Text("\nthinking ", style="dim italic"), end="")
                self._streaming_thinking = True
            self.console.print(Text(ev.thinking, style="dim italic"), end="")
            return

        if ev.is_final:
            return

        if ev.chunk:
            self._stop_spinner()
            # If we were mid-thinking, drop to a fresh line for the answer.
            if self._streaming_thinking and not self._streaming_text:
                self.console.print()
                self._streaming_thinking = False
            self._streaming_text = True
            self.console.print(Text(ev.chunk), end="")

    # -------- PLAN ------------------------------------------------------------
    def _on_planning(self, ev: PlanningEvent) -> None:
        if ev.plan is None:
            if ev.phase == "failed":
                self._aside("planning failed", style="red")
            return
        self._stop_spinner()
        self.console.print(self._plan_panel(ev.plan))

    def _plan_panel(self, plan: AgentPlan) -> Panel:
        body = Text()
        for step in plan.steps:
            glyph = _STEP_GLYPH.get(step.status, "[ ]")
            style = _STEP_STYLE.get(step.status, "white")
            body.append(f"{glyph} ", style=style)
            body.append(f"{step.id}. {step.description}\n", style=style)
        if plan.rationale:
            body.append(f"\n{plan.rationale}", style="dim italic")
        return Panel(body, title="plan", border_style="blue", expand=False)

    # -------- TOOLS -----------------------------------------------------------
    def _on_tool_call(self, ev: ToolCallEvent) -> None:
        self._stop_spinner()
        params = ", ".join(f"{k}={v!r}" for k, v in ev.parameters.items())
        if len(params) > 80:
            params = params[:80] + "…"
        line = Text("  → ", style="magenta")
        line.append(ev.tool_name, style="bold magenta")
        line.append(f"({params})", style="magenta")
        self.console.print(line)
        # A tool can take a while (web search, MCP calls). Show a live spinner
        # so the user knows work is happening, until the response arrives.
        self._start_spinner(f"running {ev.tool_name}…")

    def _on_tool_response(self, ev: ToolCallResponseEvent) -> None:
        self._stop_spinner()  # the tool finished — drop its spinner
        result = ev.tool_result
        if result is None:
            return
        ok = result.success
        glyph = "    ✓ " if ok else "    ✗ "
        style = "green" if ok else "red"
        preview = str(result.result if ok else result.error or "")
        preview = preview.replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:100] + "…"
        self.console.print(Text(glyph + preview, style=style))
        # The loop will call the model again to use this result — show the
        # thinking spinner until the next text/tool/finish event lands.
        self._start_spinner()

    # -------- EVAL ------------------------------------------------------------
    def _on_eval(self, ev: EvalEvent) -> None:
        if ev.phase == "complete":
            verdict = "passed" if ev.passed else "failed"
            style = "green" if ev.passed else "yellow"
            score = f" ({ev.score:.0%})" if ev.score is not None else ""
            self._aside(f"self-eval {verdict}{score}", style=style)
        elif ev.phase == "intermediate":
            self._aside("rethinking after a tool failure", style="yellow")

    # -------- HUMAN INPUT -----------------------------------------------------
    def show_question(self, question: str, options: list[str] | None) -> None:
        """Render an agent question. Called by the REPL when the loop pauses.

        The REPL prompts for the answer separately (see ``max_ai.cli.repl``);
        this only draws the question panel. ``UserInputRequestEvent`` arrives
        *after* the answer is given, so the REPL — not the event — drives this.
        """
        self._stop_spinner()
        self.console.print()
        body = Text(question, style="bold")
        if options:
            body.append("\n\n")
            for i, opt in enumerate(options, 1):
                body.append(f"  {i}. {opt}\n", style="cyan")
        self.console.print(
            Panel(body, title="the agent is asking", border_style="yellow", expand=False)
        )

    def _on_user_input_request(self, ev: UserInputRequestEvent) -> None:
        # The question panel is drawn by ``show_question`` when the loop pauses;
        # by the time this event arrives the answer has already been given, so
        # there is nothing to draw here. Kept as a no-op for completeness.
        pass

    # -------- APPROVAL --------------------------------------------------------
    def show_approval_request(self, tool_name: str, parameters: dict) -> None:
        """Render a tool the agent wants to run, pending the user's approval.

        Drawn by the REPL when a turn pauses for approval; the y/n answer is
        read separately (see ``max_ai.cli.repl``).
        """
        self._stop_spinner()
        self.console.print()
        params = ", ".join(f"{k}={v!r}" for k, v in parameters.items())
        if len(params) > 200:
            params = params[:200] + "…"
        body = Text()
        body.append("run ", style="dim")
        body.append(tool_name, style="bold magenta")
        body.append(f"({params})", style="magenta")
        self.console.print(
            Panel(body, title="approval needed", border_style="yellow", expand=False)
        )

    # -------- ERRORS ----------------------------------------------------------
    def _on_error(self, ev: CoreEvent) -> None:
        self.show_error(getattr(ev, "error_message", "unknown error"))

    def show_error(self, message: str) -> None:
        """Print a readable one-line error (no traceback)."""
        self._stop_spinner()
        # Collapse multi-line provider errors to keep the transcript readable.
        first_line = message.strip().splitlines()[0] if message.strip() else message
        if len(first_line) > 300:
            first_line = first_line[:300] + "…"
        self.console.print(Text(f"  error: {first_line}", style="bold red"))

    # -------- HELPERS ---------------------------------------------------------
    def _aside(self, text: str, *, style: str = "dim") -> None:
        """A small dimmed status line that doesn't interrupt the answer flow."""
        self._stop_spinner()
        self.console.print(Text(f"  {text}", style=style))

    def _start_spinner(self, label: str = "thinking…") -> None:
        # Restart with the new label if one is already running (e.g. switching
        # from "thinking" to "running web_search").
        if self._spinner is not None:
            self._stop_spinner()
        self._spinner = Live(
            Spinner("dots", text=Text(f" {label}", style="dim")),
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self._spinner.start()

    def _stop_spinner(self) -> None:
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def banner(self, agent_name: str) -> None:
        self.console.print(
            Panel(
                Text.assemble(
                    ("max_ai", "bold cyan"),
                    (" · chatting with ", "dim"),
                    (agent_name, "bold"),
                    ("\n\nType your message and press Enter. ", "dim"),
                    ("/exit", "yellow"),
                    (" or Ctrl-D to quit.", "dim"),
                ),
                border_style="cyan",
                expand=False,
            )
        )
