"""Interactive Textual terminal UI for a MaxAI agent.

The transcript is a stack of widgets (not a text log), so each piece of a
turn stays live: thinking folds into a one-line box, tool calls update in
place from running to done, and any block can be expanded with a click.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, DirectoryTree, Static, TextArea

from ..agents.agent import Agent
from ..core.event_type import (
    CoreEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelStreamChunkEvent,
    PlanningEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)
from ..core.termination.cancellation import CancellationToken
from ..types.agent_response import AgentResponse
from ..types.run_context import RunContext
from ..types.tool_call import ToolResult
from .blocks import (
    ACCENT,
    SPINNER,
    AssistantBlock,
    NoteLine,
    PlanBlock,
    Question,
    QuestionForm,
    ThinkingBlock,
    ToolBlock,
    UserBlock,
    WelcomeBox,
)
from .events import event_line
from .widgets import CommandMenu, PromptEditor

COMMANDS = {
    "/help": "show commands and shortcuts",
    "/skills": "list the agent's skills",
    "/tools": "list the tools the agent can call",
    "/clear": "start a fresh conversation (same user/session ids)",
    "/thinking": "show or hide thinking boxes",
    "/verbose": "show or hide runtime events",
    "/files": "toggle the file sidebar",
    "/exit": "quit",
}

PLACEHOLDER = 'Try "summarize this project"  ·  @file  ·  !cmd  ·  /help'
FORM_PLACEHOLDER = "↑↓ ←→ and enter to answer · or just type your own answer"

# Rendered as their own fold-able blocks instead of a generic ToolBlock.
_ASK_USER = "ask_user"
_UPDATE_PLAN = "update_plan"

# Harness events worth a line even outside verbose mode.
_ALWAYS_SHOWN = {"error", "fatal_error", "compaction"}
# Folded into the single end-of-turn summary line instead.
_NEVER_SHOWN = {
    "user_input_request", "tool_call", "tool_call_response",
    "task_complete", "reasoning_complete", "completion_rejected",
}


class Transcript(VerticalScroll):
    """Never takes focus: clicking a block must not steal it from the prompt."""

    can_focus = False


class MaxAIApp(App[None]):
    """A coding-agent TUI: block transcript, live status, fold-able details."""

    TITLE = "MaxAI"
    CSS = f"""
    Screen {{ background: #0b0b0c; }}
    #body {{ height: 1fr; }}
    #sidebar {{ width: 34; background: #111113; border-right: solid #29292d; padding: 0 1; display: none; }}
    .sidebar-toggle {{ height: 1; width: 1fr; margin-top: 1; padding: 0; border: none; background: #111113; color: #d4d4d8; text-align: left; }}
    #workspace-files, #skill-files {{ height: 1fr; background: #111113; border: none; color: #d4d4d8; }}
    #main {{ width: 1fr; padding: 0 1; }}
    #transcript {{ height: 1fr; background: #0b0b0c; scrollbar-size: 1 1; padding: 0 1; }}
    #request-card {{ height: auto; display: none; border: round {ACCENT}; padding: 0 1; margin: 1 0 0 0; }}
    #request-detail {{ height: auto; color: #f4f4f5; margin-bottom: 1; }}
    #decisions {{ height: auto; display: none; }}
    #decisions Button {{ width: 1fr; height: 1; min-height: 1; border: none; margin: 0; background: #111113; color: #d4d4d8; content-align: left middle; text-align: left; }}
    #decisions Button:hover, #decisions Button:focus {{ background: #27272a; color: {ACCENT}; }}
    #status {{ height: 1; margin-top: 1; padding: 0 1; }}
    #composer {{ height: auto; border: round #3f3f46; padding: 0 1; }}
    #composer:focus-within {{ border: round {ACCENT}; }}
    #caret {{ width: 2; color: {ACCENT}; text-style: bold; }}
    #prompt {{ height: auto; min-height: 1; max-height: 10; border: none; background: #0b0b0c; padding: 0; }}
    #usage {{ height: 1; padding: 0 1; color: #71717a; }}
    """
    BINDINGS = [
        Binding("escape", "interrupt", "Interrupt", priority=True),
        Binding("ctrl+t", "toggle_thinking", "Thinking", priority=True),
        Binding("ctrl+o", "toggle_verbose", "Verbose", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Files"),
        Binding("ctrl+l", "clear_chat", "Clear screen"),
        Binding("pageup", "scroll_transcript(-1)", "Scroll up", show=False),
        Binding("pagedown", "scroll_transcript(1)", "Scroll down", show=False),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(
        self,
        agent: Agent,
        *,
        show_thinking: bool = True,
        initial_context: RunContext | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.show_thinking = show_thinking
        self.verbose = False
        self.context: RunContext | None = initial_context
        self._request_future: asyncio.Future[str] | None = None
        self._busy = False
        self._tokens_input = 0
        self._tokens_output = 0
        self._tokens_cached = 0
        self._context_tokens = 0
        self._turn_started_at: float | None = None
        self._turn_tokens = 0
        self._frame = 0
        self._waiting = False
        self._cancel: CancellationToken | None = None
        self._agent_name = getattr(agent, "name", "agent")
        self._workspace_open = True
        self._skills_open = True
        # ask_user params by tool_call_id; the questions render at the pause.
        self._ask_params: dict[str, dict] = {}
        self._form: QuestionForm | None = None
        self._plan_call_ids: set[str] = set()
        self._plan: PlanBlock | None = None
        self._gate_retries: list[str] = []
        self._skills: list = []
        self._tools: dict[str, ToolBlock] = {}
        self._thinking: ThinkingBlock | None = None
        self._assistant: AssistantBlock | None = None

    # -------- LAYOUT -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Button("▾ WORKSPACE", id="workspace-toggle", classes="sidebar-toggle")
                yield DirectoryTree(str(self.workspace_root), id="workspace-files")
                yield Button("▾ SKILLS", id="skills-toggle", classes="sidebar-toggle")
                yield DirectoryTree(str(self.skills_root), id="skill-files")
            with Vertical(id="main"):
                yield Transcript(id="transcript")
                with Vertical(id="request-card"):
                    yield Static("", id="request-detail")
                    yield Vertical(id="decisions")
                yield Static("", id="status")
                with Horizontal(id="composer"):
                    yield Static("❯", id="caret")
                    yield PromptEditor(
                        language="markdown", placeholder=PLACEHOLDER, id="prompt",
                        highlight_cursor_line=False,
                    )
                yield CommandMenu(id="menu")
                yield Static("", id="usage")

    async def on_mount(self) -> None:
        self.query_one("#prompt", TextArea).focus()
        model = getattr(getattr(self.agent, "client", None), "model", "unknown")
        await self._mount(WelcomeBox(self._agent_name, model, self.workspace_root))
        skills = getattr(self.agent, "skills", None)
        if skills is not None:
            try:
                self._skills = list(await skills.get_skills())
            except Exception as error:  # noqa: BLE001 — a bad skill dir must not kill the UI
                await self._write_system(f"Could not load skills: {error}", style="#f87171")
        self._refresh_usage()
        self._refresh_status()
        self.set_interval(0.12, self._tick)

    @property
    def project_root(self) -> Path:
        return self.workspace_root

    def _workspace_directory(self):
        materialize = getattr(self.agent.workspace, "materialize", None)
        if materialize is None:
            root = Path(self.agent.workspace.base_root).expanduser().resolve()
            skills = root / "skills"
            skills.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(workspace_dir=root, skill_dir=skills)
        ctx = self.context or RunContext()
        return materialize(ctx.user_id, ctx.session_id)

    @property
    def workspace_root(self) -> Path:
        return self._workspace_directory().workspace_dir

    @property
    def skills_root(self) -> Path:
        return self._workspace_directory().skill_dir

    # -------- TRANSCRIPT -----------------------------------------------------------
    def _transcript(self) -> Transcript:
        return self.query_one("#transcript", Transcript)

    async def _mount(self, widget: Widget) -> Widget:
        transcript = self._transcript()
        # Follow the tail only if the user hasn't scrolled up to read.
        follow = transcript.scroll_y >= transcript.max_scroll_y - 2
        await transcript.mount(widget)
        if follow:
            transcript.scroll_end(animate=False)
        return widget

    def _follow(self) -> None:
        transcript = self._transcript()
        if transcript.scroll_y >= transcript.max_scroll_y - 4:
            transcript.scroll_end(animate=False)

    async def _write_system(self, message: str, style: str = "#71717a") -> None:
        await self._mount(NoteLine(f"• {message}", style=style))

    async def _write_user(self, value: str) -> None:
        await self._mount(UserBlock(value))

    # -------- STATUS / USAGE -----------------------------------------------------------
    def _tick(self) -> None:
        self._frame += 1
        for block in self._tools.values():
            block.tick()
        self._refresh_status()

    def _static(self, selector: str) -> Static | None:
        """``None`` once the screen is gone — timers and a finishing worker
        can still fire while the app shuts down."""
        try:
            return self.query_one(selector, Static)
        except (NoMatches, ScreenStackError):
            return None

    def _refresh_status(self) -> None:
        status = self._static("#status")
        if status is None:
            return
        if not self._busy:
            status.display = False
            return
        status.display = True
        if self._waiting:
            line = Text("⏸ Waiting for your answer… ", style="bold #fbbf24")
            line.append("(", style="#a1a1aa")
            line.append("esc", style="bold #a1a1aa")
            line.append(" to cancel the turn)", style="#a1a1aa")
            status.update(line)
            return
        elapsed = int(time.monotonic() - (self._turn_started_at or time.monotonic()))
        line = Text(f"{SPINNER[self._frame % len(SPINNER)]} ", style=f"bold {ACCENT}")
        line.append("Thinking… ", style="bold #e4e4e7")
        line.append(f"({elapsed}s · ↓ {self._turn_tokens:,} tokens · ", style="#a1a1aa")
        line.append("esc", style="bold #a1a1aa")
        line.append(" to interrupt)", style="#a1a1aa")
        status.update(line)

    def _refresh_usage(self) -> None:
        client = getattr(self.agent, "client", None)
        maximum = getattr(getattr(client, "config", None), "max_context_window", 0) or 0
        total = self._tokens_input + self._tokens_output
        line = Text()
        if self.verbose:
            line.append("verbose · ", style=ACCENT)
        line.append(f"{self._agent_name} · in {self._tokens_input:,} · out {self._tokens_output:,}"
                    f" · cached {self._tokens_cached:,} · total {total:,}", style="#71717a")
        if maximum > 0:
            left = max(0, 100 - round(self._context_tokens / maximum * 100))
            line.append(f" · ctx {left}% left", style="#71717a")
        usage = self._static("#usage")
        if usage is not None:
            usage.update(line)

    # -------- ACTIONS -----------------------------------------------------------
    async def action_clear_chat(self) -> None:
        await self._transcript().remove_children()
        self._tools.clear()
        self._plan = None
        await self._write_system("Screen cleared (the conversation is kept — /clear resets it).")

    def action_scroll_transcript(self, direction: int) -> None:
        transcript = self._transcript()
        if direction < 0:
            transcript.scroll_page_up()
        else:
            transcript.scroll_page_down()

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display

    def action_toggle_thinking(self) -> None:
        self.show_thinking = not self.show_thinking
        for block in self.query(ThinkingBlock):
            block.display = self.show_thinking
        self._refresh_usage()

    async def action_toggle_verbose(self) -> None:
        self.verbose = not self.verbose
        self._refresh_usage()

    def action_interrupt(self) -> None:
        if self._menu().display:
            self._close_menu()
            return
        if not self._busy or self._cancel is None:
            return
        # Also while paused on a question/approval: the turn is cancelled
        # and rolled back like any other interrupt.
        self._cancel.cancel()
        if self._request_future is not None and not self._request_future.done():
            self._request_future.cancel()

    async def action_submit_prompt(self) -> None:
        waiting = self._request_future is not None and not self._request_future.done()
        prompt = self.query_one("#prompt", TextArea)
        selected = self._menu().selected if self._menu().display else None
        if selected is not None:
            # Enter on a highlighted entry runs it, keeping any typed args.
            typed = prompt.text.strip().split(maxsplit=1)
            prompt.text = selected[0] + (f" {typed[1]}" if len(typed) > 1 else "")
            self._close_menu()
        if self._busy and not waiting:
            return
        value = prompt.text.strip()
        prompt.text = ""
        if waiting and self._form is not None:
            self._form.enter(value or None)
            if not self._form.typing:
                prompt.placeholder = FORM_PLACEHOLDER
            return
        if not value:
            return
        if waiting:
            self._request_future.set_result(value)
            return
        if value.startswith("/"):
            await self._run_command(value)
            return
        await self._write_user(value)
        if value.startswith("!"):
            self._begin_busy()
            self._run_shell(value[1:].strip())
        else:
            self._start_turn(self._expand_file_mentions(value))

    def _begin_busy(self) -> None:
        self._busy = True
        self._turn_started_at = time.monotonic()
        self._turn_tokens = 0
        self._plan = None
        self._gate_retries = []
        self._refresh_status()

    def _start_turn(self, task: str) -> None:
        self._begin_busy()
        self._run_turn(task)

    async def on_prompt_editor_submitted(self, event: PromptEditor.Submitted) -> None:
        await self.action_submit_prompt()

    # -------- SLASH MENU -----------------------------------------------------------
    def _menu(self) -> CommandMenu:
        return self.query_one("#menu", CommandMenu)

    def _close_menu(self) -> None:
        self._menu().hide()
        self.query_one("#prompt", PromptEditor).menu_open = False

    def _menu_entries(self) -> list[tuple[str, str, str]]:
        entries = [(cmd, desc, "command") for cmd, desc in COMMANDS.items()]
        entries += [(f"/{skill.name}", skill.description, "skill") for skill in self._skills]
        return entries

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Open the slash menu while typing a ``/command`` (before any space)."""
        if self._form is not None:
            return
        text = event.text_area.text
        if not text.startswith("/") or "\n" in text or " " in text:
            self._close_menu()
            return
        matches = [entry for entry in self._menu_entries() if entry[0].startswith(text.lower())]
        if not matches:
            self._close_menu()
            return
        self._menu().show(matches, ACCENT)
        self.query_one("#prompt", PromptEditor).menu_open = True

    def on_prompt_editor_menu_move(self, event: PromptEditor.MenuMove) -> None:
        if self._form is not None:
            self._form.move(event.delta)
            self.query_one("#prompt", PromptEditor).placeholder = FORM_PLACEHOLDER
            return
        self._menu().move(event.delta)

    def on_prompt_editor_tab_move(self, event: PromptEditor.TabMove) -> None:
        if self._form is not None:
            self._form.switch(event.delta)
            self.query_one("#prompt", PromptEditor).placeholder = FORM_PLACEHOLDER

    def on_question_form_typing_requested(self, event: QuestionForm.TypingRequested) -> None:
        prompt = self.query_one("#prompt", PromptEditor)
        prompt.placeholder = f"Your answer to “{event.question[:60]}” — type it and press enter"
        prompt.focus()

    def on_question_form_completed(self, event: QuestionForm.Completed) -> None:
        future = self._request_future
        if event.form is self._form and future is not None and not future.done():
            future.set_result("")

    def on_prompt_editor_menu_complete(self, event: PromptEditor.MenuComplete) -> None:
        selected = self._menu().selected
        if selected is None:
            return
        prompt = self.query_one("#prompt", TextArea)
        prompt.text = selected[0] + " "
        prompt.move_cursor(prompt.document.end)
        self._close_menu()

    async def _run_command(self, value: str) -> None:
        self._close_menu()
        command, _, args = value.partition(" ")
        command = command.lower()
        skill = next((s for s in self._skills if f"/{s.name}" == command), None)
        if skill is not None:
            await self._write_user(value)
            task = f'Use the "{skill.name}" skill.' + (f" {args.strip()}" if args.strip() else "")
            self._start_turn(task)
            return
        if command == "/help":
            body = Text("Commands\n", style="bold")
            for cmd, desc in COMMANDS.items():
                body.append(f"  {cmd:<12}", style=f"bold {ACCENT}")
                body.append(f"{desc}\n", style="#a1a1aa")
            body.append("\nShortcuts\n", style="bold")
            for keys, desc in (
                ("enter", "send · shift+enter or ctrl+j for a new line"),
                ("esc", "interrupt the running turn"),
                ("ctrl+t", "thinking on/off"),
                ("ctrl+o", "verbose runtime events on/off"),
                ("ctrl+b", "file sidebar"),
                ("ctrl+l", "clear the screen"),
                ("pgup/pgdn", "scroll the conversation"),
                ("@path", "attach a workspace file"),
                ("!cmd", "run a shell command in the workspace"),
                ("/ ↑↓ tab", "open, move through and complete the command menu"),
                ("click", "expand/collapse thinking, plan, questions, tool output"),
            ):
                body.append(f"  {keys:<12}", style=f"bold {ACCENT}")
                body.append(f"{desc}\n", style="#a1a1aa")
            await self._mount(NoteLine(body))
        elif command == "/skills":
            if not self._skills:
                await self._write_system("This agent has no skills.")
            else:
                body = Text("Skills  ", style="bold")
                body.append("(run one with /name, e.g. " + f"/{self._skills[0].name})\n", style="#71717a")
                for item in self._skills:
                    body.append(f"  /{item.name:<18}", style=f"bold {ACCENT}")
                    body.append(f"{item.description}\n", style="#a1a1aa")
                await self._mount(NoteLine(body))
        elif command == "/tools":
            body = Text("Tools\n", style="bold")
            for tool in getattr(self.agent, "tools", []):
                mode = getattr(getattr(tool, "approval_mode", None), "value", "")
                first = (tool.description or "").strip().splitlines()[0][:80] if tool.description else ""
                body.append(f"  {tool.name:<22}", style=f"bold {ACCENT}")
                if "ask" in str(mode).lower():
                    body.append("asks approval · ", style="#fbbf24")
                body.append(f"{first}\n", style="#a1a1aa")
            await self._mount(NoteLine(body))
        elif command == "/clear":
            ctx = self.context
            self.context = RunContext(user_id=ctx.user_id, session_id=ctx.session_id) if ctx else None
            await self._transcript().remove_children()
            self._tools.clear()
            self._plan = None
            await self._write_system("New conversation.")
        elif command == "/thinking":
            self.action_toggle_thinking()
            await self._write_system(f"Thinking {'shown' if self.show_thinking else 'hidden'}.")
        elif command == "/verbose":
            await self.action_toggle_verbose()
            await self._write_system(f"Verbose {'on' if self.verbose else 'off'}.")
        elif command == "/files":
            self.action_toggle_sidebar()
        elif command in ("/exit", "/quit"):
            self.exit()
        else:
            await self._write_system(f"Unknown command {command} — try /help.", style="#f87171")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "workspace-toggle":
            self._workspace_open = not self._workspace_open
            self.query_one("#workspace-files").display = self._workspace_open
            event.button.label = f"{'▾' if self._workspace_open else '▸'} WORKSPACE"
            return
        if event.button.id == "skills-toggle":
            self._skills_open = not self._skills_open
            self.query_one("#skill-files").display = self._skills_open
            event.button.label = f"{'▾' if self._skills_open else '▸'} SKILLS"
            return
        if self._request_future is not None and not self._request_future.done():
            label = str(event.button.label)
            self._request_future.set_result(re.sub(r"^\d+\.\s+", "", label))

    # -------- HELPERS -----------------------------------------------------------
    @staticmethod
    def _format_parameters(parameters: dict) -> str:
        """Render approval arguments as readable key/value lines, never JSON."""
        if not parameters:
            return "No parameters"
        lines: list[str] = []

        def add_value(name: str, value) -> None:
            if isinstance(value, dict):
                if not value:
                    lines.append(f"{name}: (empty)")
                else:
                    for child, nested in value.items():
                        add_value(f"{name}.{child}", nested)
            elif isinstance(value, (list, tuple)):
                if not value:
                    lines.append(f"{name}: (empty)")
                elif any(isinstance(item, (dict, list, tuple)) for item in value):
                    for index, item in enumerate(value, start=1):
                        add_value(f"{name} {index}", item)
                else:
                    lines.append(f"{name}: {', '.join(str(item) for item in value)}")
            else:
                rendered = str(value).replace("\n", " ")
                if len(rendered) > 240:
                    rendered = rendered[:237] + "..."
                lines.append(f"{name}: {rendered or '(empty)'}")

        for key, value in parameters.items():
            add_value(str(key).replace("_", " "), value)
        return "\n".join(lines)

    def _expand_file_mentions(self, value: str) -> str:
        """Attach readable project files referenced as ``@relative/path``."""
        attachments: list[str] = []
        for mention in re.findall(r"(?<!\w)@([\w./-]+)", value):
            candidate = (self.project_root / mention).resolve()
            if not candidate.is_file() or self.project_root not in candidate.parents:
                continue
            try:
                content = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            attachments.append(f"\n\n--- {mention} ---\n\n{content}")
        return value + "".join(attachments)

    # -------- STREAM RENDERING -----------------------------------------------------------
    async def _finish_thinking(self) -> None:
        if self._thinking is not None:
            self._thinking.finish()
            self._thinking = None

    async def _finish_assistant(self) -> None:
        await self._finish_thinking()
        if self._assistant is not None:
            await self._assistant.finish()
            self._assistant = None

    async def _write_thinking(self, chunk: str) -> None:
        if self._thinking is None:
            block = ThinkingBlock()
            block.display = self.show_thinking
            self._thinking = block
            await self._mount(block)
        self._thinking.append(chunk)
        self._turn_tokens += max(1, len(chunk) // 4)
        self._follow()

    async def _write_assistant_chunk(self, chunk: str) -> None:
        # Models often stream a bare "\n\n" before a tool call; that alone
        # would mount an empty answer block (a blank gap in the transcript).
        if self._assistant is None and not chunk.strip():
            return
        await self._finish_thinking()
        if self._assistant is None:
            self._assistant = AssistantBlock()
            await self._mount(self._assistant)
        await self._assistant.write(chunk)
        self._turn_tokens += max(1, len(chunk) // 4)
        self._follow()

    async def _write_event(self, event: CoreEvent) -> None:
        if isinstance(event, PlanningEvent):
            await self._render_plan(event)
            return
        if isinstance(event, ModelCallEvent):
            await self._finish_assistant()
            return
        if isinstance(event, ModelResponseEvent):
            if event.usage is not None:
                self._tokens_input += event.usage.tokens_input
                self._tokens_output += event.usage.tokens_output
                self._tokens_cached += event.usage.tokens_cached
                self._context_tokens = event.usage.tokens_input
                self._refresh_usage()
            if event.response and self._assistant is None:
                await self._write_assistant_chunk(event.response)
            await self._finish_assistant()
            return
        if isinstance(event, ToolCallEvent):
            await self._finish_assistant()
            if event.tool_name == _ASK_USER:
                self._ask_params[event.tool_call_id] = event.parameters
                return
            if event.tool_name == _UPDATE_PLAN:
                self._plan_call_ids.add(event.tool_call_id)
                return
            # A call paused for approval is re-emitted on resume: same block.
            if event.tool_call_id in self._tools:
                self._tools[event.tool_call_id].set_waiting(False)
                return
            block = ToolBlock(event.tool_name, event.parameters)
            await self._mount(block)
            self._tools[event.tool_call_id] = block
            return
        if event.event_type == "tool_approval":
            block = self._tools.get(getattr(event, "tool_call_id", ""))
            if block is not None:
                block.set_waiting(True)
            return
        if isinstance(event, ToolCallResponseEvent):
            if event.tool_call_id in self._ask_params:
                return
            if event.tool_call_id in self._plan_call_ids:
                self._plan_call_ids.discard(event.tool_call_id)
                result = event.tool_result
                if result is not None and not result.success:
                    await self._write_system(f"update_plan failed: {result.error}", style="#f87171")
                return
            block = self._tools.pop(event.tool_call_id, None)
            if block is not None:
                block.complete(event.tool_result)
            return
        if event.event_type == "completion_rejected":
            reasons = getattr(getattr(event, "decision", None), "reasons", ()) or ()
            self._gate_retries.append("; ".join(reasons) or "incomplete")
            return
        etype = event.event_type
        if etype in _NEVER_SHOWN:
            return
        if etype in _ALWAYS_SHOWN or self.verbose:
            await self._mount(NoteLine(event_line(event)))

    async def _render_plan(self, event: PlanningEvent) -> None:
        """Each update is a PlanBlock; consecutive updates with nothing in
        between refresh the same block, older ones fold to one line."""
        plan = event.plan
        if plan is None or not plan.steps:
            return
        await self._finish_assistant()
        children = self._transcript().children
        if self._plan is not None and children and children[-1] is self._plan:
            self._plan.update_plan(plan.model_copy(deep=True))
            return
        if self._plan is not None:
            self._plan.collapse()
        self._plan = PlanBlock(plan.model_copy(deep=True))
        await self._mount(self._plan)

    async def _write_turn_summary(self, response: AgentResponse) -> None:
        """One line per turn: outcome, time, finish reason and anything the
        completion gate said (retries, closing notes, blocking reasons)."""
        await self._finish_assistant()
        elapsed = int(time.monotonic() - (self._turn_started_at or time.monotonic()))
        decision = response.completion
        status = decision.status if decision is not None else None
        reasons = list(decision.reasons) if decision is not None else []
        closed = response.finish_reason == "stop" and status in (None, "completed")
        warn = bool(reasons or self._gate_retries)
        if closed:
            icon, head = ("⚠", "Done") if warn else ("✓", "Done")
            color = "#fbbf24" if warn else "#4ade80"
            line = Text(f"{icon} {head} in {elapsed}s", style=f"bold {color}")
        else:
            color = "#f87171" if response.finish_reason in ("error", "cancelled") else "#fbbf24"
            line = Text(f"■ Stopped after {elapsed}s", style=f"bold {color}")
        line.append(f" · {response.finish_reason}", style="#a1a1aa")
        if status is not None and status != "completed":
            line.append(f" · gate {status}", style=color)
        if self._gate_retries:
            count = len(self._gate_retries)
            line.append(f" · gate retried {count}×: ", style="#fbbf24")
            line.append(" | ".join(self._gate_retries)[:200], style="#a1a1aa")
        if reasons:
            line.append(" · gate: ", style=color)
            line.append("; ".join(reasons)[:300], style="#a1a1aa")
        await self._mount(NoteLine(line))

    # -------- AGENT TURN -----------------------------------------------------------
    async def _resolve_requests(self, response: AgentResponse) -> None:
        """Ask separately for every pending operation; never approve a batch."""
        ctx = response.context
        if ctx is None:
            raise RuntimeError("Cannot resume without a run context")
        self.context = ctx
        for record in response.pending_approvals:
            details = self._format_parameters(record.parameters)
            while True:
                answer = await self._request(
                    "", ["1. Yes, allow once", "2. No, deny"],
                    prompt_card=f"Allow {record.tool_name}?\n\n{details}",
                )
                choice = answer.strip().lower()
                if choice in {"yes, allow once", "allow once", "yes", "y", "1"}:
                    approved = True
                    break
                if choice in {"no, deny", "deny", "no", "n", "2"}:
                    approved = False
                    break
                await self._write_system("Answer 1 (yes) or 2 (no).")
            ctx.tool_state.apply_approval(record.id, approved=approved)
            await self._write_system(
                f"{record.tool_name}: {'approved once' if approved else 'denied'}",
                style="#4ade80" if approved else "#f87171",
            )
        questions = [q for record in response.pending_questions for q in self._questions(record)]
        if questions:
            answers = await self._ask_form(questions)
            for record in response.pending_questions:
                ctx.tool_state.apply_user_answer(record.id, answers[record.id])

    def _questions(self, record) -> list[Question]:
        """Every question a pending ask_user record holds. Labels and
        descriptions come from the call's params when we saw them; the
        record's option strings are what gets sent back either way."""
        items = record.input_questions or [
            {"question": record.input_question, "header": None, "options": record.input_options}
        ]
        raw_items = self._ask_params.get(record.id, {}).get("questions") or []
        grouped = record.input_questions is not None and len(items) > 1
        questions: list[Question] = []
        for position, item in enumerate(items):
            values = list(item.get("options") or [])
            raw = raw_items[position].get("options") if position < len(raw_items) and isinstance(raw_items[position], dict) else None
            if isinstance(raw, list) and len(raw) == len(values) and all(isinstance(o, dict) for o in raw):
                options = [(str(o.get("label", v)), str(o.get("description", ""))) for o, v in zip(raw, values)]
            else:
                options = [(v.split(" — ", 1)[0], v.split(" — ", 1)[1] if " — " in v else "") for v in values]
            text = str(item.get("question") or "Choose an answer")
            questions.append(Question(record.id, text, options, values, item.get("header"), grouped))
        return questions

    def _ensure_context(self) -> RunContext:
        if self.context is None:
            ctx = RunContext()
            ctx.session_id = ctx.session_id or ctx.run_id
            self.context = ctx
        return self.context

    @work(exclusive=True, group="agent")
    async def _run_turn(self, task: str) -> None:
        self._busy = True
        ctx = self._ensure_context()
        # Interrupt discards the turn: roll the conversation back to here.
        snapshot = ctx.model_copy(deep=True)
        self._cancel = CancellationToken()
        try:
            response = await self._consume(
                self.agent.run_stream_events(
                    task=task,
                    run_context=ctx,
                    cancellation_token=self._cancel,
                    stream_tokens=True,
                )
            )
            while response is not None and (response.needs_approval or response.needs_input):
                await self._resolve_requests(response)
                response = await self._consume(
                    self.agent.resume_stream_events(
                        run_context=response.context,
                        cancellation_token=self._cancel,
                        stream_tokens=True,
                    )
                )
            if response is not None:
                self.context = response.context
                await self._write_turn_summary(response)
        except asyncio.CancelledError:
            if self._cancel is None or not self._cancel.is_cancelled():
                raise
            self.context = snapshot
            await self._finish_assistant()
            for block in self._tools.values():
                block.complete(None)
            self._tools.clear()
            await self._write_system(
                "Interrupted · turn discarded from the conversation "
                "(side effects that already ran are not undone).",
                style="#fbbf24",
            )
        except Exception as error:
            await self._write_system(f"Failed: {error}", style="#f87171")
        finally:
            await self._finish_assistant()
            self._cancel = None
            self._busy = False
            self._waiting = False
            self._request_future = None
            self._refresh_status()
            self.query_one("#prompt", TextArea).focus()

    @work(exclusive=True, group="shell")
    async def _run_shell(self, command: str) -> None:
        """Run a command entered with ``!`` in the configured workspace."""
        if not command:
            await self._write_system("Shell command is empty.")
            self._busy = False
            self._refresh_status()
            return
        self._busy = True
        block = ToolBlock("shell", {"command": command})
        await self._mount(block)
        self._tools["__shell__"] = block
        started = datetime.now(timezone.utc)
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await process.communicate()
            result = ToolResult(
                success=True,
                tool_call_id="shell",
                result={"stdout": output.decode(errors="replace"), "exit_code": process.returncode},
                started_at=started,
                completed_at=datetime.now(timezone.utc),
            )
            self._tools.pop("__shell__", None)
            block.complete(result)
            block.set_expanded()
        except Exception as error:
            self._tools.pop("__shell__", None)
            block.complete(ToolResult(success=False, error=str(error), tool_call_id="shell"))
        finally:
            self._busy = False
            self._refresh_status()
            self.query_one("#prompt", TextArea).focus()

    async def _consume(
        self, stream: AsyncIterator[CoreEvent | AgentResponse],
    ) -> AgentResponse | None:
        response: AgentResponse | None = None
        async for item in stream:
            if isinstance(item, AgentResponse):
                response = item
            elif isinstance(item, ModelStreamChunkEvent):
                if item.thinking:
                    await self._write_thinking(item.thinking)
                if item.chunk and not item.is_final:
                    await self._write_assistant_chunk(item.chunk)
            else:
                await self._write_event(item)
        return response

    async def _ask_form(self, questions: list[Question]) -> dict[str, str]:
        """All pending questions in one form; the prompt drives it (↑↓ ←→
        enter) and still takes a free-text answer."""
        await self._finish_assistant()
        self._close_menu()
        prompt = self.query_one("#prompt", PromptEditor)
        form = QuestionForm(questions)
        await self._mount(form)
        self._form = form
        prompt.menu_open = prompt.tabs_open = True
        prompt.placeholder = FORM_PLACEHOLDER
        self._request_future = asyncio.get_running_loop().create_future()
        self._waiting = True
        self._refresh_status()
        self._transcript().scroll_to_widget(form, animate=False)
        prompt.focus()
        try:
            await self._request_future
            return form.result()
        except asyncio.CancelledError:
            form.cancel()
            raise
        finally:
            self._request_future = None
            self._form = None
            self._waiting = False
            prompt.menu_open = prompt.tabs_open = False
            prompt.placeholder = PLACEHOLDER
            self._refresh_status()

    async def _request(
        self,
        label: str,
        options: list[str] | None = None,
        *,
        prompt_card: str | None = None,
    ) -> str:
        """Keep the composer usable while the agent waits for a human."""
        await self._finish_assistant()
        if label:
            await self._write_system(label)
        card = self.query_one("#request-card", Vertical)
        input_widget = self.query_one("#prompt", TextArea)
        input_widget.placeholder = "Pick an option (click, or type its number) or write an answer…"
        self._request_future = asyncio.get_running_loop().create_future()
        self._waiting = True
        self._refresh_status()
        detail = self.query_one("#request-detail", Static)
        decisions = self.query_one("#decisions", Vertical)
        await decisions.remove_children()
        detail.update(Text(prompt_card or label, style="bold"))
        card.display = bool(prompt_card or options)
        if options:
            await decisions.mount(*(Button(option) for option in options))
            decisions.display = True
        input_widget.focus()
        try:
            return await self._request_future
        finally:
            self._request_future = None
            self._waiting = False
            card.display = False
            detail.update("")
            decisions.display = False
            await decisions.remove_children()
            self._refresh_status()
            input_widget.placeholder = PLACEHOLDER


async def run_repl(
    agent: Agent,
    *,
    show_thinking: bool = True,
    initial_context: RunContext | None = None,
) -> None:
    """Run the Textual interface for an already configured agent.

    ``show_thinking`` sets the initial state; ctrl+t or /thinking toggles it
    at runtime. ``initial_context`` pins ``user_id``/``session_id`` up front —
    needed when memory/knowledge is bound to a specific session, since
    otherwise the first turn gets a random ``session_id``.
    """
    app = MaxAIApp(agent, show_thinking=show_thinking, initial_context=initial_context)
    await app.run_async()
