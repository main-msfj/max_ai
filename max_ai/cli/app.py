"""Interactive Textual terminal UI for a MaxAI agent.

The transcript is a stack of widgets (not a text log), so each piece of a
turn stays live: thinking folds into a one-line box, tool calls update in
place from running to done, and any block can be expanded with a click.
"""

from __future__ import annotations

import asyncio
import re
import shlex
import time
import typing as t
from datetime import UTC, datetime
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, DirectoryTree, OptionList, Static, TextArea

from ..agents.agent import Agent
from ..base.loop import LoopState
from ..base.sandbox import SandboxCore
from ..base.session_store import CoreSessionStore
from ..base.tools import CoreTool
from ..capabilities.loop import ReactLoop
from ..capabilities.mcp.tool import MCPTool
from ..capabilities.sandbox.local import LocalSandbox
from ..capabilities.tools.ask_user import AskUserTool
from ..capabilities.tools.todos import WriteTodosTool
from ..core.event_type import (
    CompactionEvent,
    CoreEvent,
    ErrorEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelRetryEvent,
    ModelStreamChunkEvent,
    TodosEvent,
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)
from ..core.harness import messages as harness
from ..core.harness.permissions import Permissions
from ..core.ids import short_id
from ..core.messages import HARNESS_SOURCE, HARNESS_TAG, AssistantMessage, UserMessage
from ..core.model.session import SessionInfo
from ..core.tokenizer import default_counter
from ..core.tool import ToolDispatcher
from ..types.run_context import RunContext
from ..types.todos import Todo
from ..types.tool_call import ToolCallRecord, ToolResult
from .blocks import (
    BASH,
    SPINNER,
    AssistantBlock,
    CompactionBlock,
    NoteLine,
    PastSummaryBlock,
    Question,
    QuestionForm,
    ThinkingBlock,
    TodoBlock,
    ToolBlock,
    UserBlock,
    WelcomeBox,
    _k,  # pyright: ignore[reportPrivateUsage]
    context_bar,
    time_ago,
    tool_summary,
)
from .events import event_line
from .theme import (
    BACKGROUND,
    BORDER,
    BORDER_SUBTLE,
    DIM,
    ELEMENT,
    ERROR,
    MUTED,
    PANEL,
    PRIMARY,
    SUCCESS,
    TEXT,
    WARNING,
)
from .widgets import CommandMenu, InfoPanel, PromptEditor

COMMANDS = {
    "/help": "show commands and shortcuts",
    "/skills": "list the agent's skills",
    "/tools": "list the tools the agent can call",
    "/mcp": "list the MCP servers with their tools",
    "/resume": "pick a saved conversation and continue it",
    "/new": "start a new conversation (the current one stays saved)",
    "/session": "show this conversation's id and how to resume it",
    "/clear": "start a new conversation (same as /new)",
    "/thinking": "show or hide thinking boxes",
    "/verbose": "show or hide runtime events",
    "/files": "toggle the file sidebar",
    "/exit": "quit",
}

PLACEHOLDER = 'Try "summarize this project"  ·  @file  ·  !cmd  ·  /help'
FORM_PLACEHOLDER = "↑↓ ←→ enter · 1-9 to pick · or type your own answer"
APPROVAL_PLACEHOLDER = "Pick an option (click, or type its number) or write a new message…"

# Rendered as their own blocks instead of a generic ToolBlock.
_ASK_USER = AskUserTool.TOOL_NAME
_WRITE_TODOS = WriteTodosTool.TOOL_NAME

# Harness events worth a line even outside verbose mode.
_ALWAYS_SHOWN = {"error"}
# Folded into the end-of-turn summary line, or shown as an error already.
_NEVER_SHOWN = {
    "user_input_request", "tool_call", "tool_call_response", "tool_approval",
    "reasoning_complete", "fatal_error",
}
# Seconds a ``!command`` may run.
_SHELL_TIMEOUT = 120.0


class Transcript(VerticalScroll):
    """Never takes focus: clicking a block must not steal it from the prompt."""

    can_focus = False


class MaxAIApp(App[None]):
    """A coding-agent TUI: block transcript, live status, fold-able details."""

    TITLE = "MaxAI"
    CSS = f"""
    Screen {{ background: {BACKGROUND}; }}
    #body {{ height: 1fr; }}
    #sidebar {{ width: 34; background: {PANEL}; border-right: solid {BORDER_SUBTLE}; padding: 0 1; display: none; }}
    .sidebar-toggle {{ height: 1; width: 1fr; margin-top: 1; padding: 0; border: none; background: {PANEL}; color: {TEXT}; text-align: left; }}
    #workspace-files, #skill-files {{ height: 1fr; background: {PANEL}; border: none; color: {TEXT}; }}
    #main {{ width: 1fr; padding: 0 1; }}
    #transcript {{ height: 1fr; background: {BACKGROUND}; scrollbar-size: 1 1; padding: 0 1; }}
    #request-card {{ height: auto; display: none; border: round {PRIMARY}; padding: 0 1; margin: 1 0 0 0; }}
    #request-detail {{ height: auto; color: {TEXT}; margin-bottom: 1; }}
    #decisions {{ height: auto; display: none; }}
    #decisions Button {{ width: 1fr; height: 1; min-height: 1; border: none; margin: 0; background: {PANEL}; color: {TEXT}; content-align: left middle; text-align: left; }}
    #decisions Button:hover, #decisions Button:focus {{ background: {ELEMENT}; color: {PRIMARY}; }}
    #status {{ height: 1; margin-top: 1; padding: 0 1; }}
    #composer {{ height: auto; border: round {BORDER}; padding: 0 1; }}
    #composer:focus-within {{ border: round {PRIMARY}; }}
    #caret {{ width: 2; color: {PRIMARY}; text-style: bold; }}
    #prompt {{ height: auto; min-height: 1; max-height: 10; border: none; background: {BACKGROUND}; padding: 0; }}
    #usage {{ height: 2; padding: 0 1 1 1; color: {DIM}; }}
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
        store: CoreSessionStore | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.show_thinking = show_thinking
        self.verbose = False
        # The CLI is a host: it loads, runs and saves; the agent keeps nothing.
        self.store = store
        self.user_id = user_id or (initial_context.user_id if initial_context else "user")
        self._resume_id = session_id if initial_context is None else None
        self.context: RunContext = initial_context or self._new_context(session_id)
        self._request_future: asyncio.Future[str] | None = None
        self._busy = False
        self._tokens_input = 0
        self._tokens_output = 0
        self._tokens_cached = 0
        self._compaction: CompactionBlock | None = None
        self._turn_started_at: float | None = None
        self._turn_tokens = 0
        self._frame = 0
        self._waiting = False
        # Shown instead of "Thinking…" while the harness waits to call the model again.
        self._retry: ModelRetryEvent | None = None
        # The agent's run of this moment; esc cancels it.
        self._job: asyncio.Task[LoopState] | None = None
        self._interrupting = False  # esc asked to stop: a cancel that is ours
        self._agent_name = agent.name
        self._workspace_open = True
        self._skills_open = True
        self._form: QuestionForm | None = None
        self._todo_call_ids: set[str] = set()
        self._todos: TodoBlock | None = None
        self._skills: list[tuple[str, str]] = []  # name and description
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
                yield InfoPanel(id="info")
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
        # 1. The banner, and the MCP servers the agent was given.
        self.query_one("#prompt", TextArea).focus()
        await self._mount(WelcomeBox(self._agent_name, self.agent.client.model, self.workspace_root))
        servers = [server.server_id for server in self.agent.mcp]
        if servers:
            line = Text("MCP ", style="bold")
            line.append(" · ".join(servers), style=MUTED)
            line.append("  /mcp for details", style=DIM)
            await self._mount(NoteLine(line))
        if self.agent.sandbox is not None and not self._is_local:
            await self._write_system(
                "The sandbox is not on this computer: the file sidebar and @file are off.",
            )
        # 2. The session to open, then the skills (they may be downloaded).
        if self.store is not None:
            await self._open_session_on_start()
        self._refresh_usage()
        self._refresh_status()
        self.set_interval(0.12, self._tick)
        if self.agent.skills is not None:
            self._load_skills()

    @work(group="skills")
    async def _load_skills(self) -> None:
        """The agent's skills with their descriptions, installed in the user's sandbox."""
        skills, sandbox = self.agent.skills, self.agent.sandbox
        if skills is None or sandbox is None:
            return
        try:
            box = sandbox.bind(self.user_id, skills.variables())
            await box.prepare()
            await skills.sync(box)
            installed = await box.installed_skills()
        except Exception as error:  # noqa: BLE001 — a bad skill must not kill the UI
            await self._write_system(f"Could not load skills: {error}", style=ERROR)
            return
        self._skills = [
            (name, installed.get(name, {}).get("description", "")) for name in skills.skills
        ]

    @property
    def _is_local(self) -> bool:
        """The agent's files are on this computer: no sandbox, or a local one."""
        return self.agent.sandbox is None or isinstance(self.agent.sandbox, LocalSandbox)

    def _box(self) -> SandboxCore | None:
        """The agent's sandbox bound to this user; None without one."""
        sandbox = self.agent.sandbox
        return sandbox.bind(self.user_id) if sandbox is not None else None

    @property
    def workspace_root(self) -> Path:
        """Where this session's files are: its folder in a local sandbox,
        else the folder the program runs in."""
        box = self._box()
        if box is None or not self._is_local:
            return Path.cwd()
        root = Path(box.session_paths(self.context.session_id or self.context.run_id)["workspace"])
        root.mkdir(parents=True, exist_ok=True)
        return root

    @property
    def skills_root(self) -> Path:
        box = self._box()
        if box is None or not self._is_local:
            return Path.cwd()
        root = Path(str(box.layout.skills))
        root.mkdir(parents=True, exist_ok=True)
        return root

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

    async def _write_system(self, message: str, style: str = DIM) -> None:
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
            line = Text("⏸ Waiting for your answer… ", style=f"bold {WARNING}")
            line.append("(", style=MUTED)
            line.append("esc", style=f"bold {MUTED}")
            line.append(" to cancel the turn)", style=MUTED)
            status.update(line)
            return
        elapsed = int(time.monotonic() - (self._turn_started_at or time.monotonic()))
        if self._retry is not None:
            retry = self._retry
            line = Text(f"↻ Retrying in {retry.delay:.0f}s ", style=f"bold {WARNING}")
            line.append(f"({retry.reason.replace('_', ' ')}, try {retry.attempt + 1}"
                        f"/{retry.max_attempts}) ", style=MUTED)
        else:
            line = Text(f"{SPINNER[self._frame % len(SPINNER)]} ", style=f"bold {PRIMARY}")
            line.append("Thinking… ", style=f"bold {TEXT}")
        line.append(f"({elapsed}s · ↓ {self._turn_tokens:,} tokens · ", style=MUTED)
        line.append("esc", style=f"bold {MUTED}")
        line.append(" to interrupt)", style=MUTED)
        status.update(line)

    def _refresh_usage(self) -> None:
        """Tokens spent, the window used and where compaction starts."""
        # 1. Tokens of this CLI's life.
        line = Text()
        if self.verbose:
            line.append("verbose · ", style=PRIMARY)
        line.append(f"{self._agent_name} · in {_k(self._tokens_input)} · out {_k(self._tokens_output)}",
                    style=DIM)
        if self._tokens_cached:
            line.append(f" · cached {_k(self._tokens_cached)}", style=DIM)
        # 2. The window as the last model call shared it out.
        parts = self.context.context
        window = (parts.window if parts else 0) or self.agent.client.config.max_context_window
        used = parts.prompt + parts.tools + parts.messages if parts else 0
        line.append(" · ctx ", style=DIM)
        line.append_text(context_bar(used, window))
        # 3. Compaction starts when the messages fill their room.
        name = type(self.agent.reasoning.compaction).__name__.removesuffix("Compaction").lower()
        line.append(f" · {name} compaction", style=DIM)
        if parts is not None and parts.window:
            share = round((parts.window - parts.output - parts.margin) / parts.window * 100)
            line.append(f" at ~{share}%", style=DIM)
        usage = self._static("#usage")
        if usage is not None:
            usage.update(line)

    # -------- ACTIONS -----------------------------------------------------------
    async def action_clear_chat(self) -> None:
        await self._transcript().remove_children()
        self._tools.clear()
        self._todos = None
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
        # 1. Esc first closes what is open on top: panel, typing, menu.
        if self._info().display:
            self._info().hide()
            return
        if self._form is not None and self._form.typing:
            self._form.stop_typing()
            return
        if self._menu().display:
            self._close_menu()
            return
        # 2. Then it stops the run, or the wait for an answer; what ran stays.
        job, future = self._job, self._request_future
        if job is not None and not job.done():
            self._interrupting = True
            job.cancel()
        elif future is not None and not future.done():
            self._interrupting = True
            future.cancel()

    async def action_submit_prompt(self) -> None:
        waiting = self._request_future is not None and not self._request_future.done()
        prompt = self.query_one("#prompt", TextArea)
        selected = self._menu().selected if self._menu().display else None
        if selected is not None:
            # Enter on a highlighted entry runs it, keeping any typed args.
            typed = prompt.text.strip().split(maxsplit=1)
            prompt.text = selected[0] + (f" {typed[1]}" if len(typed) > 1 else "")
            self._close_menu()
        value = prompt.text.strip()
        # A message during a run goes to the inbox; the loop reads it on its next call.
        if self._busy and not waiting:
            if self._job is None or not value or value.startswith(("/", "!")):
                return
            prompt.text = ""
            await self._write_user(value)
            self.context.inbox.append(self._expand_file_mentions(value))
            return
        prompt.text = ""
        if value:
            self._info().hide()
        if waiting and self._form is not None:
            self._form.enter(value or None)
            if not self._form.typing:
                prompt.placeholder = FORM_PLACEHOLDER
            return
        if not value:
            return
        if waiting and self._request_future is not None:
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
        self._retry = None
        self._turn_tokens = 0
        self._todos = None
        self._refresh_status()

    def _start_turn(self, task: str | None) -> None:
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
        entries += [(f"/{name}", description, "skill") for name, description in self._skills]
        return entries

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Open the slash menu while typing a ``/command`` (before any space)."""
        text = event.text_area.text
        if self._pick_by_number(event.text_area, text):
            return
        if self._form is not None:
            return
        if not text.startswith("/") or "\n" in text or " " in text:
            self._close_menu()
            return
        matches = [entry for entry in self._menu_entries() if entry[0].startswith(text.lower())]
        if not matches:
            self._close_menu()
            return
        self._menu().show(matches, PRIMARY)
        self.query_one("#prompt", PromptEditor).menu_open = True

    def _pick_by_number(self, prompt: TextArea, text: str) -> bool:
        """While a question or approval waits, a lone digit picks that option."""
        future = self._request_future
        if len(text) != 1 or not text.isdigit() or future is None or future.done():
            return False
        number = int(text)
        if self._form is not None:
            picked = self._form.pick_number(number)
        else:
            buttons = list(self.query_one("#decisions", Vertical).query(Button))
            picked = 1 <= number <= len(buttons)
            if picked:
                future.set_result(re.sub(r"^\d+\.\s+", "", str(buttons[number - 1].label)))
        if picked:
            prompt.text = ""
        return picked

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

    def on_question_form_typing_finished(self, event: QuestionForm.TypingFinished) -> None:
        prompt = self.query_one("#prompt", PromptEditor)
        prompt.placeholder = FORM_PLACEHOLDER
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
        # 1. A skill's name runs a turn that asks for that skill.
        skill = next((name for name, _ in self._skills if f"/{name}" == command), None)
        if skill is not None:
            await self._write_user(value)
            task = f'Use the "{skill}" skill.' + (f" {args.strip()}" if args.strip() else "")
            self._start_turn(task)
            return
        # 2. The CLI's own commands.
        if command == "/help":
            self._show("Help", self._help())
        elif command == "/skills":
            if not self._skills:
                self._show("Skills", Text("This agent has no skills.", style=MUTED))
            else:
                body = Text(f"Run one with /name, e.g. /{self._skills[0][0]}\n\n", style=DIM)
                for name, description in self._skills:
                    body.append(f"  /{name:<18}", style=f"bold {PRIMARY}")
                    body.append(f"{description}\n", style=MUTED)
                self._show("Skills", body)
        elif command == "/tools":
            body = Text()
            for tool in await self._all_tools():
                body.append(f"  {tool.name:<22}", style=f"bold {PRIMARY}")
                if self._asks(tool):
                    body.append("asks approval · ", style=WARNING)
                body.append(f"{self._first_line(tool.description)}\n", style=MUTED)
            self._show("Tools", body)
        elif command == "/mcp":
            await self._show_mcp()
        elif command in ("/new", "/clear"):
            await self._switch_to(self._new_context())
            await self._write_system(f"New conversation · session {self.context.session_id}.")
        elif command == "/session":
            await self._write_system(self._session_line())
        elif command == "/resume":
            if self.store is None:
                await self._write_system("No session store configured: nothing is saved.",
                                         style=WARNING)
            else:
                self._resume_picker(args.strip() or None)
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
            await self._write_system(f"Unknown command {command} — try /help.", style=ERROR)

    @staticmethod
    def _help() -> Text:
        """The commands and the keys, for /help."""
        body = Text("Commands\n", style="bold")
        for cmd, desc in COMMANDS.items():
            body.append(f"  {cmd:<12}", style=f"bold {PRIMARY}")
            body.append(f"{desc}\n", style=MUTED)
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
            ("1-9", "pick an option of a question or approval"),
            ("click", "expand/collapse thinking and tool output; run a menu command"),
        ):
            body.append(f"  {keys:<12}", style=f"bold {PRIMARY}")
            body.append(f"{desc}\n", style=MUTED)
        return body

    async def _all_tools(self) -> list[CoreTool]:
        """Every tool of a run, MCP ones included; the agent's own if MCP fails."""
        try:
            registry = await self.agent._registry()  # pyright: ignore[reportPrivateUsage]
        except Exception as error:  # noqa: BLE001 — a server down must not kill the UI
            await self._write_system(f"Could not reach the MCP servers: {error}", style=WARNING)
            return list(self.agent.tools)
        return registry.all_tools()

    @staticmethod
    def _asks(tool: CoreTool) -> bool:
        """The tool asks before it runs, unless setting.json allows it."""
        return "ask" in str(getattr(tool.approval_mode, "value", tool.approval_mode)).lower()

    @staticmethod
    def _first_line(text: str | None) -> str:
        lines = (text or "").strip().splitlines()
        return lines[0][:80] if lines else ""

    async def _show_mcp(self) -> None:
        """/mcp: each server with its tools, connecting them if needed."""
        servers = [server.server_id for server in self.agent.mcp]
        if not servers:
            self._show("MCP servers", Text("This agent has no MCP servers.", style=MUTED))
            return
        tools = [tool for tool in await self._all_tools() if isinstance(tool, MCPTool)]
        body = Text()
        for sid in servers:
            body.append(f"  {sid}\n", style=f"bold {PRIMARY}")
            mine = [tool for tool in tools if tool.server_id == sid]
            if not mine:
                body.append("    not connected\n", style=WARNING)
            for tool in mine:
                ask = self._asks(tool)
                body.append(f"    {tool.name.removeprefix(sid + '_'):<26}", style="bold")
                body.append("asks approval\n" if ask else "auto\n", style=WARNING if ask else DIM)
        self._show("MCP servers", body)

    def _info(self) -> InfoPanel:
        return self.query_one("#info", InfoPanel)

    def _show(self, title: str, body: Text) -> None:
        """A command's output in a panel above the composer, not in the transcript."""
        self._info().show(title, body)

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """A click on the slash menu: commands run, skills wait for their task."""
        if event.option_list is not self._menu():
            return
        command, _, kind = self._menu().entries[event.option_index]
        prompt = self.query_one("#prompt", TextArea)
        self._close_menu()
        if kind == "command":
            prompt.text = ""
            await self._run_command(command)
        else:
            prompt.text = command + " "
            prompt.move_cursor(prompt.document.end)
        prompt.focus()

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
    def _format_parameters(parameters: dict[str, t.Any]) -> str:
        """Render approval arguments as readable key/value lines, never JSON."""
        if not parameters:
            return "No parameters"
        lines: list[str] = []

        def add_value(name: str, value: t.Any) -> None:
            if isinstance(value, dict):
                items = t.cast("dict[str, t.Any]", value)
                if not items:
                    lines.append(f"{name}: (empty)")
                for child, nested in items.items():
                    add_value(f"{name}.{child}", nested)
            elif isinstance(value, (list, tuple)):
                values = list(t.cast("list[t.Any]", value))
                if not values:
                    lines.append(f"{name}: (empty)")
                elif any(isinstance(item, (dict, list, tuple)) for item in values):
                    for index, item in enumerate(values, start=1):
                        add_value(f"{name} {index}", item)
                else:
                    lines.append(f"{name}: {', '.join(str(item) for item in values)}")
            else:
                rendered = str(value).replace("\n", " ")
                if len(rendered) > 240:
                    rendered = rendered[:237] + "..."
                lines.append(f"{name}: {rendered or '(empty)'}")

        for key, value in parameters.items():
            add_value(str(key).replace("_", " "), value)
        return "\n".join(lines)

    @staticmethod
    def _approval_card(record: ToolCallRecord) -> str:
        """What the approval card says. Bash: what the command does, then
        the exact command; any other tool: its arguments."""
        if record.tool_name != BASH:
            details = MaxAIApp._format_parameters(record.parameters)
            return f"Allow {record.tool_name}?\n\n{details}"
        description = str(record.parameters.get("description") or "").strip()
        command = str(record.parameters.get("command") or "").strip()
        head = f"Allow Bash?\n\n{description}\n\n" if description else "Allow Bash?\n\n"
        return head + "$ " + command.replace("\n", "\n  ")

    def _expand_file_mentions(self, value: str) -> str:
        """Attach readable workspace files referenced as ``@relative/path``."""
        if not self._is_local:
            return value
        root = self.workspace_root
        attachments: list[str] = []
        for mention in re.findall(r"(?<!\w)@([\w./-]+)", value):
            candidate = (root / mention).resolve()
            if not candidate.is_file() or root not in candidate.parents:
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

    async def _discard_partial(self) -> None:
        """Remove the answer and thinking streamed so far: the model call
        starts over and streams them again."""
        for block in (self._thinking, self._assistant):
            if block is not None:
                await block.remove()
        self._thinking = self._assistant = None

    async def _write_thinking(self, chunk: str) -> None:
        if self._thinking is None:
            block = ThinkingBlock()
            block.display = self.show_thinking
            self._thinking = block
            await self._mount(block)
        self._thinking.append(chunk)
        self._turn_tokens += default_counter().count_text(chunk)
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
        self._turn_tokens += default_counter().count_text(chunk)
        self._follow()

    async def _write_event(self, event: CoreEvent) -> None:
        """Draw one event of the run."""
        # 1. The model: retries, calls and answers.
        if isinstance(event, ModelStreamChunkEvent):
            self._retry = None
            if event.thinking:
                await self._write_thinking(event.thinking)
            if event.chunk and not event.is_final:
                await self._write_assistant_chunk(event.chunk)
            return
        if isinstance(event, ModelRetryEvent):
            if event.discard_partial:
                await self._discard_partial()
            await self._write_system(
                f"Model call failed ({event.reason}); retrying in {event.delay:.0f}s "
                f"(attempt {event.attempt + 1} of {event.max_attempts})."
            )
            self._retry = event
            self._refresh_status()
            return
        if isinstance(event, (ModelResponseEvent, ErrorEvent)) and self._retry is not None:
            self._retry = None
            self._refresh_status()
        if isinstance(event, ModelCallEvent):
            await self._finish_assistant()
            self._refresh_usage()
            return
        if isinstance(event, ModelResponseEvent):
            if event.usage is not None:
                self._tokens_input += event.usage.tokens_input
                self._tokens_output += event.usage.tokens_output
                self._tokens_cached += event.usage.tokens_cached
                self._refresh_usage()
            if event.response and self._assistant is None:
                await self._write_assistant_chunk(event.response)
            await self._finish_assistant()
            return
        # 2. The tools: their blocks, the list of todos and approvals.
        if isinstance(event, (ToolCallEvent, ToolApprovalEvent, ToolCallResponseEvent, TodosEvent)):
            await self._write_tool_event(event)
            return
        # 3. Compaction, and everything else as one line.
        if isinstance(event, CompactionEvent):
            await self._render_compaction(event)
            return
        if isinstance(event, ErrorEvent) and event.error_type == "compaction_failed":
            if self._compaction is not None:
                self._compaction.fail(event.error_message)
                self._compaction = None
                return
        etype = event.event_type
        if etype in _NEVER_SHOWN:
            return
        if etype in _ALWAYS_SHOWN or self.verbose:
            await self._mount(NoteLine(event_line(event)))

    async def _write_tool_event(
        self, event: ToolCallEvent | ToolApprovalEvent | ToolCallResponseEvent | TodosEvent
    ) -> None:
        """A tool call starts, waits for approval, or ends."""
        # 1. A call starts; one paused for approval comes again when it runs.
        if isinstance(event, ToolCallEvent):
            await self._finish_assistant()
            if event.tool_name == _ASK_USER:
                return
            if event.tool_name == _WRITE_TODOS:
                self._todo_call_ids.add(event.tool_call_id)
                return
            if event.tool_call_id in self._tools:
                self._tools[event.tool_call_id].set_waiting(False)
                return
            block = ToolBlock(event.tool_name, event.parameters)
            await self._mount(block)
            self._tools[event.tool_call_id] = block
            return
        # 2. It waits for the user.
        if isinstance(event, ToolApprovalEvent):
            waiting = self._tools.get(event.tool_call_id)
            if waiting is not None:
                waiting.set_waiting(True)
            return
        # 3. The list of todos changed.
        if isinstance(event, TodosEvent):
            await self._render_todos(event.todos)
            return
        # 4. It ended: its block shows how.
        if event.tool_call_id in self._todo_call_ids:
            self._todo_call_ids.discard(event.tool_call_id)
            result = event.tool_result
            if result is not None and not result.success:
                await self._write_system(f"{_WRITE_TODOS} failed: {result.error}", style=ERROR)
            return
        block = self._tools.pop(event.tool_call_id, None)
        if block is not None:
            block.complete(event.tool_result)

    def _close_finished_blocks(self) -> None:
        """Blocks whose calls were closed without an event (the user wrote
        instead of answering, a compaction dropped them) show how they ended."""
        records = self.context.tool_state.records
        for call_id, block in list(self._tools.items()):
            record = records.get(call_id)
            if record is None or record.is_consumed:
                block.complete(record.result if record is not None else None)
                del self._tools[call_id]

    async def _render_compaction(self, event: CompactionEvent) -> None:
        """One block per compaction: live at start, folded line at end."""
        await self._finish_assistant()
        if event.phase == "start":
            self._compaction = CompactionBlock(event.strategy)
            await self._mount(self._compaction)
            return
        block, self._compaction = self._compaction, None
        if not event.changed:
            if block is not None:
                await block.remove()  # triggered but nothing to do: no noise
            return
        if block is None:
            block = CompactionBlock(event.strategy)
            await self._mount(block)
        block.finish(event)
        self._refresh_usage()

    async def _render_todos(self, todos: list[Todo]) -> None:
        """Each update is a TodoBlock; consecutive updates with nothing in
        between refresh the same block, older ones fold to one line."""
        if not todos:
            return
        await self._finish_assistant()
        children = self._transcript().children
        if self._todos is not None and children and children[-1] is self._todos:
            self._todos.update_todos(todos)
            return
        if self._todos is not None:
            self._todos.collapse()
        self._todos = TodoBlock(todos)
        await self._mount(self._todos)

    async def _write_turn_summary(self, state: LoopState) -> None:
        """One line per turn: outcome, time, why it ended and what it cost.
        An answer the harness gave in the model's place is shown first."""
        await self._finish_assistant()
        # 1. A hook ended the turn with an answer of its own: nothing streamed it.
        message = state.message
        if message is not None and message.source == HARNESS_SOURCE and message.text().strip():
            block = AssistantBlock()
            await self._mount(block)
            await block.write(message.text())
            await block.finish()
        # 2. The line.
        elapsed = int(time.monotonic() - (self._turn_started_at or time.monotonic()))
        if state.finish_reason == "stop":
            line = Text(f"◆ Done in {elapsed}s", style=f"bold {SUCCESS}")
        else:
            color = ERROR if state.finish_reason in ("error", "cancelled") else WARNING
            line = Text(f"■ Stopped after {elapsed}s", style=f"bold {color}")
            line.append(f" · {state.finish_reason.replace('_', ' ')}", style=MUTED)
        line.append(f" · {state.iteration} model call{'s' if state.iteration != 1 else ''}",
                    style=MUTED)
        if state.compactions:
            line.append(f" · compacted {state.compactions}x", style=MUTED)
        if state.finish_reason == "length":
            line.append(
                " · the model's reply hit its output limit: ask for smaller steps", style=MUTED,
            )
        await self._mount(NoteLine(line))

    # -------- APPROVALS AND QUESTIONS -------------------------------------------------
    async def _resolve_requests(self) -> str | None:
        """Ask separately for every call that waits; never approve a batch.
        A text that is no option is a new message: it is returned and the
        calls still open are closed by the agent."""
        ctx = self.context
        open_calls = [r for r in ctx.tool_state.records.values() if not r.is_consumed]
        kept: list[str] = []  # rules kept with "always" in this batch
        # 1. Approvals, one by one.
        for record in [r for r in open_calls if r.is_pending_approval]:
            name = record.tool_name
            # A rule kept earlier in this batch may already cover this call.
            if kept and Permissions({"allow": kept}).decide(name, record.parameters) is True:
                await self.agent.approve(ctx, record.id)
                await self._write_system(f"{name}: approved (always allowed)", style=SUCCESS)
                continue
            answer = await self._ask_approval(record)
            if answer is None or isinstance(answer, str):
                return answer
            await self._apply_approval(record, answer, kept)
        # 2. Questions, all in one form.
        asked = [r for r in open_calls if r.is_awaiting_input]
        questions = [question for record in asked for question in self._questions(record)]
        if questions:
            answers = await self._ask_form(questions)
            for record in asked:
                record.store_result(AskUserTool.answer(record, answers.get(record.id, {})))
        return None

    async def _ask_approval(self, record: ToolCallRecord) -> int | str | None:
        """The option picked (0 once, 1 always, 2 deny), or a new message."""
        # 1. "Always" only when a rule can be kept and is not a dangerous call.
        keep = self.agent.sandbox is not None and Permissions.can_be_kept(
            record.tool_name, record.parameters
        )
        rule = ", ".join(Permissions.rules_for(record.tool_name, record.parameters))
        options = ["Yes, allow once"]
        if keep:
            options.append(f"Yes, always allow {rule}")
        options.append("No, deny")
        labeled = [f"{n}. {option}" for n, option in enumerate(options, start=1)]
        # 2. Ask until the answer is an option or a message.
        while True:
            answer = (await self._request(
                "", labeled, prompt_card=self._approval_card(record),
            )).strip()
            low = answer.lower()
            if low in {options[0].lower(), "allow once", "yes", "y", "1"}:
                return 0
            if keep and low in {options[1].lower(), "always", "a", "2"}:
                return 1
            if low in {options[-1].lower(), "deny", "no", "n", str(len(options))}:
                return 2
            if answer.isdigit():
                await self._write_system(f"Answer a number from 1 to {len(options)}.")
                continue
            await self._write_user(answer)
            return self._expand_file_mentions(answer)

    async def _apply_approval(self, record: ToolCallRecord, answer: int, kept: list[str]) -> None:
        """Approve or reject ``record`` as picked, and say it."""
        name = record.tool_name
        if answer == 2:
            record.reject()
            await self._write_system(f"{name}: denied", style=ERROR)
            return
        always = answer == 1
        try:
            saved = await self.agent.approve(self.context, record.id, always=always)
        except Exception as error:  # noqa: BLE001 — the setting.json write failed
            await self._write_system(f"{name}: approved once; the rule was not kept: {error}",
                                     style=WARNING)
            return
        if not always:
            await self._write_system(f"{name}: approved once", style=SUCCESS)
            return
        rules = Permissions.rules_for(name, record.parameters)
        kept.extend(rules)
        rule = ", ".join(rules)
        where = "kept in your setting.json" if saved else "for this call only"
        await self._write_system(f"{name}: always allowed {rule} ({where})", style=SUCCESS)

    @staticmethod
    def _questions(record: ToolCallRecord) -> list[Question]:
        """Every question of a waiting ask_user call, read from the call itself."""
        questions: list[Question] = []
        for item in t.cast("list[dict[str, t.Any]]", record.parameters.get("questions") or []):
            options = t.cast("list[dict[str, str]]", item.get("options") or [])
            questions.append(Question(
                record.id, item["question"],
                [(o["label"], o.get("description", "")) for o in options],
                [o["label"] for o in options],
                item.get("header"),
                bool(options and item.get("multi_select")),
            ))
        return questions

    # -------- SESSIONS -----------------------------------------------------------
    def _session_line(self) -> str:
        sid = self.context.session_id
        if self.store is None:
            return f"Session {sid} (not saved: no session store configured)."
        return f"Session {sid} · saved after every turn · resume it with /resume or --session {sid}"

    async def _save(self, ctx: RunContext) -> None:
        """Keep ``ctx`` in the store: after each batch of tools and when a turn ends."""
        if self.store is None:
            return
        try:
            await self.store.save(ctx)
        except Exception as error:  # noqa: BLE001 — a failed save must not kill the UI
            await self._write_system(f"Could not save the session: {error}", style=ERROR)

    async def _open_session_on_start(self) -> None:
        sid = self._resume_id
        if sid is None or self.store is None:
            await self._write_system(self._session_line())
            return
        try:
            ctx = await self.store.load(self.user_id, sid)
        except Exception as error:  # noqa: BLE001
            await self._write_system(f"Could not load session {sid}: {error}", style=ERROR)
            return
        if ctx is None:
            await self._write_system(f"No saved session {sid}: starting a new one with that id.")
            return
        await self._switch_to(ctx)
        await self._write_system(f"Resumed session {sid}.")
        self._answer_open_calls()

    async def _switch_to(self, ctx: RunContext) -> None:
        """Show ``ctx`` as the current conversation: clear and redraw it."""
        self.context = ctx
        await self._transcript().remove_children()
        self._tools.clear()
        self._todos = None
        await self._replay(ctx)
        self._refresh_usage()
        # The sidebar follows the session's folder.
        if self._is_local:
            self.query_one("#workspace-files", DirectoryTree).path = self.workspace_root

    def _answer_open_calls(self) -> None:
        """A resumed conversation that waits for approvals or answers asks them now."""
        if any(not record.is_consumed for record in self.context.tool_state.records.values()):
            self._begin_busy()
            self._run_turn(None, ask_first=True)

    async def _replay(self, ctx: RunContext) -> None:
        """Redraw a saved conversation from its messages (no model calls)."""
        if ctx.summary is not None:
            block = PastSummaryBlock(ctx.summary.text())
            await self._mount(block)
            block.collapse()
        for message in ctx.messages:
            if message.source == HARNESS_SOURCE or message.role in ("system", "tool"):
                continue
            if message.role == "user":
                await self._mount(UserBlock(message.text()))
            elif isinstance(message, AssistantMessage):
                if message.text().strip():
                    block = AssistantBlock()
                    await self._mount(block)
                    await block.write(message.text())
                    await block.finish()
                for call in message.tool_calls:
                    line = Text("  ● ", style=PRIMARY)
                    line.append("Bash" if call.tool_name == BASH else call.tool_name,
                                style=f"bold {TEXT}")
                    line.append(f"  {tool_summary(call.tool_name, call.parameters)}", style=DIM)
                    await self._mount(NoteLine(line))
        if ctx.todos:
            self._todos = TodoBlock(ctx.todos)
            await self._mount(self._todos)

    @work(exclusive=True, group="agent")
    async def _resume_picker(self, session_id: str | None) -> None:
        """``/resume [id]``: pick a saved session (same form as ask_user)."""
        store = self.store
        if store is None:
            return
        if session_id is None:
            sessions = [s for s in await store.list_sessions(self.user_id)
                        if s.session_id != self.context.session_id]
            if not sessions:
                await self._write_system("No other saved conversations yet.")
                return
            question = Question(
                record_id="resume", text="Resume which conversation?", header="Resume",
                options=[(s.title or "(untitled)", self._session_detail(s)) for s in sessions],
                values=[s.session_id for s in sessions],
            )
            self._busy, self._interrupting = True, False
            try:
                answers = (await self._ask_form([question])).get("resume", {})
            except asyncio.CancelledError:
                # esc cancels the wait; anything else (app exit) must propagate.
                if not self._interrupting:
                    raise
                await self._write_system("Resume cancelled.")
                return
            finally:
                self._busy, self._interrupting = False, False
                self._refresh_status()
            session_id = str(answers.get(question.text) or "").strip()
        ctx = await store.load(self.user_id, session_id) if session_id else None
        if ctx is None:
            await self._write_system(f"No saved session {session_id!r}.", style=ERROR)
            return
        await self._switch_to(ctx)
        await self._write_system(f"Resumed session {session_id}.")
        self._answer_open_calls()

    @staticmethod
    def _session_detail(info: SessionInfo) -> str:
        detail = f"{time_ago(info.updated_at)} · {info.message_count} msgs"
        if info.compactions:
            detail += f" · {info.compactions} compaction{'s' if info.compactions > 1 else ''}"
        return detail

    def _new_context(self, session_id: str | None = None) -> RunContext:
        return RunContext(user_id=self.user_id, session_id=session_id or short_id())

    # -------- AGENT TURN -----------------------------------------------------------
    @work(exclusive=True, group="agent")
    async def _run_turn(self, task: str | None, ask_first: bool = False) -> None:
        """One turn of the user, through every approval and question it asks."""
        self._busy, self._interrupting = True, False
        try:
            # 1. Runs follow while calls wait for the user; None goes on after an answer.
            waiting = ask_first
            while True:
                if waiting:
                    task = await self._resolve_requests()
                state = await self._run_once(task)
                self._close_finished_blocks()
                waiting = state.finish_reason == ReactLoop.WAITING
                if waiting:
                    # Written while the turn ran: the agent closes the calls and reads it.
                    if self.context.inbox:
                        waiting, task = False, None
                    continue
                await self._write_turn_summary(state)
                # 2. Written after the loop's last look: one more run reads it.
                if not self.context.inbox:
                    break
                task = None
        except asyncio.CancelledError:
            if not self._interrupting:
                raise
            await self._close_interrupted()
        except Exception as error:  # noqa: BLE001 — a failed turn must not kill the UI
            await self._write_system(f"Failed: {error}", style=ERROR)
        finally:
            await self._finish_assistant()
            self._close_finished_blocks()
            await self._save(self.context)
            self._job = None
            self._busy = self._waiting = self._interrupting = False
            self._request_future = None
            self._refresh_status()
            self._refresh_usage()
            self.query_one("#prompt", TextArea).focus()

    async def _run_once(self, task: str | None) -> LoopState:
        """One run of the agent, its events drawn as they come; esc cancels it."""
        state = LoopState()

        async def consume() -> LoopState:
            events = self.agent.run_stream_events(task, self.context, state=state, stream=True)
            answered = False  # tools answered since the last save
            async for event in events:
                # Saved at the model call after tools: by then every answer is in ctx.
                if isinstance(event, ToolCallResponseEvent):
                    answered = True
                elif isinstance(event, ModelCallEvent) and answered:
                    answered = False
                    await self._save(self.context)
                await self._write_event(event)
            return state

        self._job = asyncio.ensure_future(consume())
        try:
            return await self._job
        finally:
            self._job = None

    async def _close_interrupted(self) -> None:
        """Esc stopped the turn: every open call gets its answer and the
        model is told, so the conversation can go on."""
        ToolDispatcher.close_interrupted(self.context)
        self.context.messages.append(
            UserMessage(source=HARNESS_SOURCE, content=f"{HARNESS_TAG} {harness.TURN_INTERRUPTED}")
        )
        await self._finish_assistant()
        for block in self._tools.values():
            block.complete(None)
        self._tools.clear()
        await self._write_system(
            "Interrupted · what ran so far stays in the conversation.", style=WARNING,
        )

    @work(exclusive=True, group="shell")
    async def _run_shell(self, command: str) -> None:
        """Run a command entered with ``!`` in the session's workspace."""
        if not command:
            await self._write_system("Shell command is empty.")
            self._busy = False
            self._refresh_status()
            return
        self._busy = True
        block = ToolBlock(BASH, {"command": command, "description": "Your command"})
        await self._mount(block)
        self._tools["__shell__"] = block
        started = datetime.now(UTC)
        try:
            result = await self._shell(command)
            result = result.model_copy(update={"started_at": started, "completed_at": datetime.now(UTC)})
            self._tools.pop("__shell__", None)
            block.complete(result)
            block.set_expanded()
        except Exception as error:  # noqa: BLE001 — shown in the block
            self._tools.pop("__shell__", None)
            block.complete(ToolResult(success=False, error=str(error), tool_call_id="shell"))
        finally:
            self._busy = False
            self._refresh_status()
            self.query_one("#prompt", TextArea).focus()

    async def _shell(self, command: str) -> ToolResult:
        """``command`` in the agent's sandbox, in the session's folder; on
        this computer when the agent has no sandbox."""
        box = self._box()
        # 1. In the sandbox, from the session's folder.
        if box is not None:
            await box.prepare()
            folder = box.session_paths(self.context.session_id or self.context.run_id)["workspace"]
            script = f"mkdir -p {shlex.quote(folder)} && cd {shlex.quote(folder)} && {command}"
            ran = await box.execute_command(script, timeout=_SHELL_TIMEOUT)
            output = {"stdout": ran.stdout, "stderr": ran.stderr, "exit_code": ran.exit_code}
            return ToolResult.success_result("shell", output)
        # 2. Without one, here.
        process = await asyncio.create_subprocess_shell(
            command, cwd=str(self.workspace_root),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(process.communicate(), _SHELL_TIMEOUT)
        except (TimeoutError, asyncio.CancelledError):
            process.kill()  # nothing stays running after the CLI gave up on it
            raise
        output = {"stdout": out.decode(errors="replace"), "exit_code": process.returncode}
        return ToolResult.success_result("shell", output)

    async def _ask_form(self, questions: list[Question]) -> dict[str, dict[str, str | list[str]]]:
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
        input_widget.placeholder = APPROVAL_PLACEHOLDER
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


async def run_cli(
    agent: Agent,
    *,
    show_thinking: bool = True,
    initial_context: RunContext | None = None,
    store: CoreSessionStore | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Run the Textual interface for an already configured agent.

    With a ``store`` the CLI acts as a host: it saves the conversation after
    every batch of tools and every turn, and ``/resume`` (or ``session_id``)
    continues a saved one. ``show_thinking`` sets the initial state; ctrl+t
    or /thinking toggles it. ``initial_context`` pins ``user_id`` and
    ``session_id`` up front. The agent is closed when the CLI ends.
    """
    app = MaxAIApp(
        agent, show_thinking=show_thinking, initial_context=initial_context,
        store=store, user_id=user_id, session_id=session_id,
    )
    try:
        await app.run_async()
    finally:
        await agent.close()
