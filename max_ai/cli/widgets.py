"""Input widgets for the terminal chat."""

from __future__ import annotations

from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.widgets import OptionList, TextArea
from textual.widgets.option_list import Option


class PromptEditor(TextArea):
    """Enter submits; Shift+Enter or Ctrl+J inserts a line break.

    While the slash menu is open, up/down move its highlight and tab
    completes; otherwise those keys fall through to normal editing.
    """

    BINDINGS = [
        Binding("enter,ctrl+enter", "submit", "Send", priority=True),
        Binding("shift+enter,ctrl+j", "newline", "New line", priority=True),
        Binding("up", "menu_move(-1)", show=False, priority=True),
        Binding("down", "menu_move(1)", show=False, priority=True),
        Binding("tab", "menu_complete", show=False, priority=True),
        Binding("left", "tab_move(-1)", show=False, priority=True),
        Binding("right", "tab_move(1)", show=False, priority=True),
    ]

    menu_open = False
    # Set while a question form is open: ←→ switch questions (empty prompt only).
    tabs_open = False

    class Submitted(Message):
        pass

    class MenuMove(Message):
        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    class MenuComplete(Message):
        pass

    class TabMove(Message):
        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    def action_submit(self) -> None:
        self.post_message(self.Submitted())

    def action_newline(self) -> None:
        self.insert("\n")

    # These bindings replace TextArea's own up/down/tab, so with the menu
    # closed they must do the normal editing themselves.
    def action_menu_move(self, delta: int) -> None:
        if self.menu_open:
            self.post_message(self.MenuMove(delta))
        elif delta < 0:
            self.action_cursor_up()
        else:
            self.action_cursor_down()

    def action_tab_move(self, delta: int) -> None:
        if self.tabs_open and not self.text:
            self.post_message(self.TabMove(delta))
        elif delta < 0:
            self.action_cursor_left()
        else:
            self.action_cursor_right()

    def action_menu_complete(self) -> None:
        if self.menu_open:
            self.post_message(self.MenuComplete())
        else:
            self.screen.focus_next()


class CommandMenu(OptionList):
    """Slash-command picker under the composer. Never focused: the prompt
    keeps the cursor and drives the highlight."""

    can_focus = False

    DEFAULT_CSS = """
    CommandMenu {
        height: auto; max-height: 10; display: none; border: none;
        background: #111113; padding: 0 1; scrollbar-size: 1 1;
    }
    CommandMenu > .option-list--option-highlighted { background: #27272a; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entries: list[tuple[str, str, str]] = []

    def show(self, entries: list[tuple[str, str, str]], accent: str) -> None:
        """``entries`` are ``(command, description, kind)`` tuples."""
        self.entries = entries
        self.clear_options()
        width = max(len(cmd) for cmd, _, _ in entries) + 3
        for cmd, desc, kind in entries:
            label = Text()
            label.append(f"{cmd:<{width}}", style=f"bold {accent}")
            if kind != "command":
                label.append(f"[{kind}] ", style="#71717a")
            first = desc.strip().splitlines()[0] if desc.strip() else ""
            label.append(first if len(first) <= 64 else first[:63] + "…", style="#a1a1aa")
            self.add_option(Option(label))
        self.highlighted = 0
        self.display = True

    def hide(self) -> None:
        self.display = False
        self.entries = []

    def move(self, delta: int) -> None:
        if not self.entries:
            return
        current = self.highlighted or 0
        self.highlighted = (current + delta) % len(self.entries)

    @property
    def selected(self) -> tuple[str, str, str] | None:
        if not self.entries or self.highlighted is None:
            return None
        return self.entries[self.highlighted]
