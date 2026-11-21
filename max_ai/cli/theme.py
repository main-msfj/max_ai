"""MaxAI terminal palette, logo and the two color effects (gradient, shimmer), in the spirit of Grok Build's GrokNight theme.

Gray base, one accent the user picks with ``/theme``; green, amber and red only say how something went
(ok, warning, error). Paths and commands are plain bright text. Change a value once and the whole interface follows.
"""

import json
import os
import typing as t
from pathlib import Path

from rich.text import Text

from ..config import setting

# -------- SURFACES -----------------------------------------------------------
BACKGROUND = "#141414"
PANEL = "#111111"  # sidebar, cards
ELEMENT = "#242424"  # user messages, hovered buttons
BORDER_SUBTLE = "#323237"  # prompt border without focus
BORDER = "#505058"  # prompt border with focus

# -------- TEXT -----------------------------------------------------------
TEXT = "#e1e1e1"
MUTED = "#787878"  # folded tools, secondary lines
DIM = "#585858"  # details: (3 matches), timings, hints

# -------- ACCENTS -----------------------------------------------------------
class Theme:
    """The accent of the interface and the logo's colors, by name. The
    choice is read once, when the CLI starts: every block takes its colors
    then, so a new choice shows the next time it is opened."""

    # name: (accent, the logo from left to right)
    THEMES: t.ClassVar[dict[str, tuple[str, tuple[str, ...]]]] = {
        "gray": ("#e1e1e1", ("#ffffff", "#c8c8c8", "#8a8a8a")),
        "blue": ("#7aa2f7", ("#7aa2f7", "#7dcfff", "#9ad5ff")),
        "teal": ("#1abc9c", ("#1abc9c", "#4fd1c5", "#7dcfff")),
        "lilac": ("#bb9af7", ("#bb9af7", "#7aa2f7", "#7dcfff")),
    }
    DEFAULT: t.ClassVar[str] = "gray"
    ENV: t.ClassVar[str] = "MAX_AI_THEME"  # tries a theme for one run, without saving it
    FILE: t.ClassVar[str] = "cli.json"  # in ~/.max_ai: what the user chose for the CLI

    @classmethod
    def path(cls) -> Path:
        return setting.max_ai_home.expanduser() / cls.FILE

    @classmethod
    def chosen(cls) -> str:
        """The theme to use: the variable, else the saved one, else gray.
        A name that is not a theme, or a file that cannot be read, is gray."""
        name = os.environ.get(cls.ENV, "")
        if not name:
            try:
                name = str(json.loads(cls.path().read_text(encoding="utf-8")).get("theme", ""))
            except (OSError, ValueError, AttributeError):
                name = ""
        return name if name in cls.THEMES else cls.DEFAULT

    @classmethod
    def save(cls, name: str) -> None:
        """Keep ``name`` as the user's theme. It must be one of ``THEMES``."""
        if name not in cls.THEMES:
            raise ValueError(f"{name} is not a theme: {', '.join(cls.THEMES)}")
        path = cls.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"theme": name}, indent=2) + "\n", encoding="utf-8")


PRIMARY, _LOGO_COLORS = Theme.THEMES[Theme.chosen()]  # caret, spinner, running, assistant
SUCCESS = "#9ece6a"
WARNING = "#e0af68"
ERROR = "#f7768e"

# -------- MEANING (for blocks) ----------------------------------------------
SYSTEM = "#7aa2f7"  # harness and system notices
PATH = TEXT  # file paths in tool headers
COMMAND = TEXT  # shell commands
MODEL = "#1abc9c"  # model name in the status line
CODE = "#7dcfff"  # inline code
GRADIENT = _LOGO_COLORS  # the logo, left to right

# -------- EFFECTS -----------------------------------------------------------
def blend(start: str, end: str, amount: float) -> str:
    """The color ``amount`` (0 to 1) of the way from ``start`` to ``end``."""
    a, b = int(start[1:], 16), int(end[1:], 16)
    mixed = [
        round(((a >> shift) & 255) + (((b >> shift) & 255) - ((a >> shift) & 255)) * amount)
        for shift in (16, 8, 0)
    ]
    return "#" + "".join(f"{part:02x}" for part in mixed)


def gradient(text: str, colors: tuple[str, ...] = GRADIENT, bold: bool = True) -> Text:
    """``text`` painted left to right through ``colors``."""
    painted = Text()
    steps = max(len(text) - 1, 1)
    for index, char in enumerate(text):
        where = index / steps * (len(colors) - 1)
        low = min(int(where), len(colors) - 2)
        color = blend(colors[low], colors[low + 1], where - low)
        painted.append(char, style=f"bold {color}" if bold else color)
    return painted


def shimmer(text: str, frame: int, base: str = MUTED, light: str = PRIMARY) -> Text:
    """``text`` with a light that sweeps across it, one step per frame."""
    painted = Text()
    width = len(text) + 6
    center = frame % width - 3
    for index, char in enumerate(text):
        amount = max(0.0, 1 - abs(index - center) / 3)
        painted.append(char, style=f"bold {blend(base, light, amount)}")
    return painted


# -------- LOGO -----------------------------------------------------------
# Each row is (MAX part, -AI part); the first is drawn muted, the second bright.
LOGO = (
    ("█▀▄▀█ █▀▀█ ▀▄ ▄▀", "      █▀▀█ ▀█▀"),
    ("█░▀░█ █▀▀█  ░█░ ", "  ▀▀▀ █▀▀█ ░█░"),
    ("▀   ▀ ▀  ▀ ▄▀ ▀▄", "      ▀  ▀ ▀▀▀"),
)
