"""MaxAI terminal palette, logo and the two color effects (gradient, shimmer), in the spirit of Grok Build's GrokNight theme.

Gray base, one lilac accent; the other colors only carry meaning (ok, error,
path, command). Change a value once and the whole interface follows.
"""

from rich.text import Text

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
PRIMARY = "#bb9af7"  # caret, spinner, running, assistant
SUCCESS = "#9ece6a"
WARNING = "#e0af68"
ERROR = "#f7768e"

# -------- MEANING (for blocks) ----------------------------------------------
SYSTEM = "#7aa2f7"  # harness and system notices
PATH = "#ff9e64"  # file paths in tool headers
COMMAND = "#e0af68"  # shell commands
MODEL = "#1abc9c"  # model name in the status line
CODE = "#7dcfff"  # inline code
GRADIENT = ("#bb9af7", "#7aa2f7", "#7dcfff")  # logo: lilac → blue → cyan

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
