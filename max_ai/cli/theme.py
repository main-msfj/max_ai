"""MaxAI terminal palette and logo, in the spirit of OpenCode's dark theme.

Every color the UI draws comes from here: change a value once and the whole
interface follows.
"""

# -------- SURFACES -----------------------------------------------------------
BACKGROUND = "#0a0a0a"
PANEL = "#141414"
ELEMENT = "#1e1e1e"  # user messages, hovered buttons
BORDER_SUBTLE = "#3c3c3c"
BORDER = "#484848"

# -------- TEXT -----------------------------------------------------------
TEXT = "#eeeeee"
MUTED = "#a0a0a0"
DIM = "#6a6a6a"

# -------- ACCENTS -----------------------------------------------------------
PRIMARY = "#fab283"  # caret, spinner, focus, bullets
SECONDARY = "#5c9cf5"
ACCENT = "#9d7cd8"
SUCCESS = "#7fd88f"
WARNING = "#f5a742"
ERROR = "#e06c75"

# -------- LOGO -----------------------------------------------------------
# Each row is (MAX part, -AI part); the first is drawn muted, the second bright.
LOGO = (
    ("█▀▄▀█ █▀▀█ ▀▄ ▄▀", "      █▀▀█ ▀█▀"),
    ("█░▀░█ █▀▀█  ░█░ ", "  ▀▀▀ █▀▀█ ░█░"),
    ("▀   ▀ ▀  ▀ ▄▀ ▀▄", "      ▀  ▀ ▀▀▀"),
)
