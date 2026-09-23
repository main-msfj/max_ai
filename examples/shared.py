"""What the numbered examples (01_, 02_, ...) share: fixture paths, two demo
tools and the ``--session`` argument. Each example builds its own Agent."""

from __future__ import annotations

import argparse
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
LOCAL_DIR = EXAMPLES_DIR / "local"  # fixture memory, knowledge and sessions
USER_ID = "user_001"


def get_weather(city: str) -> dict[str, str | int]:
    """Return the current weather for a city. Read-only, no side effects."""
    return {"city": city, "condition": "soleado", "temp_c": 24}


def send_email(to: str, subject: str, body: str) -> dict[str, str | bool]:
    """Send an email to someone. Has a real external side effect."""
    print(f"[send_email] (simulado) para={to!r} asunto={subject!r} cuerpo={body!r}")
    return {"sent": True, "to": to}


def session_arg() -> str | None:
    """``--session <id>`` continues a saved conversation."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session")
    return parser.parse_known_args()[0].session
