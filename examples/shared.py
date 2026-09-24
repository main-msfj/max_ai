"""Shared paths, example tools, and the ``--session`` argument."""

from __future__ import annotations

import argparse
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
LOCAL_DIR = EXAMPLES_DIR / "local"  # fixture memory, knowledge and sessions
USER_ID = "user_001"


def calculate_compound_interest(principal: float, annual_rate: float, years: int) -> float:
    """Calculate a balance with annual compound interest."""
    return round(principal * (1 + annual_rate / 100) ** years, 2)


def save_report(report: str) -> str:
    """Save a report to a local file."""
    path = LOCAL_DIR / "reports" / "report.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return f"Report saved to {path}"


def session_arg() -> str | None:
    """``--session <id>`` continues a saved conversation."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session")
    return parser.parse_known_args()[0].session
