"""Short practical guides, read from data/."""

from src._app import mcp
from src.config import settings


@mcp.resource("acme://guides/vacation", mime_type="text/markdown")
def vacation_guide() -> str:
    """Checklist before you go on vacation."""
    return (settings.DATA_DIR / "vacation.md").read_text(encoding="utf-8")


@mcp.resource("acme://guides/parking", mime_type="text/markdown")
def parking_guide() -> str:
    """Practical tips for the office garage."""
    return (settings.DATA_DIR / "parking.md").read_text(encoding="utf-8")
