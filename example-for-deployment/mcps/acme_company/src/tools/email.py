"""Email tool (example: nothing is really sent)."""

from src._app import mcp


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    return {"status": "sent", "to": to, "subject": subject}
