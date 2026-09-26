"""Vacation tools (example logic)."""

import datetime as dt
import hashlib
import uuid

from mcp.types import ToolAnnotations

from src._app import mcp, store
from src.auth.jwt_handler import current_employee


def _working_days(start_date: str, end_date: str) -> int:
    start, end = dt.date.fromisoformat(start_date), dt.date.fromisoformat(end_date)
    return sum((start + dt.timedelta(n)).weekday() < 5 for n in range((end - start).days + 1))


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def calculate_vacation_days(start_date: str, end_date: str) -> dict:
    """Count the working days (Monday to Friday) between two dates, YYYY-MM-DD."""
    return {"working_days": _working_days(start_date, end_date)}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_vacation_balance() -> dict:
    """Your vacation days this year: granted, pending approval and available."""
    employee = current_employee()
    # Simulated: the same number (11-24) for each employee on every run.
    granted = 11 + int(hashlib.sha256(employee.encode()).hexdigest(), 16) % 14
    pending = sum(_working_days(r["start_date"], r["end_date"])
                  for r in store.get(employee, {}).values() if "start_date" in r)
    return {"granted": granted, "pending": pending, "available": granted - pending}


@mcp.tool()
def submit_vacation(start_date: str, end_date: str, note: str = "") -> dict:
    """Submit a vacation request to your manager; it stays pending until approved."""
    request_id = f"ACME-{uuid.uuid4().hex[:6].upper()}"
    store.setdefault(current_employee(), {})[request_id] = {
        "start_date": start_date, "end_date": end_date, "note": note, "status": "pending",
    }
    return {"request_id": request_id, "status": "pending"}
