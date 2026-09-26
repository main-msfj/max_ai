"""Parking tool (example logic)."""

from src._app import mcp, store
from src.auth.jwt_handler import current_employee


@mcp.tool()
def set_parking(spot: str, note: str = "") -> dict:
    """Remember where you parked today, for example "B1-03"."""
    store.setdefault(current_employee(), {})["parking"] = {"spot": spot, "note": note}
    return {"status": "saved", "spot": spot}
