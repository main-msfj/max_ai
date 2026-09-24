"""What a session store knows about each saved conversation."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import AwareDatetime, BaseModel, Field


class SessionInfo(BaseModel):
    """Listing entry for a saved session (e.g. a ``/resume`` menu)."""

    user_id: str
    session_id: str
    title: str = Field(default="", description="First user message, shortened.")
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    message_count: int = 0
    compactions: int = 0
