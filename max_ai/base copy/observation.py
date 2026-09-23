"""Data model for agent observations written during a run."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ObservationRecord(BaseModel):
    """A single observation written by the agent during a run.

    Observations are agent-generated notes — findings, errors,
    decisions — that persist across sessions so future runs can
    build on them without repeating the same mistakes or searches.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = Field(..., description="Session this observation belongs to")
    content: str = Field(..., description="The observation content")
    observation_type: Literal["finding", "error", "decision", "hypothesis"] = "finding"
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
