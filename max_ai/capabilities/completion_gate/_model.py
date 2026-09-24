"""Serializable options of the framework's completion gate."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RuntimeGateConfig(BaseModel):
    """Which closure checks the runtime gate runs before a turn may end."""

    enabled: bool = Field(default=True, description="False: turns close without checks.")
    plan_must_close: bool = Field(
        default=True,
        description="Nudge once when the model closes with pending plan steps.",
    )
    check_bash_outputs: bool = Field(
        default=True,
        description="Files a successful bash command declared must exist.",
    )
    nudge_bash_failures: bool = Field(
        default=True,
        description="Stop once on a failed bash command so the model fixes it or says so.",
    )
