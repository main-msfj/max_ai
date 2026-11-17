"""The agent's list of things to do, kept in the run context."""

import typing as t

from pydantic import BaseModel


class Todo(BaseModel):
    """One thing to do and how far it is."""

    content: str
    status: t.Literal["pending", "in_progress", "done"] = "pending"
