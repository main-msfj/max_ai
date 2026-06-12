"""
Scratchpad data model for agent task trackin
"""

from __future__ import annotations

import uuid
import typing as t
from pydantic import BaseModel, Field


class TodoItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str
    status: t.Literal["pending", "in_progress", "done", "skipped"] = "pending"


class Scratchpad(BaseModel):
    items: list[TodoItem] = Field(default_factory=list)

    def upsert(self, id: str, description: str, status: str) -> None:
        """Update existing item or append new one."""
        for item in self.items:
            if item.id == id:
                item.description = description
                item.status = status  # type: ignore[assignment]
                return
        self.items.append(TodoItem(id=id, description=description, status=status))  # type: ignore[arg-type]
