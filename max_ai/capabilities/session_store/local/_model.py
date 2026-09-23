"""Serializable configuration for LocalSessionStore."""

from __future__ import annotations

from pydantic import BaseModel


class LocalSessionStoreConfig(BaseModel):
    base_path: str
