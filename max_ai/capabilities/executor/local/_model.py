"""Serializable configuration for LocalExecutorConfig."""

from __future__ import annotations

from pydantic import BaseModel


class LocalExecutorConfig(BaseModel):
    max_output_bytes: int = 1 << 20
