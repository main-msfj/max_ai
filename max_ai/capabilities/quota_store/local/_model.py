"""Serializable configuration for LocalQuotaStore."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LocalQuotaStoreConfig(BaseModel):
    base_path: str
    keep_periods: int = Field(default=60, ge=1, description="Older periods are dropped.")
