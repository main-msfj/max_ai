"""Serializable Modal executor settings."""

from typing import Literal

from pydantic import BaseModel, Field


class ModalExecutorConfig(BaseModel):
    image: str
    app_name: str = "maxai-runtime"
    network: Literal["none", "unrestricted"] = "none"
    lifetime: int = Field(default=3600, ge=1, le=86400)
    max_output_bytes: int = Field(default=1 << 20, gt=0)
