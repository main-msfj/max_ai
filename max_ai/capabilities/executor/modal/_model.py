"""Serializable Modal executor settings."""

from typing import Literal

from pydantic import BaseModel, Field


class ModalExecutorConfig(BaseModel):
    """Configuration options for ``ModalExecutor``."""
    image: str | None = Field(default=None, description="Registry image; None builds the default.")
    packages: list[str] = Field(default_factory=list, description="Extra pip packages (default image).")
    app_name: str = "maxai-runtime"
    network: Literal["none", "unrestricted"] = "none"
    lifetime: int = Field(default=3600, ge=1, le=86400)
    max_output_bytes: int = Field(default=1 << 20, gt=0)
    uid: int = Field(default=1000, gt=0, description="Non-root user that runs every command.")
