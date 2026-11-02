"""Serializable Docker executor settings."""

from typing import Literal

from pydantic import BaseModel, Field


class DockerExecutorConfig(BaseModel):
    """Configuration options for ``DockerExecutor``."""
    image: str | None = Field(default=None, description="Your image; None uses max_ai's runtime.")
    dockerfile: str | None = Field(default=None, description="Path to your own Dockerfile.")
    network: Literal["internet", "none"] = Field(
        default="none", description="Docker cannot filter by domain: no allow_list.")
    max_output_bytes: int = Field(default=1 << 20, gt=0)
