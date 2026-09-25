"""Serializable Docker executor settings."""

from typing import Literal

from pydantic import BaseModel, Field


class DockerExecutorConfig(BaseModel):
    """Configuration options for ``DockerExecutor``."""
    image: str = "maxai-runtime:latest"
    network: Literal["internet", "none"] = Field(
        default="none", description="Docker cannot filter by domain: no allow_list.")
    max_output_bytes: int = Field(default=1 << 20, gt=0)
