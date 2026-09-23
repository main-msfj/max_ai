"""Serializable Docker executor settings."""

from typing import Literal

from pydantic import BaseModel, Field


class DockerExecutorConfig(BaseModel):
    image: str = "maxai-runtime:latest"
    network: Literal["none", "unrestricted"] = "none"
    max_output_bytes: int = Field(default=1 << 20, gt=0)
