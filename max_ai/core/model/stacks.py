"""Prompt stack (layer) configuration."""

import typing as t

from pydantic import BaseModel, Field


class StackConfig(BaseModel):
    """Core Stack Config"""

    name: str = Field(...)
    instructions: str | None = Field(default=None)
    description: str = Field(...)
    version: str = Field(default="1.0.0")
    is_edited: bool = Field(default=False)
    layer_class: str = Field(...)
    template: str | None = Field(default=None)
    load_from: str | None = Field(default=None)
    extra_variables: dict[str, t.Any] = Field(default_factory=dict)
