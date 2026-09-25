"""Serializable Modal executor settings."""

from typing import Literal

from pydantic import BaseModel, Field

from ....core.executor.remote import DEFAULT_FRAMEWORK, PACKAGE_REGISTRIES  # noqa: F401


class ModalExecutorConfig(BaseModel):
    """Configuration options for ``ModalExecutor``."""
    image: str | None = Field(default=None, description="Registry image; None uses max_ai's runtime.")
    dockerfile: str | None = Field(default=None, description="Path to your own Dockerfile.")
    framework: str = Field(default=DEFAULT_FRAMEWORK, description="pip requirement for max_ai.")
    packages: list[str] = Field(default_factory=list, description="Extra pip packages baked into the image.")
    app_name: str = "maxai-runtime"
    network: Literal["packages", "internet", "none"] = Field(
        default="packages",
        description="packages: package registries + allow_list; internet: everything, "
                    "or only allow_list if given; none: no network.",
    )
    allow_list: list[str] = Field(default_factory=list, description="Extra domains, '*.' wildcards allowed.")
    lifetime: int = Field(default=3600, ge=1, le=86400)
    max_output_bytes: int = Field(default=1 << 20, gt=0)
    uid: int = Field(default=1000, gt=0, description="Non-root user that runs every command.")
