"""Runtime filesystem configuration for MaxAI.

The framework uses one server workspace as the root for runtime files
shared by local and container executors. Users can override the root
with ``SERVER_WORKSPACE`` (env var or .env file); otherwise it defaults
to the current working directory at startup. Subdirectories keep stable
names so Docker and local runtimes speak the same filesystem contract.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global Settings for Max AI Framework"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    root_dir: Path = Field(
        default_factory=Path.cwd,
        validation_alias="HOST_WORKSPACE",
        description="Rolt Directory for the host Server",
    )

    # Default Folder
    mtn_folder: str = Field(default="/mnt")
    tool_dir: str = Field(default="tools")
    skill_dir: str = Field(default="skills")
    artifacts_dir: str = Field(default="artifacts")

    # Allowed artifact extensions.
    files: list[str] = Field(default=[".json", ".pdf", ".docx", ".xlsl", ".pptx"])

    @field_validator("root_dir")
    @classmethod
    def _resolve_root(cls, v: Path) -> Path:
        """Always store root_dir as an absolute, resolved path.

        Relative paths in SERVER_WORKSPACE (e.g. './data') get resolved
        against the cwd at import time, so the rest of the framework
        can rely on root_dir being canonical.
        """
        return v.expanduser().resolve()


setting = Settings()
