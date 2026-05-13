"""Runtime filesystem configuration for MaxAI.

The framework uses one server workspace as the root for runtime files
shared by local and container executors. Users can override the root
with ``SERVER_WORKSPACE``; subdirectories keep stable names so Docker
and local runtimes speak the same filesystem contract.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global Settings for Max AI Framework"""

    model_config = SettingsConfigDict(env_file=".env")
    root_dir: Path = Field(default=Path.cwd())
    tool_dir: str = Field(default="tmp/{user_id}/tools")
    skill_dir: str = Field(default="tmp/{user_id}/skills")
    artifacts_dir: str = Field(default="tmp/{user_id}/artifacts")
    sandbox: str = Field(default="MaxWorkspace", description="Base Name for Directory")
    files: list[str] = Field(default=[".json", ".pdf", ".docx", ".xlsl", ".pptx"])



setting = Settings()
