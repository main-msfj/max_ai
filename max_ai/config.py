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
    """Global Settings"""

    model_config = SettingsConfigDict(env_file=".env")

    server_dir: str = Field(default="serverWorkspace", validation_alias="SERVER_DIR")
    sandbox_name: str = Field(default="/sandbox")
    dockerfile_name: str = Field(default="Dockerfile.worker")

    def get_or_create_server_tmp_dir(self) -> Path:
        path = Path(self.server_dir) / "tmp"
        path = path.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_or_create_skill_cache_dir(self) -> Path:
        path = Path(self.server_dir) / "var" / "skills-cache"
        path = path.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_or_create_docker_worker_dir(self) -> Path:
        path = Path(self.server_dir) / "var" / "docker-cache"
        path = path.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def check_docker_and_uv(self) -> None:
        docker_dir = self.get_or_create_docker_worker_dir()

        for filename in ["pyproject.toml", self.dockerfile_name]:
            file_path = docker_dir / filename
            if not file_path.is_file():
                raise ValueError(f"{file_path} must exist")

    def create_workspace(self) -> Path:
        """Create and Validate Workspace"""
        self.get_or_create_server_tmp_dir()
        self.get_or_create_skill_cache_dir()
        self.check_docker_and_uv()
        return Path(self.server_dir).expanduser().resolve()
    

setting = Settings()
