"""Serializable configuration for MinIOWorkspace."""

from __future__ import annotations

from pydantic import Field

from ....base.workspace import WorkspaceConfig


class MinIOWorkspaceConfig(WorkspaceConfig):
    endpoint_url: str = Field(description="e.g. http://localhost:9000 or https://s3.amazonaws.com")
    bucket: str = Field(min_length=3)
    access_key_env: str = Field(default="MINIO_ACCESS_KEY", description="Env var name, never the key.")
    secret_key_env: str = Field(default="MINIO_SECRET_KEY", description="Env var name, never the key.")
    region: str | None = None
    cache_dir: str | None = None
