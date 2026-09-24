"""Serializable configuration for AzureBlobWorkspace."""

from __future__ import annotations

from pydantic import Field

from ....base.workspace import WorkspaceConfig


class AzureBlobWorkspaceConfig(WorkspaceConfig):
    """Configuration options for ``AzureBlobWorkspace``."""
    blob_url: str = Field(description="Container URL, e.g. https://<account>.blob.core.windows.net/<container>.")
    api_key_env: str = Field(default="AZURE_STORAGE_KEY", description="Env var with the account key or a SAS token.")
    cache_dir: str | None = None
