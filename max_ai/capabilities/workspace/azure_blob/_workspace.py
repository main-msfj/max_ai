"""The user's workspace in Azure Blob Storage (or Azurite, its local emulator)."""

from __future__ import annotations

import os
import typing as t
from pathlib import Path
from urllib.parse import urlparse

from .._remote import RemoteObject, RemoteWorkspace
from ._model import AzureBlobWorkspaceConfig


class AzureBlobWorkspace(RemoteWorkspace):
    """Needs only the container URL and the env var that holds the key.

        AzureBlobWorkspace("https://myaccount.blob.core.windows.net/workspaces")
        AzureBlobWorkspace("http://localhost:10000/devstoreaccount1/workspaces")  # Azurite

    The key may be the account key or a SAS token (``sig=...``). The
    container is created on first use. ``pip install 'maxai[azure-blob]'``.
    """

    component_provider_override = "max_ai.capabilities.workspace.azure_blob.AzureBlobWorkspace"
    component_schema = AzureBlobWorkspaceConfig

    def __init__(
        self,
        blob_url: str,
        *,
        api_key_env: str = "AZURE_STORAGE_KEY",
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize ``AzureBlobWorkspace``.

Parameters
----------
blob_url : str
    Value supplied for ``blob_url``.
api_key_env : str
    Value supplied for ``api_key_env``.
cache_dir : str | Path | None
    Value supplied for ``cache_dir``."""
        self.blob_url = blob_url.rstrip("/")
        self.api_key_env = api_key_env
        self.account_url, self.container, self.account_name = _parse(self.blob_url)
        self._client: t.Any = None
        super().__init__(cache_dir)

    def _location(self) -> str:
        """Perform the internal ``location`` operation for ``AzureBlobWorkspace``."""
        return self.blob_url

    def _to_config(self) -> AzureBlobWorkspaceConfig:
        """Build the serializable configuration for ``AzureBlobWorkspace``."""
        return AzureBlobWorkspaceConfig(blob_url=self.blob_url, api_key_env=self.api_key_env,
                                        cache_dir=self.cache_dir)

    @classmethod
    def _from_config(cls, config: AzureBlobWorkspaceConfig) -> AzureBlobWorkspace:
        """Create an instance from its configuration for ``AzureBlobWorkspace``.

Parameters
----------
config : AzureBlobWorkspaceConfig
    Value supplied for ``config``."""
        return cls(config.blob_url, api_key_env=config.api_key_env, cache_dir=config.cache_dir)

    # -------- CLIENT -----------------------------------------------------------
    async def _container(self) -> t.Any:
        """Perform the internal ``container`` operation for ``AzureBlobWorkspace``."""
        if self._client is None:
            try:
                from azure.core.credentials import AzureNamedKeyCredential
                from azure.core.exceptions import ResourceExistsError
                from azure.storage.blob.aio import ContainerClient
            except ImportError as error:
                raise ImportError("Install Azure Blob support: pip install 'maxai[azure-blob]'") from error
            key = os.getenv(self.api_key_env)
            if not key:
                raise ValueError(f"Set {self.api_key_env} to the storage account key or a SAS token")
            credential: t.Any = key if "sig=" in key else AzureNamedKeyCredential(self.account_name, key)
            client = ContainerClient(self.account_url, self.container, credential=credential)
            try:
                await client.create_container()
            except ResourceExistsError:
                pass
            self._client = client
        return self._client

    async def disconnect(self) -> None:
        """Release resources held for ``AzureBlobWorkspace``."""
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    # -------- STORAGE -----------------------------------------------------------
    async def _list(self, prefix: str) -> list[RemoteObject]:
        """Perform the internal ``list`` operation for ``AzureBlobWorkspace``.

Parameters
----------
prefix : str
    Value supplied for ``prefix``."""
        container = await self._container()
        return [
            RemoteObject(blob.name[len(prefix):], (blob.metadata or {}).get("sha256"))
            async for blob in container.list_blobs(name_starts_with=prefix, include=["metadata"])
        ]

    async def _get(self, key: str) -> bytes:
        """Perform the internal ``get`` operation for ``AzureBlobWorkspace``.

Parameters
----------
key : str
    Value supplied for ``key``."""
        stream = await (await self._container()).download_blob(key)
        return await stream.readall()

    async def _put(self, key: str, data: bytes, sha256: str) -> None:
        """Perform the internal ``put`` operation for ``AzureBlobWorkspace``.

Parameters
----------
key : str
    Value supplied for ``key``.
data : bytes
    Value supplied for ``data``.
sha256 : str
    Value supplied for ``sha256``."""
        await (await self._container()).upload_blob(key, data, overwrite=True, metadata={"sha256": sha256})

    async def _delete(self, key: str) -> None:
        """Perform the internal ``delete`` operation for ``AzureBlobWorkspace``.

Parameters
----------
key : str
    Value supplied for ``key``."""
        from azure.core.exceptions import ResourceNotFoundError

        try:
            await (await self._container()).delete_blob(key)
        except ResourceNotFoundError:
            pass


def _parse(blob_url: str) -> tuple[str, str, str]:
    """(account URL, container, account name) from a container URL. Azure
    puts the account in the host; Azurite and other emulators in the path."""
    url = urlparse(blob_url)
    parts = [part for part in url.path.split("/") if part]
    if not url.scheme or not parts:
        raise ValueError(f"blob_url must end with the container name: {blob_url!r}")
    container = parts[-1]
    if url.hostname and url.hostname.endswith(".blob.core.windows.net"):
        account = url.hostname.split(".")[0]
        account_url = f"{url.scheme}://{url.netloc}"
    else:
        if len(parts) < 2:
            raise ValueError(f"blob_url must be <endpoint>/<account>/<container>: {blob_url!r}")
        account = parts[-2]
        account_url = f"{url.scheme}://{url.netloc}/{'/'.join(parts[:-1])}"
    return account_url, container, account
