"""The user's workspace in MinIO or any S3-compatible storage."""

from __future__ import annotations

import asyncio
import io
import os
import typing as t
from pathlib import Path
from urllib.parse import urlparse

from .._remote import RemoteObject, RemoteWorkspace
from ._model import MinIOWorkspaceConfig


class MinIOWorkspace(RemoteWorkspace):
    """Needs the endpoint, the bucket and the env vars holding the keys.

        MinIOWorkspace("http://localhost:9000", bucket="workspaces")

    Works with AWS S3 too (``https://s3.amazonaws.com`` + ``region``). The
    bucket is created on first use. ``pip install 'maxai[minio]'``.
    """

    component_provider_override = "max_ai.capabilities.workspace.minio.MinIOWorkspace"
    component_schema = MinIOWorkspaceConfig

    def __init__(
        self,
        endpoint_url: str,
        *,
        bucket: str,
        access_key_env: str = "MINIO_ACCESS_KEY",
        secret_key_env: str = "MINIO_SECRET_KEY",
        region: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize ``MinIOWorkspace``.

Parameters
----------
endpoint_url : str
    Value supplied for ``endpoint_url``.
bucket : str
    Value supplied for ``bucket``.
access_key_env : str
    Value supplied for ``access_key_env``.
secret_key_env : str
    Value supplied for ``secret_key_env``.
region : str | None
    Value supplied for ``region``.
cache_dir : str | Path | None
    Value supplied for ``cache_dir``."""
        self.endpoint_url = endpoint_url.rstrip("/")
        self.bucket = bucket
        self.access_key_env = access_key_env
        self.secret_key_env = secret_key_env
        self.region = region
        self._client: t.Any = None
        super().__init__(cache_dir)

    def _location(self) -> str:
        """Perform the internal ``location`` operation for ``MinIOWorkspace``."""
        return f"{self.endpoint_url}/{self.bucket}"

    def _to_config(self) -> MinIOWorkspaceConfig:
        """Build the serializable configuration for ``MinIOWorkspace``."""
        return MinIOWorkspaceConfig(
            endpoint_url=self.endpoint_url, bucket=self.bucket, access_key_env=self.access_key_env,
            secret_key_env=self.secret_key_env, region=self.region, cache_dir=self.cache_dir,
        )

    @classmethod
    def _from_config(cls, config: MinIOWorkspaceConfig) -> MinIOWorkspace:
        """Create an instance from its configuration for ``MinIOWorkspace``.

Parameters
----------
config : MinIOWorkspaceConfig
    Value supplied for ``config``."""
        return cls(config.endpoint_url, bucket=config.bucket, access_key_env=config.access_key_env,
                   secret_key_env=config.secret_key_env, region=config.region, cache_dir=config.cache_dir)

    # -------- CLIENT (the MinIO SDK is sync: every call runs in a thread) --------
    async def _call(self, method: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Perform the internal ``call`` operation for ``MinIOWorkspace``.

Parameters
----------
method : str
    Value supplied for ``method``.
args : t.Any
    Value supplied for ``args``.
kwargs : t.Any
    Value supplied for ``kwargs``."""
        if self._client is None:
            self._client = await asyncio.to_thread(self._connect)
        return await asyncio.to_thread(getattr(self._client, method), *args, **kwargs)

    def _connect(self) -> t.Any:
        """Perform the internal ``connect`` operation for ``MinIOWorkspace``."""
        try:
            from minio import Minio
        except ImportError as error:
            raise ImportError("Install MinIO support: pip install 'maxai[minio]'") from error
        access, secret = os.getenv(self.access_key_env), os.getenv(self.secret_key_env)
        if not access or not secret:
            raise ValueError(f"Set {self.access_key_env} and {self.secret_key_env}")
        url = urlparse(self.endpoint_url)
        client = Minio(url.netloc, access_key=access, secret_key=secret,
                       secure=url.scheme == "https", region=self.region)
        if not client.bucket_exists(self.bucket):
            client.make_bucket(self.bucket)
        return client

    # -------- STORAGE -----------------------------------------------------------
    async def _list(self, prefix: str) -> list[RemoteObject]:
        """Perform the internal ``list`` operation for ``MinIOWorkspace``.

Parameters
----------
prefix : str
    Value supplied for ``prefix``."""
        def listing() -> list[RemoteObject]:
            """Perform the ``listing`` operation for ``MinIOWorkspace``."""
            objects = self._client.list_objects(self.bucket, prefix=prefix, recursive=True,
                                                include_user_meta=True)
            found = []
            for obj in objects:
                meta = {k.lower(): v for k, v in (obj.metadata or {}).items()}
                sha = meta.get("x-amz-meta-sha256") or meta.get("sha256")
                found.append(RemoteObject(obj.object_name[len(prefix):], sha))
            return found

        if self._client is None:
            self._client = await asyncio.to_thread(self._connect)
        return await asyncio.to_thread(listing)

    async def _get(self, key: str) -> bytes:
        """Perform the internal ``get`` operation for ``MinIOWorkspace``.

Parameters
----------
key : str
    Value supplied for ``key``."""
        response = await self._call("get_object", self.bucket, key)
        try:
            return await asyncio.to_thread(response.read)
        finally:
            response.close()
            response.release_conn()

    async def _put(self, key: str, data: bytes, sha256: str) -> None:
        """Perform the internal ``put`` operation for ``MinIOWorkspace``.

Parameters
----------
key : str
    Value supplied for ``key``.
data : bytes
    Value supplied for ``data``.
sha256 : str
    Value supplied for ``sha256``."""
        await self._call("put_object", self.bucket, key, io.BytesIO(data), len(data),
                         metadata={"sha256": sha256})

    async def _delete(self, key: str) -> None:
        """Perform the internal ``delete`` operation for ``MinIOWorkspace``.

Parameters
----------
key : str
    Value supplied for ``key``."""
        await self._call("remove_object", self.bucket, key)
