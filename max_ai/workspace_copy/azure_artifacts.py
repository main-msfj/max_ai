"""Revisioned artifact storage in an existing Azure Blob container.

Install the optional SDK dependencies with ``uv sync --extra artifacts-azure``.
Azure container versioning is an account-level setting and is not changed here;
this store exposes the current blob ETag for CAS, but does not promise that old
ETag revisions remain retrievable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from .artifacts import Artifact, ArtifactConflict, ArtifactStore, _MAX_ARTIFACT_BYTES
from .filesystem import UserFileSystem


_DEFAULT_LIST_LIMIT = 1000
_MAX_LIST_SCAN = 10_000
_MAX_OPERATION_TIMEOUT = 20
_MAX_ATTEMPTS = 3
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MISSING_SDK_MESSAGE = (
    "Azure artifact storage requires optional dependencies; install them with "
    "`uv sync --extra artifacts-azure`."
)


class _ArtifactDataError(ValueError):
    """A locally detected artifact size or integrity violation."""


class AzureBlobArtifactStore(ArtifactStore):
    """Store one current artifact per user/path in an existing blob container."""

    def __init__(
        self,
        account_url: str,
        container: str,
        prefix: str = "maxai",
        credential: Any = None,
        *,
        container_client: Any = None,
    ) -> None:
        self._account_url = self._validated_account_url(account_url)
        self._container = self._validated_container(container)
        self._prefix = self._validated_prefix(prefix)
        self._credential = credential
        self._explicit_credential = credential is not None
        self._container_client = container_client
        self._identity = self._make_identity()

    @property
    def identity(self) -> str:
        return self._identity

    def __repr__(self) -> str:
        return f"AzureBlobArtifactStore(identity={self.identity!r})"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AzureBlobArtifactStore:
        """Build a store from non-secret configuration; credentials use Azure defaults."""
        if not isinstance(config, dict):
            raise ValueError("config must be a mapping")
        allowed = {"account_url", "container", "prefix"}
        if set(config) - allowed or not {"account_url", "container"} <= set(config):
            raise ValueError("config must contain account_url and container, with optional prefix")
        return cls(**config)

    def to_config(self) -> dict[str, str]:
        """Return safe, non-secret configuration suitable for persistence."""
        if self._explicit_credential:
            raise ValueError(
                "cannot serialize a store with an explicit credential; configure "
                "the credential separately when restoring it"
            )
        return {
            "account_url": self._account_url,
            "container": self._container,
            "prefix": self._prefix,
        }

    def list(
        self, user_id: str, limit: int = _DEFAULT_LIST_LIMIT
    ) -> tuple[list[Artifact], bool]:
        user = UserFileSystem._safe_id(user_id, "user_id")
        effective_limit = self._bounded_limit(limit)
        name_prefix = self._user_prefix(user)
        results: list[Artifact] = []
        scanned = 0
        truncated = False
        try:
            blobs = self._client().list_blobs(
                name_starts_with=name_prefix, include=["metadata"]
            )
            for blob in blobs:
                scanned += 1
                if scanned > _MAX_LIST_SCAN:
                    truncated = True
                    break
                blob_name = getattr(blob, "name", None)
                path = self._path_from_blob_name(blob_name, name_prefix)
                if path is None:
                    continue
                etag = getattr(blob, "etag", None)
                size = getattr(blob, "size", None)
                metadata = getattr(blob, "metadata", None) or {}
                digest = metadata.get("sha256") if isinstance(metadata, dict) else None
                if (
                    isinstance(etag, str)
                    and etag
                    and isinstance(size, int)
                    and not isinstance(size, bool)
                    and 0 <= size <= _MAX_ARTIFACT_BYTES
                    and isinstance(digest, str)
                    and _SHA256_RE.fullmatch(digest)
                ):
                    results.append(Artifact(path, etag, digest, size))
                else:
                    # External uploads may not carry our metadata. Read the exact
                    # listed ETag so a concurrent replacement cannot mislabel it.
                    artifact, _ = self._read_blob(
                        self._client().get_blob_client(blob_name),
                        path,
                        expected_etag=etag if isinstance(etag, str) else None,
                        expected_size=size if isinstance(size, int) else None,
                    )
                    results.append(artifact)
                if len(results) > effective_limit:
                    truncated = True
                    break
        except (ArtifactConflict, FileNotFoundError, _ArtifactDataError):
            raise
        except ImportError:
            raise
        except Exception as exc:
            self._raise_safe_azure_error(exc)
        results.sort(key=lambda artifact: artifact.path)
        return results[:effective_limit], truncated

    def get(self, user_id: str, path: str) -> tuple[Artifact, bytes]:
        user = UserFileSystem._safe_id(user_id, "user_id")
        clean_path = self._validated_path(path)
        blob_name = self._blob_name(user, clean_path)
        try:
            return self._read_blob(self._client().get_blob_client(blob_name), clean_path)
        except (ArtifactConflict, FileNotFoundError, _ArtifactDataError):
            raise
        except ImportError:
            raise
        except Exception as exc:
            self._raise_safe_azure_error(exc)

    def put(
        self,
        user_id: str,
        path: str,
        data: bytes,
        expected_revision: str | None = None,
    ) -> Artifact:
        user = UserFileSystem._safe_id(user_id, "user_id")
        clean_path = self._validated_path(path)
        if not isinstance(data, bytes):
            raise ValueError("data must be bytes")
        if len(data) > _MAX_ARTIFACT_BYTES:
            raise ValueError(f"data exceeds {_MAX_ARTIFACT_BYTES} bytes")
        if expected_revision is not None and (
            not isinstance(expected_revision, str) or not expected_revision
        ):
            raise ValueError("expected_revision must be a non-empty string")

        digest = hashlib.sha256(data).hexdigest()
        kwargs: dict[str, Any] = {
            "overwrite": expected_revision is not None,
            "metadata": {"sha256": digest},
            "timeout": _MAX_OPERATION_TIMEOUT,
        }
        try:
            blob = self._client().get_blob_client(self._blob_name(user, clean_path))
            if expected_revision is not None:
                kwargs["etag"] = expected_revision
                kwargs["match_condition"] = self._if_not_modified()
            response = blob.upload_blob(data, **kwargs)
            revision = response.get("etag") if isinstance(response, dict) else None
            if not isinstance(revision, str) or not revision:
                raise RuntimeError("Azure Blob operation failed")
            return Artifact(clean_path, revision, digest, len(data))
        except Exception as exc:
            if isinstance(exc, ImportError):
                raise
            if self._is_conflict(exc):
                raise ArtifactConflict("artifact revision does not match") from None
            self._raise_safe_azure_error(exc)

    def _client(self) -> Any:
        if self._container_client is not None:
            return self._container_client
        try:
            from azure.storage.blob import BlobServiceClient, ExponentialRetry

            credential = self._credential
            if credential is None:
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()
                self._credential = credential
            service = BlobServiceClient(
                account_url=self._account_url,
                credential=credential,
                connection_timeout=5,
                read_timeout=15,
                retry_total=_MAX_ATTEMPTS - 1,
                retry_connect=_MAX_ATTEMPTS - 1,
                retry_read=_MAX_ATTEMPTS - 1,
                retry_status=_MAX_ATTEMPTS - 1,
                retry_policy=ExponentialRetry(
                    initial_backoff=1,
                    increment_base=1,
                    random_jitter_range=0,
                    retry_total=_MAX_ATTEMPTS - 1,
                    retry_connect=_MAX_ATTEMPTS - 1,
                    retry_read=_MAX_ATTEMPTS - 1,
                    retry_status=_MAX_ATTEMPTS - 1,
                ),
            )
            # Deliberately use the existing container; never create accounts or
            # containers as a side effect of constructing or using this adapter.
            self._container_client = service.get_container_client(self._container)
            return self._container_client
        except ImportError:
            raise ImportError(_MISSING_SDK_MESSAGE) from None

    def _read_blob(
        self,
        blob: Any,
        path: str,
        *,
        expected_etag: str | None = None,
        expected_size: int | None = None,
    ) -> tuple[Artifact, bytes]:
        try:
            properties = blob.get_blob_properties(timeout=_MAX_OPERATION_TIMEOUT)
            etag = getattr(properties, "etag", None)
            size = getattr(properties, "size", None)
            metadata = getattr(properties, "metadata", None) or {}
            if not isinstance(etag, str) or not etag:
                raise _ArtifactDataError("stored artifact failed integrity verification")
            if expected_etag is not None and etag != expected_etag:
                raise ArtifactConflict("artifact revision changed during read")
            if expected_size is not None and size != expected_size:
                raise ArtifactConflict("artifact revision changed during read")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise _ArtifactDataError("stored artifact failed integrity verification")
            if size > _MAX_ARTIFACT_BYTES:
                raise _ArtifactDataError(f"stored artifact exceeds {_MAX_ARTIFACT_BYTES} bytes")

            downloader = blob.download_blob(
                etag=etag,
                match_condition=self._if_not_modified(),
                timeout=_MAX_OPERATION_TIMEOUT,
            )
            data = self._read_bounded_chunks(downloader.chunks())
            digest = hashlib.sha256(data).hexdigest()
            stored_digest = metadata.get("sha256") if isinstance(metadata, dict) else None
            if stored_digest is not None and (
                not isinstance(stored_digest, str)
                or not _SHA256_RE.fullmatch(stored_digest)
                or stored_digest != digest
            ):
                raise _ArtifactDataError("stored artifact failed integrity verification")
            if len(data) != size:
                raise _ArtifactDataError("stored artifact failed integrity verification")
            return Artifact(path, etag, digest, size), data
        except (ArtifactConflict, FileNotFoundError, _ArtifactDataError):
            raise
        except Exception as exc:
            if isinstance(exc, ImportError):
                raise
            if self._is_missing(exc):
                raise FileNotFoundError(path) from None
            if self._is_conflict(exc):
                raise ArtifactConflict("artifact revision changed during read") from None
            self._raise_safe_azure_error(exc)

    @staticmethod
    def _read_bounded_chunks(chunks: Iterable[bytes]) -> bytes:
        data = bytearray()
        for chunk in chunks:
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise _ArtifactDataError("stored artifact failed integrity verification")
            if len(data) + len(chunk) > _MAX_ARTIFACT_BYTES:
                raise _ArtifactDataError(f"stored artifact exceeds {_MAX_ARTIFACT_BYTES} bytes")
            data.extend(chunk)
        return bytes(data)

    def _blob_name(self, user: str, path: str) -> str:
        user_prefix = self._user_prefix(user)
        return f"{user_prefix}{path}"

    def _user_prefix(self, user: str) -> str:
        root = f"{self._prefix}/" if self._prefix else ""
        return f"{root}{user}/"

    def _path_from_blob_name(self, blob_name: Any, user_prefix: str) -> str | None:
        if not isinstance(blob_name, str) or not blob_name.startswith(user_prefix):
            return None
        path = blob_name[len(user_prefix) :]
        try:
            return self._validated_path(path)
        except (TypeError, ValueError, UnicodeError):
            return None

    @staticmethod
    def _validated_path(path: str) -> str:
        return "/".join(UserFileSystem._conversation_file_parts(path))

    @staticmethod
    def _bounded_limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        return min(limit, _DEFAULT_LIST_LIMIT)

    @staticmethod
    def _validated_account_url(account_url: str) -> str:
        if not isinstance(account_url, str):
            raise ValueError("account_url must be an HTTP(S) URL without credentials or query")
        try:
            parsed = urlsplit(account_url)
            _ = parsed.port
        except ValueError:
            raise ValueError("account_url must be an HTTP(S) URL without credentials or query") from None
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or "@" in parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("account_url must be an HTTP(S) URL without credentials or query")
        host = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme.lower()}://{host}{path}"

    @staticmethod
    def _validated_container(container: str) -> str:
        if (
            not isinstance(container, str)
            or not 3 <= len(container) <= 63
            or "--" in container
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])?", container)
        ):
            raise ValueError("container must be a valid Azure container name")
        return container

    @staticmethod
    def _validated_prefix(prefix: str) -> str:
        if not isinstance(prefix, str):
            raise ValueError("prefix must be a relative blob prefix")
        if prefix == "":
            return ""
        if prefix.startswith("/") or prefix.endswith("/") or "\\" in prefix:
            raise ValueError("prefix must be a safe relative blob prefix")
        parts = prefix.split("/")
        if any(
            part in {"", ".", ".."}
            or len(part.encode("utf-8", errors="strict")) > 255
            or not re.fullmatch(r"[A-Za-z0-9._-]+", part)
            for part in parts
        ):
            raise ValueError("prefix must be a safe relative blob prefix")
        return prefix

    def _make_identity(self) -> str:
        material = json.dumps(
            [self._account_url, self._container, self._prefix],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"azure-blob:{hashlib.sha256(material).hexdigest()}"

    @staticmethod
    def _if_not_modified() -> Any:
        try:
            from azure.core import MatchConditions

            return MatchConditions.IfNotModified
        except ImportError:
            raise ImportError(_MISSING_SDK_MESSAGE) from None

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        status = getattr(exc, "status_code", None)
        return status if isinstance(status, int) else None

    @classmethod
    def _is_conflict(cls, exc: Exception) -> bool:
        return cls._status_code(exc) in {409, 412} or getattr(exc, "error_code", None) in {
            "BlobAlreadyExists",
            "ConditionNotMet",
            "LeaseIdMissing",
        }

    @classmethod
    def _is_missing(cls, exc: Exception) -> bool:
        return cls._status_code(exc) == 404 or getattr(exc, "error_code", None) in {
            "BlobNotFound",
            "ContainerNotFound",
        }

    @classmethod
    def _raise_safe_azure_error(cls, exc: Exception) -> None:
        if cls._is_missing(exc):
            raise FileNotFoundError("artifact not found") from None
        if cls._is_conflict(exc):
            raise ArtifactConflict("artifact revision does not match") from None
        # SDK messages may include endpoint details, query strings, or credentials.
        raise RuntimeError("Azure Blob operation failed") from None
