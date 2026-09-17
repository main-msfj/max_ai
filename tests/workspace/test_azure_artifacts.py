import hashlib
from types import SimpleNamespace

import pytest
from azure.core import MatchConditions

from max_ai.workspace.artifacts import ArtifactConflict
from max_ai.workspace.azure_artifacts import AzureBlobArtifactStore


class AzureFailure(Exception):
    def __init__(self, status_code, message="sensitive endpoint detail"):
        super().__init__(message)
        self.status_code = status_code


class FakeDownload:
    def __init__(self, data):
        self.data = data

    def chunks(self):
        for offset in range(0, len(self.data), 7):
            yield self.data[offset : offset + 7]


class FakeBlobClient:
    def __init__(self, container, name):
        self.container = container
        self.name = name

    def upload_blob(self, data, **kwargs):
        assert kwargs["timeout"] == 20
        assert kwargs["metadata"] == {"sha256": hashlib.sha256(data).hexdigest()}
        current = self.container.blobs.get(self.name)
        if kwargs["overwrite"] is False:
            assert "etag" not in kwargs
            assert "match_condition" not in kwargs
            if current is not None:
                raise AzureFailure(409)
        else:
            assert kwargs["etag"]
            assert kwargs["match_condition"] is MatchConditions.IfNotModified
            if current is None or current["etag"] != kwargs["etag"]:
                raise AzureFailure(412)
        version = self.container.next_version
        self.container.next_version += 1
        etag = f'"etag-{version}"'
        self.container.blobs[self.name] = {
            "data": data,
            "etag": etag,
            "metadata": kwargs["metadata"],
        }
        return {"etag": etag}

    def get_blob_properties(self, **kwargs):
        assert kwargs == {"timeout": 20}
        record = self.container.blobs.get(self.name)
        if record is None:
            raise AzureFailure(404)
        return SimpleNamespace(
            etag=record["etag"],
            size=len(record["data"]),
            metadata=record["metadata"],
        )

    def download_blob(self, **kwargs):
        assert kwargs["timeout"] == 20
        assert kwargs["match_condition"] is MatchConditions.IfNotModified
        record = self.container.blobs.get(self.name)
        if record is None:
            raise AzureFailure(404)
        if record["etag"] != kwargs["etag"]:
            raise AzureFailure(412)
        return FakeDownload(record["data"])


class FakeContainerClient:
    def __init__(self):
        self.blobs = {}
        self.next_version = 1
        self.list_prefixes = []

    def get_blob_client(self, name):
        return FakeBlobClient(self, name)

    def list_blobs(self, *, name_starts_with, include):
        assert include == ["metadata"]
        self.list_prefixes.append(name_starts_with)
        return [
            SimpleNamespace(name=name, etag=record["etag"], size=len(record["data"]), metadata=record["metadata"])
            for name, record in sorted(self.blobs.items())
            if name.startswith(name_starts_with)
        ]


def make_store(container=None, **kwargs):
    return AzureBlobArtifactStore(
        "https://account.blob.core.windows.net",
        "artifact-data",
        container_client=container or FakeContainerClient(),
        **kwargs,
    )


def test_put_is_create_only_and_updates_use_etag_compare_and_swap():
    backend = FakeContainerClient()
    store = make_store(backend, prefix="root")
    created = store.put("user1", "session1/report.bin", b"first")

    assert created.path == "session1/report.bin"
    assert created.revision == '"etag-1"'
    assert created.sha256 == hashlib.sha256(b"first").hexdigest()
    assert created.size == 5
    assert list(backend.blobs) == ["root/user1/session1/report.bin"]

    with pytest.raises(ArtifactConflict):
        store.put("user1", "session1/report.bin", b"create again")
    updated = store.put(
        "user1", "session1/report.bin", b"second", created.revision
    )
    assert updated.revision == '"etag-2"'
    with pytest.raises(ArtifactConflict):
        store.put("user1", "session1/report.bin", b"stale", created.revision)
    assert store.get("user1", "session1/report.bin") == (updated, b"second")


def test_list_is_user_scoped_bounded_and_computes_missing_external_hashes():
    backend = FakeContainerClient()
    store = make_store(backend, prefix="root")
    store.put("user1", "session1/a.bin", b"alpha")
    store.put("user1", "session1/b.bin", b"beta")
    store.put("user2", "session1/private.bin", b"private")
    external_name = "root/user1/session2/external.bin"
    backend.blobs[external_name] = {
        "data": b"external upload",
        "etag": '"external-etag"',
        "metadata": {},
    }

    items, truncated = store.list("user1", limit=2)
    assert [item.path for item in items] == ["session1/a.bin", "session1/b.bin"]
    assert truncated is True

    items, truncated = store.list("user1", limit=10)
    assert [item.path for item in items] == [
        "session1/a.bin",
        "session1/b.bin",
        "session2/external.bin",
    ]
    assert items[2].sha256 == hashlib.sha256(b"external upload").hexdigest()
    assert truncated is False
    assert all("user2" not in prefix for prefix in backend.list_prefixes)


def test_get_verifies_stored_digest_and_missing_blob_is_file_not_found():
    backend = FakeContainerClient()
    store = make_store(backend)
    artifact = store.put("user1", "session1/file.bin", b"content")
    assert store.get("user1", "session1/file.bin") == (artifact, b"content")

    record = backend.blobs["maxai/user1/session1/file.bin"]
    record["metadata"] = {"sha256": "0" * 64}
    with pytest.raises(ValueError, match="integrity"):
        store.get("user1", "session1/file.bin")
    with pytest.raises(FileNotFoundError):
        store.get("user1", "session1/missing.bin")


def test_invalid_paths_users_limits_and_oversized_data_are_rejected():
    store = make_store()
    for path in (
        "../outside.bin",
        "/absolute.bin",
        "session1/../outside.bin",
        "session1/.hidden",
        "single-segment",
    ):
        with pytest.raises(ValueError):
            store.put("user1", path, b"bad")
    with pytest.raises(ValueError):
        store.put("../user", "session1/file.bin", b"bad")
    with pytest.raises(ValueError):
        store.list("user1", 0)
    with pytest.raises(ValueError):
        store.put("user1", "session1/large.bin", b"x" * (8 * 1024 * 1024 + 1))


def test_errors_do_not_disclose_endpoint_details_and_config_is_safe():
    backend = FakeContainerClient()
    store = make_store(backend, credential="secret-credential", prefix="safe/root")
    assert "secret-credential" not in repr(store)
    assert "account.blob.core.windows.net" not in repr(store)
    assert store.identity == make_store(prefix="safe/root").identity
    assert store.identity != make_store(prefix="different").identity

    with pytest.raises(ValueError, match="explicit credential"):
        store.to_config()
    default_store = make_store(prefix="safe/root")
    config = default_store.to_config()
    assert config == {
        "account_url": "https://account.blob.core.windows.net",
        "container": "artifact-data",
        "prefix": "safe/root",
    }
    restored = AzureBlobArtifactStore.from_config(config)
    assert restored.identity == default_store.identity
    with pytest.raises(ValueError):
        AzureBlobArtifactStore(
            "https://account.blob.core.windows.net/?sig=secret", "artifact-data"
        )

    backend.get_blob_client = lambda name: PermissionBlobClient()
    with pytest.raises(RuntimeError) as error:
        store.get("user1", "session1/private.bin")
    assert "sensitive endpoint detail" not in str(error.value)
    assert "secret" not in str(error.value)

    backend.get_blob_client = lambda name: InvalidValueBlobClient()
    with pytest.raises(RuntimeError) as error:
        store.get("user1", "session1/private.bin")
    assert "sensitive endpoint detail" not in str(error.value)

    def client_value_error():
        raise ValueError("sensitive endpoint detail with secret")

    store._client = client_value_error
    with pytest.raises(RuntimeError) as error:
        store.put("user1", "session1/private.bin", b"private")
    assert "sensitive endpoint detail" not in str(error.value)
    assert "secret" not in str(error.value)


def test_identity_preserves_endpoint_path_case_and_container_names_are_validated():
    lower = AzureBlobArtifactStore(
        "https://localhost:10000/account/Blob", "artifact-data", prefix="root"
    )
    upper = AzureBlobArtifactStore(
        "https://LOCALHOST:10000/account/blob", "artifact-data", prefix="root"
    )
    assert lower.identity != upper.identity
    assert AzureBlobArtifactStore(
        "HTTPS://LOCALHOST:10000/account/Blob/", "artifact-data", prefix="root"
    ).identity == lower.identity
    for invalid in ("a", "ab", "ab--cd", "-abc", "abc-"):
        with pytest.raises(ValueError, match="container"):
            AzureBlobArtifactStore("https://localhost", invalid)


def test_default_client_uses_bounded_exponential_retry_without_network(monkeypatch):
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import ExponentialRetry

    captured = {}
    sentinel = object()

    class FakeService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_container_client(self, name):
            captured["container"] = name
            return sentinel

    monkeypatch.setattr(
        "azure.storage.blob.BlobServiceClient", FakeService
    )
    store = AzureBlobArtifactStore(
        "https://account.blob.core.windows.net", "artifact-data"
    )
    assert store._client() is sentinel
    assert isinstance(captured["retry_policy"], ExponentialRetry)
    assert captured["retry_policy"].total_retries == 2
    assert captured["retry_policy"].connect_retries == 2
    assert captured["retry_policy"].read_retries == 2
    assert captured["retry_policy"].status_retries == 2
    assert captured["connection_timeout"] == 5
    assert captured["read_timeout"] == 15
    assert captured["container"] == "artifact-data"
    assert isinstance(store._credential, DefaultAzureCredential)


class PermissionBlobClient:
    def get_blob_properties(self, **kwargs):
        raise AzureFailure(403, "sensitive endpoint detail with secret")

    def upload_blob(self, data, **kwargs):
        raise AzureFailure(403, "sensitive endpoint detail with secret")


class InvalidValueBlobClient:
    def get_blob_properties(self, **kwargs):
        raise ValueError("sensitive endpoint detail with secret")


def test_sdk_is_loaded_lazily_and_missing_optional_dependency_is_actionable(monkeypatch):
    store = AzureBlobArtifactStore(
        "https://account.blob.core.windows.net", "artifact-data"
    )
    assert store.identity.startswith("azure-blob:")
    original_import = __import__

    def blocked_azure_import(name, *args, **kwargs):
        if name == "azure.storage.blob" or name.startswith("azure.storage"):
            raise ImportError("SDK intentionally hidden for this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_azure_import)
    with pytest.raises(ImportError, match=r"uv sync --extra artifacts-azure"):
        store.get("user1", "session1/file.bin")
