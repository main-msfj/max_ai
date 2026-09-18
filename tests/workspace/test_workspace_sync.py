from __future__ import annotations

import hashlib

import pytest

from max_ai.workspace_copy.artifacts import Artifact, ArtifactConflict
from max_ai.workspace_copy.artifacts import LocalArtifactStore
from max_ai.workspace_copy.filesystem import UserFileSystem
from max_ai.workspace_copy.managed import ManagedFileSystem
from max_ai.workspace_copy.sync import WorkspaceSync


class MemoryArtifactStore:
    identity = "memory-test-store"

    def __init__(self):
        self.data: dict[tuple[str, str], tuple[Artifact, bytes]] = {}
        self.fail_list = False
        self.fail_put = False
        self.put_calls = 0
        self.truncated = False

    def list(self, user_id: str, limit: int = 1000):
        if self.fail_list:
            raise ConnectionError("secret=must-not-leak")
        rows = [value[0] for (user, _), value in sorted(self.data.items()) if user == user_id]
        return rows[:limit], self.truncated or len(rows) > limit

    def get(self, user_id: str, path: str):
        return self.data[(user_id, path)]

    def put(self, user_id: str, path: str, data: bytes, expected_revision: str | None = None):
        self.put_calls += 1
        if self.fail_put:
            raise ConnectionError("credential=must-not-leak")
        key = (user_id, path)
        current = self.data.get(key)
        if (current is None and expected_revision is not None) or (
            current is not None and current[0].revision != expected_revision
        ):
            raise ArtifactConflict("CAS mismatch")
        revision = f"opaque-{0 if current is None else int(current[0].revision.rsplit('-', 1)[-1]) + 1}"
        artifact = Artifact(path, revision, hashlib.sha256(data).hexdigest(), len(data))
        self.data[key] = artifact, data
        return artifact


def _item(result, path):
    return next(item for item in result["items"] if item["path"] == path)


def test_restart_restores_lost_local_file(tmp_path):
    store = MemoryArtifactStore()
    fs = ManagedFileSystem(tmp_path, artifact_store=store)
    fs.create_text_file("alice", "s1", "notes.txt", "first")
    target = tmp_path / "alice" / "s1" / "notes.txt"
    target.unlink()

    restarted = WorkspaceSync(tmp_path, store=store)
    result = restarted.reconcile("alice")

    assert target.read_bytes() == b"first"
    assert _item(result, "s1/notes.txt")["state"] == "synced"


def test_remote_and_local_updates_pull_push_and_conflict(tmp_path):
    store = MemoryArtifactStore()
    fs = ManagedFileSystem(tmp_path, artifact_store=store)
    fs.create_text_file("alice", "s1", "notes.txt", "base")
    path = tmp_path / "alice" / "s1" / "notes.txt"

    current = store.data[("alice", "s1/notes.txt")][0]
    store.put("alice", "s1/notes.txt", b"remote", expected_revision=current.revision)
    result = fs.refresh("alice")
    assert path.read_bytes() == b"remote"
    assert _item(result, "s1/notes.txt")["state"] == "synced"

    path.write_bytes(b"local")
    fs.refresh("alice")
    assert store.data[("alice", "s1/notes.txt")][1] == b"local"

    path.write_bytes(b"local edit")
    current = store.data[("alice", "s1/notes.txt")][0]
    store.put("alice", "s1/notes.txt", b"remote edit", expected_revision=current.revision)
    result = fs.refresh("alice")
    assert _item(result, "s1/notes.txt")["state"] == "conflict"
    assert path.read_bytes() == b"local edit"
    assert store.data[("alice", "s1/notes.txt")][1] == b"remote edit"


def test_external_file_is_discovered_and_binary_import_is_preserved(tmp_path):
    store = MemoryArtifactStore()
    fs = ManagedFileSystem(tmp_path, artifact_store=store)
    fs.user_root("alice")
    target = tmp_path / "alice" / "outside" / "placed.bin"
    target.parent.mkdir()
    target.write_bytes(b"\x00\x01binary")

    result = fs.refresh("alice")
    assert store.data[("alice", "outside/placed.bin")][1] == b"\x00\x01binary"
    assert _item(result, "outside/placed.bin")["state"] == "synced"

    imported = fs.import_bytes("alice", "s2", "payload.dat", b"\x00payload")
    assert imported["sync"]["state"] == "synced"
    assert store.data[("alice", "s2/payload.dat")][1] == b"\x00payload"


def test_outage_is_sanitized_and_retried(tmp_path):
    store = MemoryArtifactStore()
    fs = ManagedFileSystem(tmp_path, artifact_store=store)
    fs.create_text_file("alice", "s1", "notes.txt", "durable locally")
    store.fail_list = True

    result = fs.refresh("alice")
    assert _item(result, "s1/notes.txt")["state"] == "error"
    assert "secret" not in str(result)
    assert fs.read_snapshot("alice", "s1/notes.txt") == b"durable locally"

    store.fail_list = False
    assert _item(fs.refresh("alice"), "s1/notes.txt")["state"] == "synced"


def test_user_isolation_and_status_shape(tmp_path):
    store = MemoryArtifactStore()
    fs = ManagedFileSystem(tmp_path, artifact_store=store)
    fs.create_text_file("alice", "s1", "a.txt", "a")
    fs.create_text_file("bob", "s1", "b.txt", "b")

    assert set(store.data) == {("alice", "s1/a.txt"), ("bob", "s1/b.txt")}
    status = fs.sync_status("alice", "s1/a.txt")
    assert status["state"] == "synced"
    assert status["revision"].startswith("opaque-")
    assert fs.sync_status("alice", "s1/b.txt")["state"] == "pending"


def test_truncated_scan_does_not_infer_missing_remote_or_push(tmp_path, monkeypatch):
    store = MemoryArtifactStore()
    raw = UserFileSystem(tmp_path)
    raw.write_bytes("alice", "s1/a.txt", b"a")
    store.truncated = True
    sync = WorkspaceSync(tmp_path, store=store)

    result = sync.reconcile("alice")
    assert result["truncated"] is True
    assert _item(result, "s1/a.txt")["state"] == "pending"
    assert store.data == {}


def test_put_outage_survives_restart_and_retries_once(tmp_path):
    store = MemoryArtifactStore()
    raw = UserFileSystem(tmp_path)
    raw.write_bytes("alice", "s1/a.txt", b"pending bytes")
    store.fail_put = True

    first = WorkspaceSync(tmp_path, store=store).reconcile("alice")
    assert _item(first, "s1/a.txt")["state"] == "error"
    assert "credential" not in str(first)
    store.fail_put = False
    restarted = WorkspaceSync(tmp_path, store=store)
    retried = restarted.reconcile("alice")
    assert _item(retried, "s1/a.txt")["state"] == "synced"
    assert store.data[("alice", "s1/a.txt")][1] == b"pending bytes"
    assert store.put_calls == 2


def test_local_artifact_store_survives_fresh_instances_and_cache_loss(tmp_path):
    store_root = tmp_path / "remote-store"
    local_root = tmp_path / "workspace"
    first_store = LocalArtifactStore(store_root)
    raw = UserFileSystem(local_root)
    raw.write_bytes("alice", "s1/a.txt", b"persisted")
    assert _item(WorkspaceSync(local_root, store=first_store).reconcile("alice"), "s1/a.txt")["state"] == "synced"

    (local_root / "alice" / "s1" / "a.txt").unlink()
    fresh_store = LocalArtifactStore(store_root)
    result = WorkspaceSync(local_root, store=fresh_store).reconcile("alice")
    assert _item(result, "s1/a.txt")["state"] == "synced"
    assert (local_root / "alice" / "s1" / "a.txt").read_bytes() == b"persisted"


def test_invalid_remote_path_is_not_echoed(tmp_path):
    store = MemoryArtifactStore()
    store.data[("alice", "../../private/token")] = (
        Artifact("../../private/token", "etag-secret", "0" * 64, 1),
        b"x",
    )
    result = WorkspaceSync(tmp_path, store=store).reconcile("alice")
    assert result["items"] == [{
        "path": "<invalid-remote>",
        "state": "error",
        "revision": None,
        "sha256": None,
        "error": "invalid remote artifact metadata",
    }]
    assert "private" not in str(result)
