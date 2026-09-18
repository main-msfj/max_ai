import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from max_ai.workspace_copy.artifacts import Artifact, ArtifactConflict, LocalArtifactStore
from max_ai.workspace_copy.filesystem import UserFileSystem


def test_artifact_revisions_survive_store_restart_and_are_integrity_checked(tmp_path):
    store = LocalArtifactStore(tmp_path / "trusted")
    first = store.put("user1", "session1/image.bin", b"\x00first\xff")
    second = store.put(
        "user1", "session1/image.bin", b"\x00second\xff", first.revision
    )

    restarted = LocalArtifactStore(tmp_path / "trusted")
    assert restarted.identity == store.identity
    assert restarted.root == (tmp_path / "trusted").resolve()
    assert restarted.get("user1", "session1/image.bin") == (second, b"\x00second\xff")
    assert restarted.get_revision("user1", "session1/image.bin", first.revision) == (
        first,
        b"\x00first\xff",
    )
    assert first.sha256 == hashlib.sha256(b"\x00first\xff").hexdigest()
    assert first.size == len(b"\x00first\xff")


def test_artifact_create_only_and_revision_conflicts_are_stable(tmp_path):
    first_store = LocalArtifactStore(tmp_path / "trusted")
    second_store = LocalArtifactStore(tmp_path / "trusted")
    original = first_store.put("user1", "session1/file.dat", b"original")

    with pytest.raises(ArtifactConflict):
        second_store.put("user1", "session1/file.dat", b"overwrite")
    with pytest.raises(ArtifactConflict):
        second_store.put(
            "user1", "session1/file.dat", b"stale", expected_revision="stale"
        )
    assert first_store.get("user1", "session1/file.dat") == (original, b"original")


def test_concurrent_revision_cas_allows_exactly_one_update(tmp_path):
    store_a = LocalArtifactStore(tmp_path / "trusted")
    store_b = LocalArtifactStore(tmp_path / "trusted")
    initial = store_a.put("user1", "session1/file.dat", b"initial")

    def update(store, content):
        try:
            return store.put("user1", "session1/file.dat", content, initial.revision)
        except ArtifactConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda args: update(*args), [(store_a, b"a"), (store_b, b"b")])
        )
    assert sum(result is not None for result in results) == 1
    assert isinstance(next(result for result in results if result is not None), Artifact)


def test_artifact_paths_are_scoped_and_users_are_isolated(tmp_path):
    store = LocalArtifactStore(tmp_path / "trusted")
    user_one = store.put("user1", "session1/same.bin", b"one")
    user_two = store.put("user2", "session1/same.bin", b"two")
    assert user_one.path == user_two.path
    assert store.get("user1", "session1/same.bin")[1] == b"one"
    assert store.get("user2", "session1/same.bin")[1] == b"two"

    for path in (
        "../outside.bin",
        "/absolute.bin",
        "session1/../outside.bin",
        "session1/.hidden",
        "session1/.maxai-secret",
        "artifacts/file.bin",
    ):
        with pytest.raises(ValueError):
            store.put("user1", path, b"bad")


def test_artifact_limit_and_listing_truncation(tmp_path):
    store = LocalArtifactStore(tmp_path / "trusted")
    store.put("user1", "session1/a.bin", b"a")
    store.put("user1", "session1/b.bin", b"b")
    artifacts, truncated = store.list("user1", limit=1)
    assert [artifact.path for artifact in artifacts] == ["session1/a.bin"]
    assert truncated
    with pytest.raises(ValueError):
        store.put("user1", "session1/large.bin", b"x" * (8 * 1024 * 1024 + 1))


def test_filesystem_binary_snapshot_and_create_only_write(tmp_path):
    filesystem = UserFileSystem(tmp_path / "workspace")
    content = b"\x00\xff\x80binary\n"
    result = filesystem.write_bytes("user1", "session1/photo.bin", content)
    assert result == {
        "path": "session1/photo.bin",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    assert filesystem.read_snapshot("user1", "session1/photo.bin") == content
    with pytest.raises(FileExistsError):
        filesystem.write_bytes("user1", "session1/photo.bin", b"replacement")

    updated = filesystem.write_bytes(
        "user1", "session1/photo.bin", b"updated", result["sha256"]
    )
    assert updated["sha256"] == hashlib.sha256(b"updated").hexdigest()
    assert filesystem.read_snapshot("user1", "session1/photo.bin") == b"updated"
    with pytest.raises(ValueError, match="does not match"):
        filesystem.write_bytes("user1", "session1/photo.bin", b"stale", result["sha256"])


def test_filesystem_binary_methods_reject_traversal_dot_paths_and_symlinks(tmp_path):
    root = tmp_path / "workspace"
    filesystem = UserFileSystem(root)
    filesystem.conversation_root("user1", "session1")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"private")
    (root / "user1" / "session1" / "linked.bin").symlink_to(outside)

    for path in (
        "../outside.bin",
        "/absolute.bin",
        "session1/../outside.bin",
        "session1/.hidden",
        "session1/.maxai-internal",
        "session1/linked.bin",
        "tools/file.bin",
        "single-segment",
    ):
        with pytest.raises((ValueError, FileExistsError)):
            filesystem.write_bytes("user1", path, b"bad")
    with pytest.raises(ValueError):
        filesystem.read_snapshot("user1", "session1/linked.bin")
    assert outside.read_bytes() == b"private"
