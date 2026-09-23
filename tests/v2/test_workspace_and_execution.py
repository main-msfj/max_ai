
import pytest
from max_ai.base.execution_workspace import ExecutionWorkspace, WorkspaceConflictError


@pytest.fixture
def execution_copy(workspace, request):
    copies = []
    def make(user="alice", conversation="chat-1"):
        copy = ExecutionWorkspace(workspace, user, conversation)
        copies.append(copy)
        return copy
    def cleanup():
        for copy in copies:
            copy.discard()
    request.addfinalizer(cleanup)
    return make


def test_materialize_creates_user_skills_and_conversation_layout(directory):
    assert directory.root.is_dir()
    assert directory.skill_dir == directory.root / "skills"
    assert directory.conversation_dir == directory.root / "chat-1"
    assert directory.artifacts_dir == directory.conversation_dir


def test_materialize_preserves_existing_files_and_isolates_users(workspace):
    first = workspace.materialize("alice", "one")
    first.skill_dir.joinpath("custom.md").write_text("keep")
    second = workspace.materialize("alice", "two")
    other = workspace.materialize("bob", "one")
    assert second.skill_dir.joinpath("custom.md").read_text() == "keep"
    assert not other.skill_dir.joinpath("custom.md").exists()
    assert first.root == second.root
    assert first.conversation_dir != second.conversation_dir


@pytest.mark.parametrize("user,conversation", [("../bad", "ok"), ("ok", "../bad")])
def test_materialize_rejects_unsafe_identifiers(workspace, user, conversation):
    with pytest.raises(ValueError):
        workspace.materialize(user, conversation)


def test_execution_workspace_reports_binary_create_modify_delete(workspace, directory, execution_copy):
    directory.root.joinpath("old.bin").write_bytes(b"before")
    directory.root.joinpath("gone.bin").write_bytes(b"delete me")
    copy = execution_copy()
    copy.root.joinpath("old.bin").write_bytes(b"after\x00")
    copy.root.joinpath("new.bin").write_bytes(b"\x00\xff")
    copy.root.joinpath("gone.bin").unlink()
    changes = copy.changes()
    assert changes.modified == ("old.bin",)
    assert changes.created == ("new.bin",)
    assert changes.deleted == ("gone.bin",)


def test_execution_workspace_publishes_and_discard_removes_copy(workspace, directory, execution_copy):
    directory.root.joinpath("old.txt").write_text("old")
    copy = execution_copy()
    copy.root.joinpath("old.txt").write_text("new")
    copy.root.joinpath("made.txt").write_text("made")
    copy.root.joinpath("old.txt").unlink()
    assert copy.publish().paths == ("made.txt", "old.txt")
    assert not directory.root.joinpath("old.txt").exists()
    assert directory.root.joinpath("made.txt").read_text() == "made"
    root = copy._container
    copy.discard()
    assert not root.exists()


def test_publish_preserves_unrelated_concurrent_edits(workspace, directory, execution_copy):
    directory.root.joinpath("tracked.txt").write_text("base")
    copy = execution_copy()
    copy.root.joinpath("tracked.txt").write_text("copy")
    directory.root.joinpath("other.txt").write_text("concurrent")
    copy.publish()
    assert directory.root.joinpath("tracked.txt").read_text() == "copy"
    assert directory.root.joinpath("other.txt").read_text() == "concurrent"


def test_publish_conflict_retains_copy_for_correction(workspace, directory, execution_copy):
    directory.root.joinpath("same.txt").write_text("base")
    copy = execution_copy()
    copy.root.joinpath("same.txt").write_text("copy")
    directory.root.joinpath("same.txt").write_text("host")
    with pytest.raises(WorkspaceConflictError) as error:
        copy.publish()
    assert error.value.paths == ("same.txt",)
    assert copy.root.joinpath("same.txt").read_text() == "copy"


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_execution_workspace_rejects_symlink_and_special_files(workspace, directory, kind):
    if kind == "symlink":
        directory.root.joinpath("link").symlink_to("missing")
    else:
        import os
        os.mkfifo(directory.root / "pipe")
    with pytest.raises(ValueError):
        ExecutionWorkspace(workspace, "alice", "chat-1")


def test_discarded_execution_workspace_cannot_be_used(workspace, execution_copy):
    copy = execution_copy()
    copy.discard()
    with pytest.raises(RuntimeError):
        copy.changes()
