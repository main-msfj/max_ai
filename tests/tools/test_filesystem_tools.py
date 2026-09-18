from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from max_ai.base.tools import ToolContext
from max_ai.capabilities.workspace.local import WorkspaceLocal
from max_ai.config import setting
from max_ai.capabilities.tools.file_system import FileSystemTools
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode
from max_ai.workspace_copy.filesystem import UserFileSystem


def context(user_id: str, session_id: str, **deps: object) -> ToolContext:
    return ToolContext(
        run_id="run-test",
        user_id=user_id,
        session_id=session_id,
        deps=deps,
    )


def get_tool(tools: FileSystemTools, name: str):
    return next(tool for tool in tools.tools if tool.name == name)


async def invoke(tools: FileSystemTools, name: str, ctx: ToolContext, **params):
    tool = get_tool(tools, name)
    record = ToolCallRecord(tool_name=name, parameters=params)
    return await tool.execute(record, ctx)


@pytest.mark.asyncio
async def test_user_scope_conversations_find_write_and_cross_conversation_edit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Workspace"
    tools = FileSystemTools(root)
    u1_s1 = context("u1", "s1")
    u1_s2 = context("u1", "s2")
    u2_s1 = context("u2", "s1")

    first = await invoke(
        tools,
        "write_file",
        u1_s1,
        path="reports/summary.txt",
        content="first report",
    )
    second = await invoke(
        tools,
        "write_file",
        u1_s2,
        path="reports/summary.txt",
        content="second report",
    )
    other_user = await invoke(
        tools,
        "write_file",
        u2_s1,
        path="reports/summary.txt",
        content="private report",
    )
    assert first.success and second.success and other_user.success
    assert first.result["path"] == "s1/reports/summary.txt"
    assert (root / "u1" / "s2" / "reports" / "summary.txt").read_text() == "second report"

    found = await invoke(tools, "find_files", u1_s2, pattern="summary.txt")
    assert found.success
    assert [item["path"] for item in found.result["items"]] == [
        "s1/reports/summary.txt",
        "s2/reports/summary.txt",
    ]

    read_previous_session = await invoke(
        tools,
        "read_file",
        u1_s2,
        path="s1/reports/summary.txt",
    )
    assert read_previous_session.success
    assert read_previous_session.result["content"] == "first report"

    digest = hashlib.sha256(b"first report").hexdigest()
    edited = await invoke(
        tools,
        "edit_file",
        u1_s2,
        path="s1/reports/summary.txt",
        old_text="first",
        new_text="updated",
        expected_sha256=digest,
    )
    assert edited.success
    reread = await invoke(
        tools,
        "read_file",
        u1_s1,
        path="s1/reports/summary.txt",
    )
    assert reread.success and reread.result["content"] == "updated report"

    other_user_list = await invoke(tools, "find_files", u2_s1, pattern="summary.txt")
    assert other_user_list.success
    assert [item["path"] for item in other_user_list.result["items"]] == [
        "s1/reports/summary.txt"
    ]
    assert other_user_list.result["items"][0]["bytes"] == len("private report")


@pytest.mark.asyncio
async def test_traversal_absolute_paths_and_cross_user_access_are_rejected(
    tmp_path: Path,
) -> None:
    tools = FileSystemTools(tmp_path / "Workspace")
    writer = await invoke(
        tools,
        "write_file",
        context("alice", "s1"),
        path="secret.txt",
        content="private",
    )
    assert writer.success

    for path in ("../alice/s1/secret.txt", "/alice/s1/secret.txt"):
        result = await invoke(
            tools, "read_file", context("bob", "s1"), path=path
        )
        assert result.success is False
        assert result.error


@pytest.mark.asyncio
async def test_symlinks_hardlinks_and_nonregular_files_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Workspace"
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    created = await invoke(tools, "write_file", ctx, path="source.txt", content="safe")
    assert created.success
    source = root / "u1" / "s1" / "source.txt"
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (source.parent / "linked.txt").symlink_to(outside)
    os.link(source, source.parent / "hardlinked.txt")
    os.mkfifo(source.parent / "pipe.txt")

    for path in ("s1/linked.txt", "s1/hardlinked.txt", "s1/pipe.txt"):
        result = await invoke(tools, "read_file", ctx, path=path)
        assert result.success is False


@pytest.mark.asyncio
async def test_symlinked_user_and_conversation_roots_cannot_cross_users(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Workspace"
    victim_conversation = root / "user-2" / "conversation-2"
    victim_conversation.mkdir(parents=True)
    (victim_conversation / "secret.txt").write_text("private", encoding="utf-8")
    (root / "linked-user").symlink_to(root / "user-2", target_is_directory=True)

    tools = FileSystemTools(root)
    filesystem = UserFileSystem(root)

    linked_user_write = await invoke(
        tools,
        "write_file",
        context("linked-user", "conversation-2"),
        path="attempt.txt",
        content="must not escape",
    )
    assert linked_user_write.success is False
    linked_user_read = await invoke(
        tools,
        "read_file",
        context("linked-user", "conversation-2"),
        path="conversation-2/secret.txt",
    )
    assert linked_user_read.success is False
    with pytest.raises(ValueError):
        filesystem.user_root("linked-user")
    with pytest.raises(ValueError):
        filesystem.conversation_root("linked-user", "conversation-2")

    user_root = root / "user-1"
    user_root.mkdir()
    (user_root / "linked-conversation").symlink_to(
        victim_conversation, target_is_directory=True
    )
    linked_conversation_write = await invoke(
        tools,
        "write_file",
        context("user-1", "linked-conversation"),
        path="attempt.txt",
        content="must not escape",
    )
    assert linked_conversation_write.success is False
    linked_conversation_read = await invoke(
        tools,
        "read_file",
        context("user-1", "linked-conversation"),
        path="linked-conversation/secret.txt",
    )
    assert linked_conversation_read.success is False
    with pytest.raises(ValueError):
        filesystem.conversation_root("user-1", "linked-conversation")
    assert not (victim_conversation / "attempt.txt").exists()


@pytest.mark.asyncio
async def test_binary_files_return_metadata_and_are_skipped_by_search(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Workspace"
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    binary = b"needle\x00binary payload"
    binary_path = root / "u1" / "s1" / "binary.dat"
    binary_path.parent.mkdir(parents=True)
    binary_path.write_bytes(binary)

    read = await invoke(tools, "read_file", ctx, path="s1/binary.dat")
    assert read.success
    assert read.result["binary"] is True
    assert read.result["bytes"] == len(binary)
    assert read.result["sha256"] == hashlib.sha256(binary).hexdigest()
    assert "content" not in read.result

    searched = await invoke(tools, "search_text", ctx, query="needle")
    assert searched.success and searched.result["matches"] == []


@pytest.mark.asyncio
async def test_write_file_is_create_only(tmp_path: Path) -> None:
    tools = FileSystemTools(tmp_path / "Workspace")
    ctx = context("u1", "s1")
    created = await invoke(
        tools, "write_file", ctx, path="note.txt", content="original"
    )
    duplicate = await invoke(
        tools, "write_file", ctx, path="note.txt", content="replacement"
    )

    assert created.success
    assert duplicate.success is False
    current = await invoke(tools, "read_file", ctx, path="s1/note.txt")
    assert current.success and current.result["content"] == "original"


@pytest.mark.asyncio
async def test_stale_digest_does_not_modify_file(tmp_path: Path) -> None:
    tools = FileSystemTools(tmp_path / "Workspace")
    ctx = context("u1", "s1")
    created = await invoke(tools, "write_file", ctx, path="note.txt", content="original")
    assert created.success

    edited = await invoke(
        tools,
        "edit_file",
        ctx,
        path="s1/note.txt",
        old_text="original",
        new_text="changed",
        expected_sha256="0" * 64,
    )
    assert edited.success is False
    current = await invoke(tools, "read_file", ctx, path="s1/note.txt")
    assert current.success and current.result["content"] == "original"


@pytest.mark.asyncio
async def test_edit_rejects_ambiguous_exact_replacement(tmp_path: Path) -> None:
    tools = FileSystemTools(tmp_path / "Workspace")
    ctx = context("u1", "s1")
    content = "repeat this, then repeat this"
    created = await invoke(
        tools, "write_file", ctx, path="note.txt", content=content
    )
    assert created.success

    edited = await invoke(
        tools,
        "edit_file",
        ctx,
        path="s1/note.txt",
        old_text="repeat",
        new_text="replace",
        expected_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    assert edited.success is False
    current = await invoke(tools, "read_file", ctx, path="s1/note.txt")
    assert current.success and current.result["content"] == content


@pytest.mark.asyncio
async def test_read_and_listing_limits_are_reported(tmp_path: Path) -> None:
    tools = FileSystemTools(tmp_path / "Workspace")
    ctx = context("u1", "s1")
    for name in ("a.txt", "b.txt"):
        result = await invoke(tools, "write_file", ctx, path=name, content="alpha\nbeta")
        assert result.success

    listed = await invoke(tools, "list_files", ctx, path="s1", limit=1)
    assert listed.success
    assert len(listed.result["items"]) == 1
    assert listed.result["truncated"] is True

    read = await invoke(
        tools, "read_file", ctx, path="s1/a.txt", max_bytes=3
    )
    assert read.success
    assert read.result["content"] == "alp"
    assert read.result["truncated"] is True
    assert read.result["sha256"] == hashlib.sha256(b"alpha\nbeta").hexdigest()

    searched = await invoke(
        tools, "search_text", ctx, query="a", path="s1", limit=1
    )
    assert searched.success
    assert len(searched.result["matches"]) == 1
    assert searched.result["truncated"] is True


def test_tool_schemas_hide_context_and_root_and_approval_modes(tmp_path: Path) -> None:
    tools = FileSystemTools(tmp_path / "explicit-root")
    assert [tool.name for tool in tools.tools] == [
        "list_files",
        "find_files",
        "search_text",
        "read_file",
        "write_file",
        "edit_file",
    ]
    for tool in tools.tools:
        properties = tool.parameters.get("properties", {})
        assert "context" not in properties
        assert "user_id" not in properties
        assert "session_id" not in properties
        assert "root" not in properties
        assert "filesystem_root" not in properties
    assert get_tool(tools, "list_files").approval_mode == ToolApprovalMode.AUTO_APPROVED
    assert get_tool(tools, "read_file").approval_mode == ToolApprovalMode.AUTO_APPROVED
    assert get_tool(tools, "write_file").approval_mode == ToolApprovalMode.ASK_APPROVED
    assert get_tool(tools, "edit_file").approval_mode == ToolApprovalMode.ASK_APPROVED


@pytest.mark.asyncio
async def test_explicit_root_overrides_dependency_root_and_dependency_root_is_used(
    tmp_path: Path,
) -> None:
    explicit_root = tmp_path / "explicit"
    dependency_root = tmp_path / "dependency"
    tools = FileSystemTools(explicit_root)
    result = await invoke(
        tools,
        "write_file",
        context("u1", "s1", filesystem_root=dependency_root),
        path="explicit.txt",
        content="explicit",
    )
    assert result.success
    assert (explicit_root / "u1" / "s1" / "explicit.txt").exists()
    assert not dependency_root.exists()

    dependency_tools = FileSystemTools()
    result = await invoke(
        dependency_tools,
        "write_file",
        context("u1", "s1", filesystem_root=dependency_root),
        path="dependency.txt",
        content="dependency",
    )
    assert result.success
    assert (dependency_root / "u1" / "s1" / "dependency.txt").exists()


def test_user_filesystem_validates_single_segment_ids(tmp_path: Path) -> None:
    filesystem = UserFileSystem(tmp_path / "Workspace")
    user_root = filesystem.user_root("user-1")
    assert user_root == (tmp_path / "Workspace" / "user-1").resolve()
    assert user_root.is_dir()
    conversation_root = filesystem.conversation_root("user-1", "session_1")
    assert conversation_root == (
        tmp_path / "Workspace" / "user-1" / "session_1"
    ).resolve()
    assert conversation_root.is_dir()
    with pytest.raises(ValueError):
        filesystem.user_root("../other")
    with pytest.raises(ValueError):
        filesystem.conversation_root("user-1", "session/other")


@pytest.mark.asyncio
async def test_conversation_roots_reject_reserved_ids_before_materializing(
    tmp_path: Path,
) -> None:
    filesystem = UserFileSystem(tmp_path / "Workspace")
    tools = FileSystemTools(filesystem)

    for reserved_id in ("tools", "skills", "artifacts"):
        with pytest.raises(ValueError, match="reserved"):
            filesystem.conversation_root("user-1", reserved_id)
        write = await invoke(
            tools,
            "write_file",
            context("user-1", reserved_id),
            path="note.txt",
            content="must be rejected",
        )
        assert write.success is False

    assert not (tmp_path / "Workspace" / "user-1").exists()


def test_root_materialization_rejects_symlinked_user_and_conversation_dirs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Workspace"
    filesystem = UserFileSystem(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "linked-user").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        filesystem.user_root("linked-user")

    (root / "user-1").mkdir()
    (root / "user-1" / "linked-conversation").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(ValueError):
        filesystem.conversation_root("user-1", "linked-conversation")


@pytest.mark.asyncio
async def test_internal_temporary_files_are_hidden_from_paths_and_scans(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Workspace"
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    conversation = root / "u1" / "s1"
    conversation.mkdir(parents=True)
    (conversation / "visible.txt").write_text("published", encoding="utf-8")
    temp_name = ".maxai-0123456789abcdef01234567.tmp"
    (conversation / temp_name).write_text("partially written", encoding="utf-8")

    listed = await invoke(tools, "list_files", ctx, path="s1")
    assert listed.success
    assert [item["path"] for item in listed.result["items"]] == [
        "s1/visible.txt"
    ]

    found = await invoke(tools, "find_files", ctx, pattern="maxai")
    assert found.success and found.result["items"] == []
    searched = await invoke(tools, "search_text", ctx, query="partially written")
    assert searched.success and searched.result["matches"] == []

    read_temp = await invoke(
        tools, "read_file", ctx, path=f"s1/{temp_name}"
    )
    assert read_temp.success is False


def test_workspace_materialize_rejects_traversal_and_runtime_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Workspace"
    workspace = WorkspaceLocal(root)

    with pytest.raises(ValueError):
        workspace.map_directory("../victim")
    assert not root.exists()

    victim = root / "victim"
    victim.mkdir(parents=True)
    (root / "linked-user").symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError):
        workspace.materialize("linked-user")
    assert list(victim.iterdir()) == []

    user_root = root / "alice"
    user_root.mkdir()
    (user_root / "tools").symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError):
        workspace.materialize("alice")
    assert list(victim.iterdir()) == []


@pytest.mark.parametrize(
    "directory_setting", ("tool_dir", "skill_dir", "artifacts_dir")
)
def test_workspace_materialize_validates_runtime_directory_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_setting: str
) -> None:
    root = tmp_path / "Workspace"
    monkeypatch.setattr(setting, directory_setting, "../outside")
    workspace = WorkspaceLocal(root)

    with pytest.raises(ValueError):
        workspace.materialize("alice")
    assert not root.exists()
