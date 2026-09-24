"""Filesystem tools: one workspace per user, a scratchpad per session, and no
way out of either (traversal, symlinks, other users)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from max_ai.base.tools import ToolContext
from max_ai.capabilities.tools.file_system import FileSystemTools
from max_ai.capabilities.workspace.local import LocalWorkspace, UserFileSystem
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


def context(user_id: str, session_id: str, **deps: object) -> ToolContext:
    return ToolContext(run_id="run-test", user_id=user_id, session_id=session_id, deps=deps)


def get_tool(tools: FileSystemTools, name: str):
    return next(tool for tool in tools.tools if tool.name == name)


async def invoke(tools: FileSystemTools, name: str, ctx: ToolContext, **params):
    record = ToolCallRecord(tool_name=name, parameters=params)
    return await get_tool(tools, name).execute(record, ctx)


def refused(result) -> bool:
    """Refused by the filesystem, not by a mistyped tool parameter."""
    return result.success is False and "Invalid parameters" not in (result.error or "")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "Workspace"


# -------- SCOPE -----------------------------------------------------------
async def test_sessions_share_the_user_workspace_and_users_never_meet(root):
    tools = FileSystemTools(root)
    first = await invoke(tools, "write_file", context("u1", "s1"), file_name="reports/a.txt", content="uno")
    assert first.success and first.result["path"] == "workspace/reports/a.txt"
    assert (root / "u1" / "workspace" / "reports" / "a.txt").read_text() == "uno"

    # Another session of the same user sees and edits the same workspace.
    read = await invoke(tools, "read_file", context("u1", "s2"), file_name="reports/a.txt")
    assert read.success and read.result["content"] == "uno"
    edited = await invoke(tools, "edit_file", context("u1", "s2"), file_name="reports/a.txt",
                          old_text="uno", new_text="dos",
                          expected_sha256=hashlib.sha256(b"uno").hexdigest())
    assert edited.success

    # Another user has their own workspace and can't see this one.
    other = await invoke(tools, "read_file", context("u2", "s1"), file_name="reports/a.txt")
    assert refused(other)
    found = await invoke(tools, "find_files", context("u2", "s1"), file_name="a.txt")
    assert found.success and found.result["items"] == []


async def test_scratchpad_belongs_to_its_session(root):
    tools = FileSystemTools(root)
    written = await invoke(tools, "write_file", context("u1", "s1"), file_name="scratchpad/n.txt", content="tmp")
    assert written.success and written.result["path"] == "scratchpad/s1/n.txt"
    assert (await invoke(tools, "read_file", context("u1", "s1"), file_name="scratchpad/n.txt")).success
    assert refused(await invoke(tools, "read_file", context("u1", "s2"), file_name="scratchpad/n.txt"))


async def test_paths_returned_by_the_tools_can_be_passed_back(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    await invoke(tools, "write_file", ctx, file_name="reports/a.txt", content="hola")
    await invoke(tools, "write_file", ctx, file_name="scratchpad/n.txt", content="tmp")
    found = await invoke(tools, "find_files", ctx, file_name="a.txt")
    returned = found.result["items"][0]["path"]  # "workspace/reports/a.txt"
    assert (await invoke(tools, "read_file", ctx, file_name=returned)).result["content"] == "hola"
    listed = await invoke(tools, "list_directory", ctx, path="workspace/reports")
    assert [item["path"] for item in listed.result["items"]] == ["workspace/reports/a.txt"]
    scratch = await invoke(tools, "read_file", ctx, file_name="scratchpad/s1/n.txt")
    assert scratch.result["content"] == "tmp"


# -------- ESCAPES -----------------------------------------------------------
async def test_traversal_absolute_and_blocked_paths_are_refused(root):
    tools = FileSystemTools(root)
    assert (await invoke(tools, "write_file", context("alice", "s1"), file_name="secret.txt", content="x")).success
    bob = context("bob", "s1")
    for path in ("../alice/workspace/secret.txt", "/alice/workspace/secret.txt",
                 "reports/../../alice/workspace/secret.txt", "tools/x.py", "artifacts/x", ".hidden/x"):
        assert refused(await invoke(tools, "read_file", bob, file_name=path)), path
    for path in ("../escape.txt", "/etc/passwd", ".env"):
        assert refused(await invoke(tools, "write_file", bob, file_name=path, content="x")), path
    assert not (root / "escape.txt").exists()


async def test_symlinks_hardlinks_and_special_files_are_refused(root, tmp_path):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    assert (await invoke(tools, "write_file", ctx, file_name="source.txt", content="safe")).success
    workspace = root / "u1" / "workspace"
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "linked.txt").symlink_to(outside)
    (workspace / "linked-dir").symlink_to(tmp_path, target_is_directory=True)
    os.link(workspace / "source.txt", workspace / "hardlinked.txt")
    os.mkfifo(workspace / "pipe.txt")

    for path in ("linked.txt", "linked-dir/outside.txt", "hardlinked.txt", "pipe.txt"):
        assert refused(await invoke(tools, "read_file", ctx, file_name=path)), path
    assert refused(await invoke(tools, "write_file", ctx, file_name="linked-dir/new.txt", content="x"))
    assert not (tmp_path / "new.txt").exists()


async def test_a_symlinked_user_or_workspace_cannot_reach_another_user(root):
    victim = root / "victim" / "workspace"
    victim.mkdir(parents=True)
    (victim / "secret.txt").write_text("private", encoding="utf-8")
    (root / "linked-user").symlink_to(root / "victim", target_is_directory=True)
    (root / "mallory").mkdir()
    (root / "mallory" / "workspace").symlink_to(victim, target_is_directory=True)

    tools = FileSystemTools(root)
    for user in ("linked-user", "mallory"):
        ctx = context(user, "s1")
        assert refused(await invoke(tools, "read_file", ctx, file_name="secret.txt")), user
        assert refused(await invoke(tools, "write_file", ctx, file_name="attempt.txt", content="x")), user
    assert not (victim / "attempt.txt").exists()
    with pytest.raises(ValueError):
        UserFileSystem(root).user_root("linked-user")


def test_ids_must_be_single_safe_segments(root):
    filesystem = UserFileSystem(root)
    assert filesystem.workspace_root("user-1") == root.resolve() / "user-1" / "workspace"
    assert filesystem.scratchpad_root("user-1", "s_1") == root.resolve() / "user-1" / "scratchpad" / "s_1"
    for bad in ("../other", "a/b", "", "."):
        with pytest.raises(ValueError):
            filesystem.user_root(bad)
    with pytest.raises(ValueError):
        filesystem.scratchpad_root("user-1", "session/other")


async def test_reserved_names_are_not_session_ids(root):
    filesystem = UserFileSystem(root)
    tools = FileSystemTools(filesystem)
    for reserved in ("tools", "skills", "artifacts"):
        with pytest.raises(ValueError, match="reserved"):
            filesystem.scratchpad_root("user-1", reserved)
        write = await invoke(tools, "write_file", context("user-1", reserved),
                             file_name="scratchpad/note.txt", content="x")
        assert refused(write)


async def test_internal_temporary_files_are_invisible(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    workspace = root / "u1" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "visible.txt").write_text("published", encoding="utf-8")
    temp = ".maxai-0123456789abcdef01234567.tmp"
    (workspace / temp).write_text("partially written", encoding="utf-8")

    listed = await invoke(tools, "list_directory", ctx)
    assert [item["path"] for item in listed.result["items"]] == ["workspace/visible.txt"]
    assert (await invoke(tools, "find_files", ctx, file_name="*.tmp")).result["items"] == []
    assert (await invoke(tools, "search_text", ctx, query="partially")).result["matches"] == []
    assert refused(await invoke(tools, "read_file", ctx, file_name=temp))


def test_materialize_refuses_symlinked_user_and_runtime_dirs(root, tmp_path):
    workspace = LocalWorkspace(root=root)
    victim = tmp_path / "victim"
    victim.mkdir()
    root.mkdir(exist_ok=True)
    (root / "linked-user").symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError):
        workspace.materialize("linked-user", "s1")
    (root / "alice").mkdir()
    (root / "alice" / "skills").symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError):
        workspace.materialize("alice", "s1")
    with pytest.raises(ValueError):
        workspace.materialize("../victim", "s1")
    assert list(victim.iterdir()) == []


# -------- CONTENT -----------------------------------------------------------
async def test_binary_files_return_metadata_and_are_skipped_by_search(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    binary = b"needle\x00binary payload"
    path = root / "u1" / "workspace" / "binary.dat"
    path.parent.mkdir(parents=True)
    path.write_bytes(binary)

    read = await invoke(tools, "read_file", ctx, file_name="binary.dat")
    assert read.success and read.result["binary"] is True and "content" not in read.result
    assert read.result["sha256"] == hashlib.sha256(binary).hexdigest()
    assert (await invoke(tools, "search_text", ctx, query="needle")).result["matches"] == []


async def test_write_is_create_only_and_edits_need_the_current_digest(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    assert (await invoke(tools, "write_file", ctx, file_name="note.txt", content="original")).success
    assert refused(await invoke(tools, "write_file", ctx, file_name="note.txt", content="replacement"))
    stale = await invoke(tools, "edit_file", ctx, file_name="note.txt", old_text="original",
                         new_text="changed", expected_sha256="0" * 64)
    assert refused(stale)
    current = await invoke(tools, "read_file", ctx, file_name="note.txt")
    assert current.result["content"] == "original"


async def test_ambiguous_edits_and_deletes_with_a_stale_digest_are_refused(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    content = "repeat this, then repeat this"
    await invoke(tools, "write_file", ctx, file_name="note.txt", content=content)
    digest = hashlib.sha256(content.encode()).hexdigest()
    assert refused(await invoke(tools, "edit_file", ctx, file_name="note.txt", old_text="repeat",
                                new_text="replace", expected_sha256=digest))
    assert refused(await invoke(tools, "delete_file", ctx, file_name="note.txt", expected_sha256="0" * 64))
    assert (await invoke(tools, "delete_file", ctx, file_name="note.txt", expected_sha256=digest)).success
    assert not (root / "u1" / "workspace" / "note.txt").exists()


async def test_read_list_and_search_report_their_limits(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    for name in ("a.txt", "b.txt"):
        assert (await invoke(tools, "write_file", ctx, file_name=name, content="alpha\nbeta")).success

    listed = await invoke(tools, "list_directory", ctx, limit=1)
    assert len(listed.result["items"]) == 1 and listed.result["truncated"] is True
    read = await invoke(tools, "read_file", ctx, file_name="a.txt", max_bytes=3)
    assert (read.result["content"], read.result["truncated"]) == ("alp", True)
    assert read.result["sha256"] == hashlib.sha256(b"alpha\nbeta").hexdigest()
    searched = await invoke(tools, "search_text", ctx, query="a", limit=1)
    assert len(searched.result["matches"]) == 1 and searched.result["truncated"] is True


# -------- SCHEMAS & ROOTS -----------------------------------------------------------
def test_schemas_hide_the_context_and_writes_ask_for_approval(root):
    tools = FileSystemTools(root)
    assert [tool.name for tool in tools.tools] == [
        "list_directory", "find_files", "search_text", "read_file", "write_file",
        "edit_file", "create_directory", "file_info", "delete_file",
    ]
    for tool in tools.tools:
        properties = set(tool.parameters.get("properties", {}))
        assert not properties & {"context", "user_id", "session_id", "root", "filesystem_root"}
    approvals = {tool.name: tool.approval_mode for tool in tools.tools}
    assert {n for n, mode in approvals.items() if mode == ToolApprovalMode.ASK_APPROVED} == {
        "write_file", "edit_file", "delete_file",
    }


async def test_an_explicit_root_wins_over_the_dependency_root(tmp_path):
    explicit, dependency = tmp_path / "explicit", tmp_path / "dependency"
    ctx = context("u1", "s1", filesystem_root=dependency)
    assert (await invoke(FileSystemTools(explicit), "write_file", ctx, file_name="e.txt", content="x")).success
    assert (explicit / "u1" / "workspace" / "e.txt").exists() and not dependency.exists()
    assert (await invoke(FileSystemTools(), "write_file", ctx, file_name="d.txt", content="x")).success
    assert (dependency / "u1" / "workspace" / "d.txt").exists()
