"""Filesystem tools: one workspace per user, and no way out of it
(traversal, symlinks, other users)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from max_ai.base.tools import ToolContext
from max_ai.capabilities.tools.file_system import FileSystemTools
from max_ai.capabilities.workspace.local import LocalWorkspace, UserFileSystem
from max_ai.types.run_context import RunContext
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
    first = await invoke(tools, "WriteFile", context("u1", "s1"), file_path="reports/a.txt", content="uno")
    assert first.success and first.result["path"] == "workspace/reports/a.txt"
    assert (root / "u1" / "workspace" / "reports" / "a.txt").read_text() == "uno"

    # Another session of the same user sees and edits the same workspace.
    read = await invoke(tools, "ReadFile", context("u1", "s2"), file_path="reports/a.txt")
    assert read.success and read.result["content"] == "     1\tuno"
    edited = await invoke(tools, "EditFile", context("u1", "s2"), file_path="reports/a.txt",
                          old_string="uno", new_string="dos")
    assert edited.success

    # Another user has their own workspace and can't see this one.
    other = await invoke(tools, "ReadFile", context("u2", "s1"), file_path="reports/a.txt")
    assert refused(other)
    found = await invoke(tools, "FindFiles", context("u2", "s1"), pattern="a.txt")
    assert found.success and found.result["files"] == []


async def test_paths_returned_by_the_tools_can_be_passed_back(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    await invoke(tools, "WriteFile", ctx, file_path="reports/a.txt", content="hola")
    found = await invoke(tools, "FindFiles", ctx, pattern="a.txt")
    returned = found.result["files"][0]  # "workspace/reports/a.txt"
    assert (await invoke(tools, "ReadFile", ctx, file_path=returned)).result["content"] == "     1\thola"
    listed = await invoke(tools, "ListDirectory", ctx, path="workspace/reports")
    assert [item["path"] for item in listed.result["items"]] == ["workspace/reports/a.txt"]


async def test_binary_formats_are_refused_with_a_way_forward(root):
    tools = FileSystemTools(root)
    result = await invoke(tools, "WriteFile", context("u1", "s1"), file_path="trip.XLSX", content="UEsDBBQ")
    assert refused(result) and "running code" in result.error
    assert not (root / "u1" / "workspace" / "trip.XLSX").exists()
    assert (await invoke(tools, "WriteFile", context("u1", "s1"), file_path="make_trip.py", content="x")).success


# -------- ESCAPES -----------------------------------------------------------
async def test_traversal_absolute_and_blocked_paths_are_refused(root):
    tools = FileSystemTools(root)
    assert (await invoke(tools, "WriteFile", context("alice", "s1"), file_path="secret.txt", content="x")).success
    bob = context("bob", "s1")
    for path in ("../alice/workspace/secret.txt", "/alice/workspace/secret.txt",
                 "reports/../../alice/workspace/secret.txt", "tools/x.py", "artifacts/x", ".hidden/x"):
        assert refused(await invoke(tools, "ReadFile", bob, file_path=path)), path
    for path in ("../escape.txt", "/etc/passwd", ".env"):
        assert refused(await invoke(tools, "WriteFile", bob, file_path=path, content="x")), path
    assert not (root / "escape.txt").exists()


async def test_symlinks_hardlinks_and_special_files_are_refused(root, tmp_path):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    assert (await invoke(tools, "WriteFile", ctx, file_path="source.txt", content="safe")).success
    workspace = root / "u1" / "workspace"
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "linked.txt").symlink_to(outside)
    (workspace / "linked-dir").symlink_to(tmp_path, target_is_directory=True)
    os.link(workspace / "source.txt", workspace / "hardlinked.txt")
    os.mkfifo(workspace / "pipe.txt")

    for path in ("linked.txt", "linked-dir/outside.txt", "hardlinked.txt", "pipe.txt"):
        assert refused(await invoke(tools, "ReadFile", ctx, file_path=path)), path
    assert refused(await invoke(tools, "WriteFile", ctx, file_path="linked-dir/new.txt", content="x"))
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
        assert refused(await invoke(tools, "ReadFile", ctx, file_path="secret.txt")), user
        assert refused(await invoke(tools, "WriteFile", ctx, file_path="attempt.txt", content="x")), user
    assert not (victim / "attempt.txt").exists()
    with pytest.raises(ValueError):
        UserFileSystem(root).user_root("linked-user")


def test_ids_must_be_single_safe_segments(root):
    filesystem = UserFileSystem(root)
    assert filesystem.workspace_root("user-1") == root.resolve() / "user-1" / "workspace"
    for bad in ("../other", "a/b", "", "."):
        with pytest.raises(ValueError):
            filesystem.user_root(bad)
    with pytest.raises(ValueError):
        UserFileSystem._safe_session_id("session/other")


def test_reserved_names_are_not_session_ids():
    for reserved in ("tools", "skills", "artifacts"):
        with pytest.raises(ValueError, match="reserved"):
            UserFileSystem._safe_session_id(reserved)


async def test_internal_temporary_files_are_invisible(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    workspace = root / "u1" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "visible.txt").write_text("published", encoding="utf-8")
    temp = ".maxai-0123456789abcdef01234567.tmp"
    (workspace / temp).write_text("partially written", encoding="utf-8")

    listed = await invoke(tools, "ListDirectory", ctx)
    assert [item["path"] for item in listed.result["items"]] == ["workspace/visible.txt"]
    assert (await invoke(tools, "FindFiles", ctx, pattern="*.tmp")).result["files"] == []
    assert (await invoke(tools, "SearchFile", ctx, pattern="partially")).result["matches"] == []
    assert refused(await invoke(tools, "ReadFile", ctx, file_path=temp))


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

    read = await invoke(tools, "ReadFile", ctx, file_path="binary.dat")
    assert read.success and read.result["binary"] is True and "content" not in read.result
    assert read.result["bytes"] == len(binary) and "sha256" not in read.result
    assert (await invoke(tools, "SearchFile", ctx, pattern="needle")).result["matches"] == []
    # Reading a binary file (its size only) is enough to delete it.
    assert (await invoke(tools, "DeleteFile", ctx, file_path="binary.dat")).success


async def test_overwrites_edits_and_deletes_need_a_fresh_read(root):
    tools = FileSystemTools(root)
    writer, other = context("u1", "s1"), context("u1", "s2")
    note = root / "u1" / "workspace" / "note.txt"
    assert (await invoke(tools, "WriteFile", writer, file_path="note.txt", content="original")).success

    # A session that never read the file can't overwrite, edit or delete it.
    assert refused(await invoke(tools, "WriteFile", other, file_path="note.txt", content="x"))
    assert refused(await invoke(tools, "EditFile", other, file_path="note.txt",
                                old_string="original", new_string="x"))
    assert refused(await invoke(tools, "DeleteFile", other, file_path="note.txt"))
    assert note.read_text() == "original"

    # After reading, it can; its own write counts as a read for the next edit.
    await invoke(tools, "ReadFile", other, file_path="note.txt")
    assert (await invoke(tools, "WriteFile", other, file_path="note.txt", content="second")).success
    assert (await invoke(tools, "EditFile", other, file_path="note.txt",
                         old_string="second", new_string="third")).success

    # Changed by someone else (bash, the user): read it again first.
    note.write_text("changed outside")
    assert refused(await invoke(tools, "EditFile", other, file_path="note.txt",
                                old_string="changed", new_string="x"))
    assert refused(await invoke(tools, "DeleteFile", other, file_path="note.txt"))
    await invoke(tools, "ReadFile", other, file_path="note.txt")
    assert (await invoke(tools, "DeleteFile", other, file_path="note.txt")).success
    assert not note.exists()


async def test_ambiguous_edits_are_refused(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    content = "repeat this, then repeat this"
    await invoke(tools, "WriteFile", ctx, file_path="note.txt", content=content)
    assert refused(await invoke(tools, "EditFile", ctx, file_path="note.txt", old_string="repeat",
                                new_string="replace"))
    assert (root / "u1" / "workspace" / "note.txt").read_text() == content


async def test_reads_are_kept_in_the_run_context(root):
    tools = FileSystemTools(root)
    run = RunContext()
    ctx = context("u1", "s1", run_context=run)
    await invoke(tools, "WriteFile", ctx, file_path="a.txt", content="uno")
    assert set(run.file_reads) == {"workspace/a.txt"}

    # Saved with the session and loaded back, the read still counts.
    restored = RunContext.model_validate_json(run.model_dump_json())
    again = context("u1", "s1", run_context=restored)
    assert (await invoke(tools, "EditFile", again, file_path="a.txt",
                         old_string="uno", new_string="dos")).success


async def test_read_list_and_search_report_their_limits(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    for name in ("a.txt", "b.txt"):
        assert (await invoke(tools, "WriteFile", ctx, file_path=name, content="alpha\nbeta")).success

    listed = await invoke(tools, "ListDirectory", ctx, limit=1)
    assert len(listed.result["items"]) == 1 and listed.result["truncated"] is True
    searched = await invoke(tools, "SearchFile", ctx, pattern="a", limit=1)
    assert len(searched.result["matches"]) == 1 and searched.result["truncated"] is True


# -------- SCHEMAS & ROOTS -----------------------------------------------------------
def test_schemas_describe_every_parameter_and_writes_ask_for_approval(root):
    tools = FileSystemTools(root)
    assert [tool.name for tool in tools.tools] == [
        "ReadFile", "WriteFile", "EditFile", "DeleteFile", "ListDirectory", "FindFiles", "SearchFile",
    ]
    for tool in tools.tools:
        properties = tool.parameters.get("properties", {})
        assert not set(properties) & {"context", "user_id", "session_id", "root", "filesystem_root"}
        assert all(spec.get("description") for spec in properties.values()), tool.name
    approvals = {tool.name: tool.approval_mode for tool in tools.tools}
    assert {n for n, mode in approvals.items() if mode == ToolApprovalMode.ASK_APPROVED} == {
        "WriteFile", "EditFile", "DeleteFile",
    }


async def test_an_explicit_root_wins_over_the_dependency_root(tmp_path):
    explicit, dependency = tmp_path / "explicit", tmp_path / "dependency"
    ctx = context("u1", "s1", filesystem_root=dependency)
    assert (await invoke(FileSystemTools(explicit), "WriteFile", ctx, file_path="e.txt", content="x")).success
    assert (explicit / "u1" / "workspace" / "e.txt").exists() and not dependency.exists()
    assert (await invoke(FileSystemTools(), "WriteFile", ctx, file_path="d.txt", content="x")).success
    assert (dependency / "u1" / "workspace" / "d.txt").exists()


# -------- CLAUDE CODE STYLE -----------------------------------------------------------
async def test_read_numbers_lines_and_pages_long_files(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    workspace = root / "u1" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "long.txt").write_text("\n".join(f"line {n}" for n in range(1, 2501)) + "\n")
    (workspace / "wide.txt").write_text("x" * 2500)

    first = (await invoke(tools, "ReadFile", ctx, file_path="long.txt")).result
    assert first["content"].splitlines()[0] == "     1\tline 1"
    assert first["lines"] == "1-2000 of 2500" and "offset=2001" in first["note"]
    rest = (await invoke(tools, "ReadFile", ctx, file_path="long.txt", offset=2499, limit=5)).result
    assert rest["content"] == "  2499\tline 2499\n  2500\tline 2500" and "note" not in rest
    wide = (await invoke(tools, "ReadFile", ctx, file_path="wide.txt")).result["content"]
    assert wide.endswith("x…") and len(wide) == 7 + 2000 + 1


async def test_edit_replaces_once_or_everywhere(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    note = root / "u1" / "workspace" / "note.txt"
    await invoke(tools, "WriteFile", ctx, file_path="note.txt", content="a cat, a cat")

    repeated = await invoke(tools, "EditFile", ctx, file_path="note.txt", old_string="cat", new_string="dog")
    assert refused(repeated) and "appears 2 times" in repeated.error
    missing = await invoke(tools, "EditFile", ctx, file_path="note.txt", old_string="bird", new_string="x")
    assert refused(missing) and "was not found" in missing.error
    everywhere = await invoke(tools, "EditFile", ctx, file_path="note.txt", old_string="cat",
                              new_string="dog", replace_all=True)
    assert everywhere.result["replacements"] == 2 and note.read_text() == "a dog, a dog"


async def test_find_globs_and_search_regex(root):
    tools = FileSystemTools(root)
    ctx = context("u1", "s1")
    for name, text in (("reports/q1.md", "total: 10"), ("reports/2024/q2.md", "Total: 20"),
                       ("data/sales.csv", "total,30"), ("notes.txt", "nothing")):
        await invoke(tools, "WriteFile", ctx, file_path=name, content=text)

    async def find(pattern: str, **extra) -> list[str]:
        return (await invoke(tools, "FindFiles", ctx, pattern=pattern, **extra)).result["files"]

    assert await find("*.md") == ["workspace/reports/2024/q2.md", "workspace/reports/q1.md"]
    assert await find("**/*.md") == await find("*.md")
    assert await find("reports/*.md") == ["workspace/reports/2024/q2.md", "workspace/reports/q1.md"]
    assert await find("*.csv", path="data") == ["workspace/data/sales.csv"]
    assert await find("sales") == ["workspace/data/sales.csv"]

    async def search(pattern: str, **extra) -> list[str]:
        result = await invoke(tools, "SearchFile", ctx, pattern=pattern, **extra)
        return [f'{m["path"]}:{m["line"]}' for m in result.result["matches"]]

    assert await search(r"total[:,] \d+|total,\d+") == ["workspace/data/sales.csv:1", "workspace/reports/q1.md:1"]
    assert await search("total", glob="*.md", ignore_case=True) == [
        "workspace/reports/2024/q2.md:1", "workspace/reports/q1.md:1"]
    bad = await invoke(tools, "SearchFile", ctx, pattern="(")
    assert refused(bad) and "regular expression" in bad.error
