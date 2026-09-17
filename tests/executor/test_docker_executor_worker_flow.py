from __future__ import annotations

from pathlib import Path

import pytest

from max_ai.base.tools import ToolContext
from max_ai.executor.docker import DockerExecutor
from max_ai.executor.docker import worker
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import DockerToolRef


def _repo_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    package = repo / "max_ai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='maxai'\n", encoding="utf-8")
    (repo / "README.md").write_text("readme\n", encoding="utf-8")
    return repo


def test_docker_executor_session_key_scopes_canonical_root_user_and_session(
    tmp_path: Path,
) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))
    root = tmp_path / "Workspace"
    executor.workspace_root = root.resolve()
    context = ToolContext(
        run_id="run_1",
        user_id="User_1",
        session_id="session_1",
        deps={"filesystem_root": root},
    )
    same_scope = ToolContext(
        run_id="run_2",
        user_id="User_1",
        session_id="session_1",
        deps={"filesystem_root": root / "nested" / ".."},
    )
    other_user = ToolContext(
        run_id="run_1",
        user_id="user_1",
        session_id="session_1",
        deps={"filesystem_root": root},
    )
    other_root = ToolContext(
        run_id="run_1",
        user_id="User_1",
        session_id="session_1",
        deps={"filesystem_root": tmp_path / "OtherWorkspace"},
    )

    assert executor._bash_session_key(context) == executor._bash_session_key(
        same_scope
    )
    assert executor._bash_session_key(context) != executor._bash_session_key(other_user)
    assert executor._bash_session_key(context) != executor._bash_session_key(other_root)
    assert executor._bash_container_name(context) != executor._bash_container_name(
        other_user
    )


def test_docker_executor_session_key_uses_exact_identifier_without_normalization(
    tmp_path: Path,
) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))
    context = ToolContext(
        run_id="run_1",
        user_id="user_1",
        session_id="Session_1",
        deps={"filesystem_root": tmp_path / "Workspace"},
    )
    other_session = ToolContext(
        run_id="run_1",
        user_id="user_1",
        session_id="session_1",
        deps={"filesystem_root": tmp_path / "Workspace"},
    )
    run_scope = ToolContext(
        run_id="run_1",
        user_id="user_1",
        deps={"filesystem_root": tmp_path / "Workspace"},
    )

    assert executor._bash_session_key(context) != executor._bash_session_key(
        other_session
    )
    assert executor._bash_session_key(context) != executor._bash_session_key(run_scope)


def test_docker_executor_uses_session_root_for_new_filesystem_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))
    root = tmp_path / "Workspace"
    executor.workspace_root = root.resolve()
    conversation_root = root / "user_1" / "session_1"
    context = ToolContext(
        run_id="run_1",
        user_id="user_1",
        session_id="session_1",
        deps={
            "filesystem_root": root,
            "artifacts_dir": conversation_root,
        },
    )
    monkeypatch.setattr(executor, "_docker_visible_path", lambda path: path)

    host_root, cwd, artifacts = executor._runtime_layout_for_context(context)

    assert host_root == root / "user_1"
    assert cwd == "/mnt/session_1"
    assert artifacts == "/mnt/session_1"
    assert (host_root / "tools").is_dir()
    assert (host_root / "skills").is_dir()
    assert conversation_root.is_dir()
    assert not (host_root / "artifacts").exists()
    assert executor._docker_mount_args(host_root) == ["-v", f"{host_root}:/mnt"]
    assert executor._docker_env_args(artifacts) == [
        "-e",
        "RUNTIME_DIR=/mnt",
        "-e",
        "TOOLS_DIR=/mnt/tools",
        "-e",
        "SKILLS_DIR=/mnt/skills",
        "-e",
        "ARTIFACTS_DIR=/mnt/session_1",
        "-e",
        "WORKSPACE_DIR=/mnt/session_1",
    ]


def test_docker_executor_rejects_filesystem_root_that_differs_from_binding(
    tmp_path: Path,
) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))
    executor.workspace_root = (tmp_path / "bound").resolve()
    context = ToolContext(
        run_id="run_1",
        user_id="user_1",
        session_id="session_1",
        deps={"filesystem_root": tmp_path / "dependency"},
    )

    with pytest.raises(ValueError, match="must match the filesystem_root"):
        executor._runtime_layout_for_context(context)


@pytest.mark.asyncio
async def test_docker_executor_validates_session_before_connecting(
    tmp_path: Path, monkeypatch
) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))
    root = tmp_path / "Workspace"
    executor.workspace_root = root.resolve()
    context = ToolContext(
        run_id="run_1",
        user_id="user_1",
        session_id="..",
        deps={"filesystem_root": root},
    )
    connected = False

    async def ensure_connected() -> None:
        nonlocal connected
        connected = True

    monkeypatch.setattr(executor, "_ensure_connected", ensure_connected)

    result = await executor.run(
        tool=None,  # The invalid context must be rejected before tool dispatch.
        record=ToolCallRecord(tool_name="bash"),
        tool_context=context,
    )

    assert not connected
    assert not result.success
    assert "session_id" in result.error


def test_docker_executor_rejects_parent_directory_user_id() -> None:
    with pytest.raises(ValueError):
        DockerExecutor._clean_user_id("..")


def test_docker_executor_container_path_joins_under_mnt() -> None:
    executor = DockerExecutor()

    assert executor._container_path("skills") == "/mnt/skills"
    assert executor._container_path("/artifacts/") == "/mnt/artifacts"


def test_docker_executor_truncates_large_output() -> None:
    text, truncated, original = DockerExecutor._truncate_text("abcdef", 3)

    assert text == "abc\n[output truncated]"
    assert truncated is True
    assert original == 6


def test_worker_falls_back_to_builtin_bash_when_module_is_missing(monkeypatch) -> None:
    def missing_ref(module_name: str, qualname: str):
        raise ModuleNotFoundError("No module named 'max_ai.tools.bash'")

    monkeypatch.setattr(worker, "_resolve_ref", missing_ref)
    tool_ref = DockerToolRef(
        kind="class",
        module="max_ai.tools.bash",
        qualname="BashTool",
        config={"timeout_seconds": 10, "max_output_chars": 1000},
    )

    tool = worker._build_tool(tool_ref)

    assert tool.name == "bash"
    assert tool.timeout_seconds == 10


def test_worker_falls_back_to_builtin_skill_bash_when_module_is_missing(monkeypatch) -> None:
    def missing_ref(module_name: str, qualname: str):
        raise ModuleNotFoundError("No module named 'max_ai.tools.bash'")

    monkeypatch.setattr(worker, "_resolve_ref", missing_ref)
    tool_ref = DockerToolRef(
        kind="class",
        module="max_ai.tools.bash",
        qualname="SkillBashTool",
        config={"timeout_seconds": 10, "max_output_chars": 1000},
    )

    tool = worker._build_tool(tool_ref)

    assert tool.name == "bash"
    assert tool.timeout_seconds == 10
