from __future__ import annotations

import time
from pathlib import Path

import pytest

from max_ai.base.executor import CoreExecutor
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.executor.docker.docker import DockerExecutor, _BashSession
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode


class LayoutExecutor(CoreExecutor):
    async def bind_to_workspace(self, workspace_registry_root: str | Path) -> None:
        self.workspace_root = Path(workspace_registry_root)

    async def run(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        cancellation_token=None,
    ) -> ToolResult:
        return ToolResult.success_result(record.id, {})


class LayoutExecutorTool(CoreTool):
    def __init__(self) -> None:
        super().__init__(
            name="layout",
            description="Layout test tool",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": True}

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context=None,
        cancellation_token=None,
    ) -> ToolResult:
        return ToolResult.success_result(tool_request.id, {})


class TrackingDockerExecutor(DockerExecutor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.removed: list[str] = []

    async def _remove_container(self, name: str) -> None:
        self.removed.append(name)

    async def _container_exists(self, name: str) -> bool:
        return name not in self.removed


def _repo_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    package = repo / "max_ai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='maxai'\n", encoding="utf-8")
    (repo / "README.md").write_text("readme\n", encoding="utf-8")
    return repo


def test_core_executor_stores_default_timeout() -> None:
    executor = LayoutExecutor(default_timeout=42)
    assert executor.default_timeout == 42


@pytest.mark.asyncio
async def test_bind_to_workspace_sets_workspace_root(tmp_path: Path) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))

    await executor.bind_to_workspace(tmp_path / "workspace")

    assert executor.workspace_root == (tmp_path / "workspace").resolve()


def test_host_runtime_uses_workspace_root_and_user_id(tmp_path: Path) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))
    executor.workspace_root = tmp_path / "workspace"

    assert executor._host_runtime("u1") == (tmp_path / "workspace" / "u1").resolve()


def test_ensure_runtime_layout_creates_user_dirs(tmp_path: Path) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))
    host_runtime = tmp_path / "workspace" / "u1"

    executor._ensure_runtime_layout(host_runtime)

    assert (host_runtime / "tools").is_dir()
    assert (host_runtime / "skills").is_dir()
    assert (host_runtime / "artifacts").is_dir()


def test_docker_run_once_command_uses_worker_and_mounts(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    workspace = tmp_path / "workspace"
    executor = DockerExecutor(repo_root=repo)
    executor.workspace_root = workspace
    host_runtime = workspace / "u1"

    command = executor._docker_run_once_command(host_runtime)

    assert command[:5] == ["docker", "run", "--rm", "-i", "--network"]
    assert f"{host_runtime}:/mnt" in command
    assert f"{workspace / '.docker-runtime' / 'app'}:/app:ro" in command
    assert "PYTHONPATH=/app" in command
    assert "MAX_AI_TOOL_SOURCE_DIR=/app/tools" in command
    assert command[-5:] == ["python", "-m", "max_ai.executor.docker.worker", "-", "-"]


def test_prepare_app_mount_stages_package_metadata_and_tool_files(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    tool_file = tmp_path / "docker_tools.py"
    tool_file.write_text("VALUE = 42\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    executor = DockerExecutor(repo_root=repo, tool_files=[tool_file])
    executor.workspace_root = workspace

    app_mount = executor._prepare_app_mount()

    assert app_mount == workspace / ".docker-runtime" / "app"
    assert (app_mount / "pyproject.toml").is_file()
    assert (app_mount / "README.md").is_file()
    assert (app_mount / "max_ai" / "__init__.py").is_file()
    assert (app_mount / "tools" / "docker_tools.py").read_text(encoding="utf-8") == "VALUE = 42\n"


def test_prepare_app_mount_accepts_single_tool_file_path(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    tool_file = tmp_path / "docker_tools.py"
    tool_file.write_text("VALUE = 42\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    executor = DockerExecutor(repo_root=repo, tool_files=str(tool_file))
    executor.workspace_root = workspace

    app_mount = executor._prepare_app_mount()

    assert executor.tool_files == [tool_file.resolve()]
    assert (app_mount / "tools" / "docker_tools.py").is_file()


def test_tool_result_from_stdout_uses_last_json_line() -> None:
    stdout = "\n".join(
        [
            "debug line",
            '{"success":true,"error":null,"result":5,"failure_reason":null,'
            '"tool_call_id":"call_1","metadata":{}}',
        ]
    )

    result = DockerExecutor._tool_result_from_stdout(stdout)

    assert result is not None
    assert result.success is True
    assert result.result == 5


def test_bash_container_name_is_stable_and_prefixed(tmp_path: Path) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))
    context = ToolContext(run_id="Run 1", session_id="Session 1", user_id="u1")

    assert executor._bash_container_name(context) == "maxai-bash-session-1"


def test_docker_executor_expands_read_skill_alias_to_container_path(tmp_path: Path) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))

    command = executor._expand_internal_bash_command("read_skill create-ppt")

    assert command == "cat '/mnt/skills/create-ppt/SKILL.md'"


@pytest.mark.asyncio
async def test_prune_expired_bash_sessions_removes_expired_containers(tmp_path: Path) -> None:
    executor = TrackingDockerExecutor(repo_root=_repo_fixture(tmp_path), bash_ttl_seconds=0)
    executor._bash_sessions["run_1"] = _BashSession(
        name="maxai-bash-run-1",
        last_used_at=time.monotonic() - 10,
    )

    await executor._prune_expired_bash_sessions()

    assert executor._bash_sessions == {}
    assert executor.removed == ["maxai-bash-run-1"]


@pytest.mark.asyncio
async def test_disconnect_closes_kept_alive_bash_sessions(tmp_path: Path) -> None:
    executor = TrackingDockerExecutor(repo_root=_repo_fixture(tmp_path))
    executor._bash_sessions["run_1"] = _BashSession(
        name="maxai-bash-run-1",
        last_used_at=time.monotonic(),
    )

    await executor.disconnect()

    assert executor._bash_sessions == {}
    assert executor.removed == ["maxai-bash-run-1"]
