from __future__ import annotations

import pytest

from pathlib import Path

from max_ai.base.executor import CoreExecutor
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.executor.docker.docker import DockerExecutor, _ComposeSession
from max_ai.tools import BashTool
from max_ai.termination import CancellationToken
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode


class LayoutExecutor(CoreExecutor):
    async def run(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        return ToolResult.success_result(record.id, {})


class TrackingDockerExecutor(DockerExecutor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.down_projects: list[str] = []
        self.compose_projects: list[str] = []

    async def _compose_down_project(self, project_name: str, env: dict[str, str]) -> None:
        self.down_projects.append(project_name)

    async def _list_compose_projects(self, env: dict[str, str]) -> list[str]:
        return list(self.compose_projects)


def test_core_executor_stores_default_timeout() -> None:
    executor = LayoutExecutor(default_timeout=42)
    assert executor.default_timeout == 42


def test_docker_command_uses_worker_and_user_paths(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)
    context = ToolContext(run_id="run_1", user_id="u1")

    command = executor._docker_command(context)
    env = executor._compose_env(context)

    assert command[:2] == ["docker", "compose"]
    assert "-f" in command
    assert "-p" in command
    assert command[-8:] == [
        "exec",
        "-T",
        "runtime",
        "python",
        "-m",
        "max_ai.executor.docker.worker",
        "-",
        "-",
    ]
    assert env["MAXAI_RUNTIME_IMAGE"] == "maxai-sandbox:py311"
    assert env["MAXAI_HOST_RUNTIME_DIR"] == str(tmp_path.resolve() / "tmp" / "u1")
    assert env["MAXAI_CONTAINER_WORKSPACE"] == "/sandbox"
    assert "MAXAI_SOURCE_DIR" not in env


def test_bash_command_executes_directly_in_sandbox(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)
    context = ToolContext(run_id="run_1", user_id="u1")

    command = executor._bash_command(context, "ls skills/")

    assert command[-6:] == ["exec", "-T", "runtime", "bash", "-lc", "ls skills/"]


def test_compose_mounts_only_runtime_sandbox() -> None:
    compose = (
        Path("max_ai")
        / "executor"
        / "docker"
        / "docker-compose.yml"
    ).read_text(encoding="utf-8")

    assert "${MAXAI_HOST_RUNTIME_DIR}:${MAXAI_CONTAINER_WORKSPACE:-/sandbox}" in compose
    assert "maxai-src" not in compose
    assert "PYTHONPATH" not in compose


def test_docker_build_uses_local_sandbox_dockerfile(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)

    assert executor._dockerfile_path().name == "Dockerfile.sandbox"
    assert executor._dockerfile_path().parent.name == "docker"


def test_compose_up_force_recreates_runtime(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)
    context = ToolContext(run_id="run_1", user_id="u1")
    command = executor._compose_up_command(context)

    assert command[-4:] == ["up", "-d", "--force-recreate", "runtime"]


def test_get_or_create_tmp_dir_creates_user_layout(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)
    root = executor.get_or_create_tmp_dir("u1")

    assert root == tmp_path.resolve() / "tmp" / "u1"
    assert (root / "skills").is_dir()
    assert (root / "artifacts").is_dir()
    assert (root / "tools").is_dir()


def test_sync_tool_source_to_user_workspace(tmp_path: Path) -> None:
    tool_source = tmp_path / "docker_tools.py"
    tool_source.write_text("VALUE = 42\n", encoding="utf-8")
    server_workspace = tmp_path / "server"
    executor = DockerExecutor(tool_source=tool_source, server_workspace=server_workspace)

    executor._sync_tool_source("u1")

    copied = server_workspace.resolve() / "tmp" / "u1" / "tools" / "docker_tools.py"
    assert copied.read_text(encoding="utf-8") == "VALUE = 42\n"


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


def test_bash_tool_keeps_compose_alive(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)

    assert executor._keeps_compose_alive(
        BashTool(approval_mode=ToolApprovalMode.AUTO_APPROVED)
    )
    assert not executor._keeps_compose_alive(LayoutExecutorTool())


def test_remember_persistent_session_only_for_bash(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)
    context = ToolContext(run_id="run_1", user_id="u1")
    env = executor._compose_env(context)

    executor._remember_persistent_session_if_needed(LayoutExecutorTool(), context, env)
    assert executor._compose_sessions == {}

    executor._remember_persistent_session_if_needed(
        BashTool(approval_mode=ToolApprovalMode.AUTO_APPROVED),
        context,
        env,
    )
    assert set(executor._compose_sessions) == {"maxai_run_1"}


@pytest.mark.asyncio
async def test_prune_expired_bash_sessions(tmp_path: Path) -> None:
    executor = TrackingDockerExecutor(server_workspace=tmp_path, bash_ttl_seconds=0)
    context = ToolContext(run_id="run_1", user_id="u1")
    env = executor._compose_env(context)
    executor._remember_persistent_session_if_needed(
        BashTool(approval_mode=ToolApprovalMode.AUTO_APPROVED),
        context,
        env,
    )

    await executor._prune_expired_sessions()

    assert executor._compose_sessions == {}
    assert executor.down_projects == ["maxai_run_1"]


@pytest.mark.asyncio
async def test_disconnect_closes_kept_alive_sessions(tmp_path: Path) -> None:
    executor = TrackingDockerExecutor(server_workspace=tmp_path)
    context = ToolContext(run_id="run_1", user_id="u1")
    env = executor._compose_env(context)
    executor._remember_persistent_session_if_needed(
        BashTool(approval_mode=ToolApprovalMode.AUTO_APPROVED),
        context,
        env,
    )

    await executor.disconnect()

    assert executor._compose_sessions == {}
    assert executor.down_projects == ["maxai_run_1"]


@pytest.mark.asyncio
async def test_prune_orphaned_compose_projects_skips_current_and_tracked(
    tmp_path: Path,
) -> None:
    executor = TrackingDockerExecutor(server_workspace=tmp_path)
    context = ToolContext(run_id="run_2", user_id="u1")
    env = executor._compose_env(context)
    executor.compose_projects = [
        "maxai_old",
        "maxai_run_1",
        "maxai_run_2",
        "other_project",
    ]
    executor._compose_sessions["maxai_run_1"] = _ComposeSession(
        env=env,
        last_used_at=0,
    )

    await executor._prune_orphaned_projects(context, env)

    assert executor.down_projects == ["maxai_old"]


@pytest.mark.asyncio
async def test_prune_orphaned_compose_projects_can_be_disabled(tmp_path: Path) -> None:
    executor = TrackingDockerExecutor(
        server_workspace=tmp_path,
        cleanup_orphaned_projects=False,
    )
    context = ToolContext(run_id="run_1", user_id="u1")
    env = executor._compose_env(context)
    executor.compose_projects = ["maxai_old"]

    await executor._prune_orphaned_projects(context, env)

    assert executor.down_projects == []


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
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        return ToolResult.success_result(tool_request.id, {})
