from __future__ import annotations

from pathlib import Path

from max_ai.base.executor import CoreExecutor
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.executor.docker.docker import DockerExecutor
from max_ai.termination import CancellationToken
from max_ai.types.tool_call import ToolCallRecord, ToolResult


class LayoutExecutor(CoreExecutor):
    async def run(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        return ToolResult.success_result(record.id, {})


def test_core_executor_stores_server_workspace(tmp_path: Path) -> None:
    executor = LayoutExecutor(server_workspace=tmp_path)
    assert executor.server_workspace == tmp_path.resolve()


def test_docker_command_uses_worker_and_user_paths(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)
    context = ToolContext(run_id="run_1", user_id="u1")

    command = executor._docker_command(context)

    assert command[:4] == ["docker", "run", "--rm", "-i"]
    assert f"{tmp_path.resolve()}:/sandbox" in command
    assert "MAX_AI_TOOL_SOURCE_DIR=/sandbox/tmp/u1/tools" in command
    assert "SKILLS_DIR=/sandbox/tmp/u1/skills" in command
    assert "WORKSPACE_DIR=/sandbox/tmp/u1/workspace" in command
    assert command[-5:] == [
        "python",
        "-m",
        "max_ai.executor.docker.worker",
        "-",
        "-",
    ]


def test_get_or_create_tmp_dir_creates_user_layout(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)
    root = executor.get_or_create_tmp_dir("u1")

    assert root == tmp_path.resolve() / "tmp" / "u1"
    assert (root / "skills").is_dir()
    assert (root / "workspace").is_dir()
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
