from __future__ import annotations

from pathlib import Path

from max_ai.base.tools import ToolContext
from max_ai.executor.docker import DockerExecutor
from max_ai.executor.docker import worker
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import DockerToolRef


def test_docker_executor_builds_worker_payload(tmp_path: Path) -> None:
    tool_source = tmp_path / "docker_tools.py"
    tool_source.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    executor = DockerExecutor(
        tool_source=tool_source,
        server_workspace=tmp_path / "server",
    )
    record = ToolCallRecord(
        tool_name="add",
        parameters={"a": 2, "b": 3},
    )
    context = ToolContext(
        run_id="run_1",
        user_id="user_1",
        session_id="session_1",
        retry_count=1,
        deps={"x": "y"},
    )
    tool_ref = DockerToolRef(
        kind="function",
        module="docker_tools",
        qualname="add",
        options={"name": "add"},
    )

    payload = executor._invocation_payload(tool_ref, record, context)

    assert payload["tool_ref"]["module"] == "docker_tools"
    assert payload["tool_ref"]["qualname"] == "add"
    assert payload["record"]["id"] == record.id
    assert payload["record"]["parameters"] == {"a": 2, "b": 3}
    assert payload["context"] == {
        "run_id": "run_1",
        "user_id": "user_1",
        "session_id": "session_1",
        "retry_count": 1,
        "deps": {
            "x": "y",
            "runtime_root": "/sandbox",
            "tools_dir": "/sandbox/tools",
            "skills_dir": "/sandbox/skills",
            "artifacts_dir": "/sandbox/artifacts",
        },
    }
    assert payload["tool_sources"] == {
        "docker_tools.py": "def add(a: int, b: int) -> int:\n    return a + b\n"
    }


def test_docker_executor_command_runs_worker_with_container_paths(tmp_path: Path) -> None:
    server_workspace = tmp_path / "server"
    tool_source = tmp_path / "docker_tools.py"
    tool_source.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

    executor = DockerExecutor(
        tool_source=tool_source,
        server_workspace=server_workspace,
    )

    context = ToolContext(run_id="run_1", user_id="user_1")
    command = executor._docker_command(context)
    env = executor._compose_env(context)

    assert command[:2] == ["docker", "compose"]
    assert "-f" in command
    assert "-p" in command
    assert f"{tool_source.resolve()}:/sandbox/docker_tools.py:ro" not in command
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
    assert env["MAXAI_HOST_RUNTIME_DIR"] == str(
        server_workspace.resolve() / "tmp" / "user_1"
    )
    assert env["MAXAI_CONTAINER_WORKSPACE"] == "/sandbox"
    assert "MAXAI_SOURCE_DIR" not in env


def test_docker_executor_parses_last_tool_result_from_stdout() -> None:
    stdout = "\n".join(
        [
            "debug line from a tool",
            '{"success":true,"error":null,"result":5,"failure_reason":null,'
            '"tool_call_id":"call_1","metadata":{}}',
        ]
    )

    result = DockerExecutor._tool_result_from_stdout(stdout)

    assert result is not None
    assert result.success is True
    assert result.result == 5


def test_docker_executor_syncs_file_tool_source_to_workspace(tmp_path: Path) -> None:
    server_workspace = tmp_path / "server"
    tool_source = tmp_path / "docker_tools.py"
    tool_source.write_text("VALUE = 42\n", encoding="utf-8")
    executor = DockerExecutor(
        tool_source=tool_source,
        server_workspace=server_workspace,
    )

    executor._sync_tool_source("user_1")

    copied = server_workspace / "tmp" / "user_1" / "tools" / "docker_tools.py"
    assert copied.read_text(encoding="utf-8") == "VALUE = 42\n"


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
