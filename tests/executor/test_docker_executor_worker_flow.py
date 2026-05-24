from __future__ import annotations

from pathlib import Path

from max_ai.base.tools import ToolContext
from max_ai.executor.docker import DockerExecutor
from max_ai.executor.docker import worker
from max_ai.tools.function_as_tool import FunctionAsTool
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import DockerToolRef


def add(a: int, b: int) -> int:
    return a + b


def _repo_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    package = repo / "max_ai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='maxai'\n", encoding="utf-8")
    (repo / "README.md").write_text("readme\n", encoding="utf-8")
    return repo


def test_docker_executor_builds_worker_payload(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    executor = DockerExecutor(repo_root=repo)
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
    tool = FunctionAsTool(add, name="add")

    payload = executor._invocation_payload(tool, record, context)

    assert payload["tool_ref"]["module"] == __name__
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
            "runtime_root": "/mnt",
            "tools_dir": "/mnt/tools",
            "skills_dir": "/mnt/skills",
            "artifacts_dir": "/mnt/artifacts",
        },
    }
    assert "tool_sources" not in payload


def test_docker_executor_command_runs_worker_with_container_paths(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    workspace = tmp_path / "workspace"
    tool_source = tmp_path / "docker_tools.py"
    tool_source.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    executor = DockerExecutor(repo_root=repo, tool_files=[tool_source])
    executor.workspace_root = workspace

    context = ToolContext(run_id="run_1", user_id="user_1")
    host_runtime = executor._host_runtime(context.user_id)
    command = executor._docker_run_once_command(host_runtime)

    assert command[:3] == ["docker", "run", "--rm"]
    assert f"{host_runtime}:/mnt" in command
    assert f"{workspace / '.docker-runtime' / 'app'}:/app:ro" in command
    assert "PYTHONPATH=/app" in command
    assert "MAX_AI_TOOL_SOURCE_DIR=/app/tools" in command
    assert command[-5:] == ["python", "-m", "max_ai.executor.docker.worker", "-", "-"]


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


def test_docker_executor_stages_file_tool_source(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    workspace = tmp_path / "workspace"
    tool_source = tmp_path / "docker_tools.py"
    tool_source.write_text("VALUE = 42\n", encoding="utf-8")
    executor = DockerExecutor(repo_root=repo, tool_files=[tool_source])
    executor.workspace_root = workspace

    app_mount = executor._prepare_app_mount()

    copied = app_mount / "tools" / "docker_tools.py"
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
