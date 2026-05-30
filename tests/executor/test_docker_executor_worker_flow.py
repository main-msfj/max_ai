from __future__ import annotations

from pathlib import Path

from max_ai.base.tools import ToolContext
from max_ai.executor.docker import DockerExecutor
from max_ai.executor.docker import worker
from max_ai.types.tools import DockerToolRef


def _repo_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    package = repo / "max_ai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='maxai'\n", encoding="utf-8")
    (repo / "README.md").write_text("readme\n", encoding="utf-8")
    return repo


def test_docker_executor_session_key_prefers_session_id(tmp_path: Path) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))
    context = ToolContext(run_id="run_1", user_id="user_1", session_id="session_1")

    assert executor._bash_session_key(context) == "session_1"


def test_docker_executor_session_key_falls_back_to_run_id(tmp_path: Path) -> None:
    executor = DockerExecutor(repo_root=_repo_fixture(tmp_path))
    context = ToolContext(run_id="run_1", user_id="user_1")

    assert executor._bash_session_key(context) == "run_1"


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
