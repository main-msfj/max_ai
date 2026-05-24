from __future__ import annotations

from pathlib import Path

from max_ai.executor.docker import DockerExecutor


def test_docker_executor_stages_minimal_app_and_tool_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    package = repo / "max_ai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='maxai'\n", encoding="utf-8")
    (repo / "README.md").write_text("readme\n", encoding="utf-8")

    tool_file = tmp_path / "custom_tools.py"
    tool_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")

    workspace = tmp_path / "workspace"
    executor = DockerExecutor(repo_root=repo, tool_files=[tool_file])
    executor.workspace_root = workspace

    app_mount = executor._prepare_app_mount()

    assert app_mount == workspace / ".docker-runtime" / "app"
    assert (app_mount / "pyproject.toml").is_file()
    assert (app_mount / "README.md").is_file()
    assert (app_mount / "max_ai" / "__init__.py").is_file()
    assert (app_mount / "tools" / "custom_tools.py").read_text(encoding="utf-8") == (
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )


def test_docker_executor_sets_tool_source_dir_env() -> None:
    env_args = DockerExecutor()._docker_env_args()

    assert "MAX_AI_TOOL_SOURCE_DIR=/app/tools" in env_args
