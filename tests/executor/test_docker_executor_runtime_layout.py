from __future__ import annotations

from pathlib import Path

from max_ai.executor.docker import DockerExecutor


def test_docker_executor_creates_runtime_layout(tmp_path: Path) -> None:
    host_runtime = tmp_path / "workspace" / "user_1"
    executor = DockerExecutor(repo_root=tmp_path)

    executor._ensure_runtime_layout(host_runtime)

    assert (host_runtime / "tools").is_dir()
    assert (host_runtime / "skills").is_dir()
    assert (host_runtime / "artifacts").is_dir()


def test_docker_executor_sets_runtime_env() -> None:
    env_args = DockerExecutor()._docker_env_args()

    assert "RUNTIME_DIR=/mnt" in env_args
    assert "TOOLS_DIR=/mnt/tools" in env_args
    assert "SKILLS_DIR=/mnt/skills" in env_args
    assert "ARTIFACTS_DIR=/mnt/artifacts" in env_args
    assert "WORKSPACE_DIR=/mnt/artifacts" in env_args
