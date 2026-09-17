import asyncio

import pytest

from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.runtime.docker import DockerExecutor
from max_ai.runtime.local import LocalExecutor
from max_ai.runtime.modal import ModalExecutor
from max_ai.config import setting


@pytest.mark.asyncio
async def test_skill_materialization_preserves_workspace_edits(tmp_path, workspace, monkeypatch):
    monkeypatch.setattr(setting, "root_dir", tmp_path / "settings-root")
    source = tmp_path / "source" / "demo"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("---\nname: Demo\ndescription: test\n---\nbody")
    skills = LocalSkillRegistry(tmp_path / "source", ["demo"])
    await skills.prepare()
    directory = workspace.materialize("u", "c")
    skills.materialize(directory)
    (directory.skill_dir / "demo" / "edited.txt").write_text("keep")
    skills.materialize(directory)
    assert (directory.skill_dir / "demo" / "edited.txt").read_text() == "keep"
    assert (await skills.get_skills())[0].name == "Demo"


def test_skill_registry_validates_selection_and_requires_prepare(tmp_path, workspace, monkeypatch):
    monkeypatch.setattr(setting, "root_dir", tmp_path / "settings-root")
    with pytest.raises(ValueError):
        LocalSkillRegistry(tmp_path, ["bad/name"])
    skills = LocalSkillRegistry(tmp_path, ["ok"])
    with pytest.raises(RuntimeError):
        skills.materialize(workspace.materialize("u", "c"))


def test_local_executor_configuration_and_command_validation():
    executor = LocalExecutor(max_output_bytes=100)
    assert executor.max_output_bytes == 100
    with pytest.raises(ValueError):
        LocalExecutor(max_output_bytes=0)


@pytest.mark.asyncio
async def test_local_executor_rejects_empty_commands(workspace):
    executor = LocalExecutor()
    session = await executor.connect(workspace, "u", "c")
    with pytest.raises(ValueError):
        await asyncio.wait_for(executor.execute(session, " ", timeout=2), timeout=5)
    await executor.clean(session)


@pytest.mark.asyncio
async def test_local_executor_returns_bounded_output_and_workspace_env(workspace):
    executor = LocalExecutor(max_output_bytes=4)
    session = await executor.connect(workspace, "u", "c")
    result = await asyncio.wait_for(
        executor.execute(session, "printf 123456", timeout=2), timeout=5,
    )
    assert result.stdout == "1234"
    assert result.truncated
    path_executor = LocalExecutor(max_output_bytes=4096)
    path_session = await path_executor.connect(workspace, "u", "path")
    result = await asyncio.wait_for(
        path_executor.execute(path_session, "printf $WORKSPACE", timeout=2), timeout=5,
    )
    assert result.stdout == path_session.workspace_path
    await executor.clean(session)
    await path_executor.clean(path_session)


@pytest.mark.parametrize("factory", [DockerExecutor, lambda: ModalExecutor(image="img")])
def test_remote_executor_configuration_contract(factory):
    executor = factory()
    assert executor.network == "none"
    with pytest.raises(ValueError):
        DockerExecutor(network="bad")
    with pytest.raises(ValueError):
        ModalExecutor(image="img", network="bad")


@pytest.mark.asyncio
async def test_docker_start_contract_uses_copy_mount_and_nonroot_check(tmp_path, monkeypatch):
    from max_ai.base.workspace import Workspace
    from max_ai.base.execution_workspace import ExecutionWorkspace
    from max_ai.base.environment import ExecutionResult
    import max_ai.runtime.docker as docker_module

    calls = []
    async def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return ExecutionResult("", "", 0)
    monkeypatch.setattr(docker_module, "run_process", fake_run)
    executor = DockerExecutor(image="custom", network="unrestricted")
    persistent = Workspace(tmp_path / "agents")
    copy = ExecutionWorkspace(persistent, "u", "c")
    try:
        session = await executor.connect(copy.workspace, "u", "c")
        start_argv = calls[0][0]
        assert "--network=bridge" in start_argv
        assert "--read-only" in start_argv
        assert any(arg.startswith("type=bind,src=") and ",dst=/workspaces/u" in arg
                   for arg in start_argv)
        assert f"src={copy.root}" in next(arg for arg in start_argv if arg.startswith("type=bind"))
        assert str(persistent.base_root) not in next(arg for arg in start_argv if arg.startswith("type=bind"))
        assert "test \"$(id -u)\" -ne 0" in calls[1][0][-1]
    finally:
        if "session" in locals():
            await executor.clean(session)
        copy.discard()


def test_modal_contract_stores_lifetime_and_image_without_connecting():
    executor = ModalExecutor(image="custom", app_name="app", lifetime=10, max_output_bytes=20)
    assert (executor.image, executor.app_name, executor.lifetime) == ("custom", "app", 10)


@pytest.mark.parametrize("kwargs", [{"lifetime": 0}, {"lifetime": 86401}, {"max_output_bytes": 0}])
def test_modal_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        ModalExecutor(image="img", **kwargs)


def test_docker_rejects_invalid_output_limit():
    with pytest.raises(ValueError):
        DockerExecutor(max_output_bytes=0)
