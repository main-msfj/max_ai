"""Provider configs live beside, and serialize independently from, their runtime code."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module", "implementation", "config", "values"),
    [
        ("memory.local", "LocalMemoryRegistry", "LocalMemoryRegistryConfig",
         {"user_id": "u", "session_id": "s", "base_path": "/tmp"}),
        ("memory.mongodb", "MongoDBMemoryRegistry", "MongoDBMemoryRegistryConfig",
         {"user_id": "u", "session_id": "s"}),
        ("knowledge.local", "LocalKnowledgeRegistry", "LocalKnowledgeRegistryConfig",
         {"name": "docs", "description": "Docs", "base_path": "/tmp"}),
        ("knowledge.mongodb", "MongoDBKnowledgeRegistry", "MongoDBKnowledgeRegistryConfig",
         {"name": "docs", "description": "Docs"}),
        ("skills.local", "LocalSkillRegistry", "LocalSkillRegistryConfig",
         {"source": "/tmp", "skills": []}),
        ("context.local", "LocalContextRegistry", "LocalContextRegistryConfig",
         {"user_id": "u", "session_id": "s", "base_path": "/tmp"}),
        ("context.sqlite", "SQLiteContextRegistry", "SQLiteContextRegistryConfig",
         {"user_id": "u", "session_id": "s", "base_path": "/tmp"}),
        ("executor.local", "LocalExecutor", "LocalExecutorConfig", {}),
        ("executor.docker", "DockerExecutor", "DockerExecutorConfig", {}),
        ("executor.modal", "ModalExecutor", "ModalExecutorConfig", {"image": "example:latest"}),
        ("workspace.local", "LocalWorkspace", "LocalWorkspaceConfig", {}),
    ],
)
def test_provider_config_is_in_model_module_and_round_trips(module, implementation, config, values):
    package = importlib.import_module(f"max_ai.capabilities.{module}")
    runtime_type = getattr(package, implementation)
    config_type = getattr(package, config)
    assert runtime_type.component_schema is config_type
    assert config_type.__module__.endswith("._model")
    instance = config_type(**values)
    assert config_type.model_validate_json(instance.model_dump_json()) == instance


def test_executor_components_restore_from_serialized_config():
    from max_ai.capabilities.executor.docker import DockerExecutor
    from max_ai.capabilities.executor.modal import ModalExecutor

    for original in (DockerExecutor(image="example:latest"),
                     ModalExecutor(image="example:latest")):
        restored = type(original).deserialize(original.serialize())
        assert restored._to_config() == original._to_config()


def test_sqlite_memory_config_is_serializable_without_loading_runtime():
    from max_ai.capabilities.memory.sqlite import SQLiteMemoryRegistryConfig

    config = SQLiteMemoryRegistryConfig(user_id="u", base_path="/tmp")
    assert SQLiteMemoryRegistryConfig.model_validate_json(config.model_dump_json()) == config


def test_modal_network_modes():
    import pytest

    from max_ai.capabilities.executor.modal import PACKAGE_REGISTRIES, ModalExecutor

    packages = ModalExecutor()
    assert packages._allowed_domains() == list(PACKAGE_REGISTRIES)
    extra = ModalExecutor(allow_list=["api.github.com"])
    assert extra._allowed_domains() == [*PACKAGE_REGISTRIES, "api.github.com"]
    assert "besides the package registries it can only reach: api.github.com" in extra.describe_environment()

    assert ModalExecutor(network="internet")._allowed_domains() is None
    only = ModalExecutor(network="internet", allow_list=["*.exa.ai"])
    assert only._allowed_domains() == ["*.exa.ai"]
    assert "It can only reach: *.exa.ai." in only.describe_environment()

    none = ModalExecutor(network="none")
    assert none._allowed_domains() is None and "no network access" in none.describe_environment()
    with pytest.raises(ValueError, match="allow_list"):
        ModalExecutor(network="none", allow_list=["pypi.org"])

    restored = ModalExecutor.deserialize(extra.serialize())
    assert restored.network == "packages" and restored.allow_list == ["api.github.com"]


def test_docker_network_is_internet_or_none():
    import pytest

    from max_ai.capabilities.executor.docker import DockerExecutor

    assert DockerExecutor().network == "none"
    assert "It has internet access." in DockerExecutor(network="internet").describe_environment()
    with pytest.raises(ValueError, match="one of internet, none"):
        DockerExecutor(network="packages")


def test_modal_image_choices():
    import pytest

    from max_ai.capabilities.executor.modal import ModalExecutor

    ours = ModalExecutor()
    assert "pip, uv or npm" in ours.describe_environment()
    assert "Node.js/npm" in ours.describe_environment()
    custom = ModalExecutor(image="python:3.12-slim")
    assert "with pip before running it" in custom.describe_environment()
    assert "Node.js" not in custom.describe_environment()
    with pytest.raises(ValueError, match="image or dockerfile"):
        ModalExecutor(image="python:3.12-slim", dockerfile="Dockerfile")

    pinned = ModalExecutor(dockerfile="my.Dockerfile", framework="maxai==0.1.0")
    restored = ModalExecutor.deserialize(pinned.serialize())
    assert (restored.dockerfile, restored.framework) == ("my.Dockerfile", "maxai==0.1.0")


async def test_docker_builds_its_image_once_and_adds_max_ai_to_yours(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from max_ai.capabilities.executor.docker import _executor as docker

    calls, built = [], set()

    async def fake_run(argv, *, stdin=None, **kwargs):
        calls.append((list(argv), stdin))
        if argv[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(exit_code=0 if argv[3] in built else 1, stderr="")
        built.add(argv[3])
        return SimpleNamespace(exit_code=0, stderr="")

    monkeypatch.setattr(docker, "run_process", fake_run)

    theirs = docker.DockerExecutor(image="python:3.12-slim", framework="maxai==0.1.0")
    tag = await theirs._ensure_image()
    layer = next(stdin for argv, stdin in calls if argv[:2] == ["docker", "build"])
    assert layer.startswith("FROM python:3.12-slim\n")
    assert '/opt/maxai/bin/pip install --no-cache-dir "maxai==0.1.0"' in layer
    assert "chown 1000:1000 /workspaces" in layer

    # Same recipe: no rebuild, neither in this executor nor in a new one.
    calls.clear()
    assert await theirs._ensure_image() == tag and calls == []
    assert await docker.DockerExecutor(image="python:3.12-slim", framework="maxai==0.1.0")._ensure_image() == tag
    assert not any(argv[:2] == ["docker", "build"] for argv, _ in calls)

    # Default: max_ai's runtime Dockerfile, max_ai from `framework`.
    calls.clear()
    await docker.DockerExecutor()._ensure_image()
    argv, stdin = next((a, s) for a, s in calls if a[:2] == ["docker", "build"])
    assert f"MAXAI={docker.DEFAULT_FRAMEWORK}" in argv and "ARG MAXAI=" in stdin

    # A Dockerfile is built with its folder as context, then gets the layer.
    (tmp_path / "Dockerfile").write_text("FROM debian:bookworm-slim\n")
    calls.clear()
    await docker.DockerExecutor(dockerfile=str(tmp_path / "Dockerfile"))._ensure_image()
    builds = [(a, s) for a, s in calls if a[:2] == ["docker", "build"]]
    assert builds[0][0][-3:] == ["-f", str(tmp_path / "Dockerfile"), str(tmp_path)]
    assert builds[1][1].startswith(f"FROM {builds[0][0][3]}\n")


async def test_docker_build_failure_says_what_the_image_needs(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from max_ai.capabilities.executor.docker import _executor as docker

    async def fake_run(argv, **kwargs):
        return SimpleNamespace(exit_code=1, stderr="python3: not found")

    monkeypatch.setattr(docker, "run_process", fake_run)
    with pytest.raises(RuntimeError, match="needs Linux, bash, git and python3"):
        await docker.DockerExecutor(image="alpine")._ensure_image()
