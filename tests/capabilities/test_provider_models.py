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
        ("tools.agent_as_tool", "AgentAsTool", "AgentAsToolConfig",
         {"agent": {"provider": "example.Agent", "config": {}}}),
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
