"""Provider configs live beside, and serialize independently from, their runtime code."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module", "implementation", "config", "values"),
    [
        ("knowledge.local", "LocalKnowledgeRegistry", "LocalKnowledgeRegistryConfig",
         {"name": "docs", "description": "Docs", "base_path": "/tmp"}),
        ("knowledge.mongodb", "MongoDBKnowledgeRegistry", "MongoDBKnowledgeRegistryConfig",
         {"name": "docs", "description": "Docs"}),
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
