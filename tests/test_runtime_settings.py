"""Iteration limit and idle timeout: one place each, defaults from settings."""

from max_ai.capabilities.executor.local import LocalExecutor
from max_ai.capabilities.reasoning.react import ReactLoop
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.config import setting
from max_ai.core.environment.manager import EnvironmentManager


def test_the_loop_follows_the_setting_unless_it_has_its_own(monkeypatch):
    monkeypatch.setattr(setting, "max_loop_iterations", 42)
    stored = ReactLoop().serialize()
    assert "max_loop_iterations" not in stored.config  # a stored agent keeps following it
    assert ReactLoop.deserialize(stored).max_loop_iterations == 42
    assert ReactLoop(7).max_loop_iterations == 7
    assert ReactLoop.deserialize(ReactLoop(7).serialize()).max_loop_iterations == 7


def test_idle_timeout_comes_from_the_setting(monkeypatch, tmp_path):
    monkeypatch.setattr(setting, "environment_idle_timeout", 12.5)
    manager = EnvironmentManager(LocalExecutor(), LocalWorkspace(root=tmp_path))
    assert manager.idle_timeout == 12.5
