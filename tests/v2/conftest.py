from pathlib import Path

import pytest

from max_ai.base.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path):
    return Workspace(tmp_path / "agents")


@pytest.fixture
def directory(workspace):
    return workspace.materialize("alice", "chat-1")
