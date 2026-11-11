"""The global settings come from Settings and the project's .agent/ from
Layout; neither overwrites a file that already exists."""

import json

from max_ai.config import Settings
from max_ai.core.harness.root import Layout


def test_create_global_writes_the_global_settings(tmp_path):
    path = Settings(max_ai_home=tmp_path / "home").create_global()
    assert path == tmp_path / "home" / "setting.global.json"
    assert json.loads(path.read_text()) == {"skills": {}}
    path.write_text('{"skills": {"pdf": "x"}}')
    Settings(max_ai_home=tmp_path / "home").create_global()
    assert json.loads(path.read_text()) == {"skills": {"pdf": "x"}}


def test_create_project_builds_the_agent_folder(tmp_path):
    root = Layout(project=tmp_path).create_project()
    assert root == tmp_path / ".agent"
    assert (root / "skills").is_dir() and (root / "workspace").is_dir()
    assert json.loads((root / "setting.json").read_text()) == {}


def test_an_existing_project_setting_is_never_overwritten(tmp_path):
    layout = Layout(project=tmp_path)
    layout.create_project()
    layout.settings.write_text('{"mine": true}')
    layout.create_project()
    assert json.loads(layout.settings.read_text()) == {"mine": True}
