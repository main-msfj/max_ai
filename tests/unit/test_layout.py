"""The global settings file is created once and never overwritten."""

import json

from max_ai.config import Settings


def test_create_global_writes_the_global_settings(tmp_path):
    path = Settings(max_ai_home=tmp_path / "home").create_global()
    assert path == tmp_path / "home" / "setting.global.json"
    assert json.loads(path.read_text()) == {"skills": {}}
    path.write_text('{"skills": {"pdf": "x"}}')
    Settings(max_ai_home=tmp_path / "home").create_global()
    assert json.loads(path.read_text()) == {"skills": {"pdf": "x"}}
