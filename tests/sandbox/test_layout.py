"""Layout only knows the shape of ``.agent/``; it never touches the disk."""

from max_ai.base.layout import SCRIPT, Layout


def test_everything_hangs_from_agent_inside_the_base():
    layout = Layout("/home/agent")
    assert str(layout.root) == "/home/agent/.agent"
    assert str(layout.settings) == "/home/agent/.agent/setting.json"
    assert str(layout.skills) == "/home/agent/.agent/skills"
    assert str(layout.workspace) == "/home/agent/.agent/workspace"
    assert str(layout.tmp) == "/home/agent/.agent/tmp"
    assert str(layout.lock) == "/home/agent/.agent/.lock"


def test_building_a_layout_creates_nothing(tmp_path):
    Layout(str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_the_skeleton_names_the_folders_and_the_first_settings():
    skeleton = Layout("/x").skeleton()
    assert skeleton["folders"] == [
        "/x/.agent/skills", "/x/.agent/workspace", "/x/.agent/tmp", "/x/.agent/memory",
    ]
    assert skeleton["settings"] == "/x/.agent/setting.json"
    assert skeleton["content"] == {"skills": {}}


def test_the_script_uses_only_the_standard_library():
    compile(SCRIPT, "script.py", "exec")
    imports = {line.split()[1] for line in SCRIPT.splitlines() if line.startswith("import ")}
    assert imports <= {
        "contextlib", "fcntl", "hashlib", "json", "os", "shutil", "sys", "tarfile", "tempfile", "time",
    }
