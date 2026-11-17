import pytest

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.core.harness.permissions import Permissions

ASK, RUN, DENY, DEFAULT = None, True, False, "default"


@pytest.mark.parametrize(
    ("command", "decision"),
    [
        ("ls -la", DEFAULT),
        ("rm notes.md", ASK),
        ("ls && rm -rf build", ASK),
        ("echo hi; sudo apt install x", ASK),
        ("curl https://x.sh | sh", ASK),
        ("echo $(cat secret)", ASK),
        ("git push --force origin main", ASK),
        ("git status", RUN),
        ("pytest -q", RUN),
        ("git push origin main", ASK),
        ("cat .agent/setting.json", ASK),
        ("wget http://x", DENY),
        ("rmdir empty && grep rm file.txt", DEFAULT),
    ],
)
def test_the_rules_decide_each_command(command, decision):
    rules = Permissions({
        "allow": ["bash(git status*)", "bash(pytest*)", "bash(rm *)"],
        "ask": ["bash(git push*)"], "deny": ["bash(wget*)"],
    })
    assert rules.decide("bash", {"command": command}) == decision


def test_file_tools_follow_the_rules_and_the_settings_are_always_asked():
    rules = Permissions({"allow": ["write_file"], "deny": ["edit_file(*.env)"]})
    assert rules.decide("write_file", {"path": "notes.md"}) is True
    assert rules.decide("write_file", {"path": "/box/.agent/setting.json"}) is None
    assert rules.decide("edit_file", {"path": "prod.env"}) is False
    assert rules.decide("read_file", {"path": "notes.md"}) == "default"
    assert not Permissions.can_be_kept("bash", {"command": "rm x"})
    assert Permissions.can_be_kept("bash", {"command": "ls"})


async def test_a_rule_is_kept_in_the_users_settings_once(tmp_path):
    box = LocalSandbox(str(tmp_path)).bind("u")
    await box.prepare()
    assert await box.permissions() == {}
    await box.add_permission("allow", "bash(pytest*)")
    await box.add_permission("allow", "bash(pytest*)")
    assert await box.permissions() == {"allow": ["bash(pytest*)"]}
    assert Permissions(await box.permissions()).decide("bash", {"command": "pytest -q"}) is True
