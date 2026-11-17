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


async def test_a_rule_is_kept_once_and_can_be_taken_out(tmp_path):
    box = LocalSandbox(str(tmp_path)).bind("u")
    await box.prepare()
    assert await box.permissions() == {}
    await box.add_permission("allow", "bash(pytest*)")
    await box.add_permission("allow", "bash(pytest*)")
    assert await box.permissions() == {"allow": ["bash(pytest*)"]}
    assert Permissions(await box.permissions()).decide("bash", {"command": "pytest -q"}) is True
    await box.remove_permission("allow", "bash(pytest*)")
    await box.remove_permission("deny", "bash(rm*)")  # not there: nothing changes
    assert await box.permissions() == {"allow": []}


def test_always_keeps_one_rule_per_command_and_a_lone_cd_needs_none():
    assert Permissions.rules_for("bash", {"command": "pytest -q tests/"}) == ["bash(pytest*)"]
    assert Permissions.rules_for("write_file", {"path": "notes.md"}) == ["write_file"]
    command = "cd /box/s && cat >> a.css <<'EOF'\n.qr { x: 1; }\nEOF\nsleep 1; curl -s x | grep -n qr"
    rules = Permissions.rules_for("bash", {"command": command})
    assert rules == ["bash(cat*)", "bash(sleep*)", "bash(curl*)", "bash(grep*)"]  # no cd, no CSS
    kept = Permissions({"allow": rules})
    assert kept.decide("bash", {"command": command}) is True
    assert kept.decide("bash", {"command": "cd /box/s && pytest"}) == "default"  # pytest not allowed
    assert kept.decide("bash", {"command": "cd /box/s"}) is True
    assert not Permissions.can_be_kept("bash", {"command": "cd /box/s"})  # nothing to keep
    assert kept.decide("bash", {"command": "cat a && rm x"}) is None  # still asked


def test_separators_inside_quotes_do_not_split_the_command():
    command = "python3 -c \"\nimport csv\nprint('a; b | c')\n\" && head -5 out.csv | wc -l"
    rules = Permissions.rules_for("bash", {"command": command})
    assert rules == ["bash(python3*)", "bash(head*)", "bash(wc*)"]
