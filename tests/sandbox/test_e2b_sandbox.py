"""E2BSandbox: one machine in E2B per user. The tests that create a machine
need ``E2B_API_KEY`` and are ``integration``: they cost money."""

import asyncio
import importlib.util
import os

import pytest
from dotenv import load_dotenv

from max_ai.base.sandbox import SandboxCore
from max_ai.capabilities.sandbox.e2b import E2BSandbox
from max_ai.errors.sandbox import SandboxError

load_dotenv()
needs_e2b = pytest.mark.skipif(
    not os.environ.get("E2B_API_KEY") or importlib.util.find_spec("e2b") is None,
    reason="E2B_API_KEY is not set or the e2b extra is not installed",
)
FETCH = "python3 -c \"import urllib.request as u; print(u.urlopen('{}', timeout=15).status)\""


@pytest.fixture
async def e2b(request):
    """Bound sandboxes whose machines are deleted after the test."""
    made: list[E2BSandbox] = []

    def build(**options) -> E2BSandbox:
        name = f"test-{request.node.name.replace('_', '-')[:40]}-{len(made)}"
        made.append(E2BSandbox(**options).bind(name, {"MY_VARIABLE": "x"}))
        return made[-1]

    yield build
    for sandbox in made:
        await sandbox.remove()


# -------- without E2B ------------------------------------------------------------
def test_an_identifier_cannot_be_a_path():
    with pytest.raises(ValueError):
        E2BSandbox().bind("../x").name


async def test_it_needs_a_user():
    with pytest.raises(SandboxError, match="bind"):
        await E2BSandbox().execute_command("true")


async def test_it_needs_the_key_and_says_which_variable(monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    with pytest.raises(SandboxError, match="NO_SUCH_KEY"):
        await E2BSandbox(api_key_env="NO_SUCH_KEY").bind("user-1").execute_command("true")


async def test_without_the_e2b_package_it_says_how_to_install_it(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "e2b_secret")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SandboxError, match=r"max_ai\[e2b\]"):
        await E2BSandbox().bind("user-1").execute_command("true")


def test_it_is_stored_with_the_name_of_the_key_never_the_key(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "e2b_secret")
    stored = E2BSandbox(networks=["pypi.org"]).serialize()
    assert stored.config == {
        "networks": ["pypi.org"], "api_key_env": "E2B_API_KEY", "extra_env": [],
    }
    assert "e2b_secret" not in stored.model_dump_json()
    again = SandboxCore.deserialize(stored.model_dump())
    assert isinstance(again, E2BSandbox) and again.networks == ["pypi.org"]


# -------- with E2B ---------------------------------------------------------------
@needs_e2b
@pytest.mark.integration
async def test_commands_run_in_the_users_machine(e2b, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "secret-value")
    monkeypatch.setenv("OTHER_KEY", "other-value")
    sandbox = e2b(extra_env=["MY_API_KEY"])
    await sandbox.prepare()
    result = await sandbox.execute_command("pwd; echo $MY_VARIABLE; ls .agent")
    assert result.stdout.split() == [
        "/home/agent", "x", "memory", "setting.json", "skills", "tasks", "tmp", "workspace",
    ]
    assert (await sandbox.execute_command("cat", stdin=b"hello")).stdout == "hello"
    assert (await sandbox.execute_command("exit 3")).exit_code == 3
    # The E2B key stays on this machine.
    seen = (await sandbox.execute_command("env")).stdout
    assert "E2B_API_KEY" not in seen and "other-value" not in seen
    # Only what extra_env names comes from this machine.
    assert "MY_API_KEY=secret-value" in seen


@needs_e2b
@pytest.mark.integration
async def test_the_model_s_commands_have_no_privileges(e2b, skill_archive):
    sandbox = e2b()
    await sandbox.prepare()
    layout = sandbox.layout
    await sandbox.add_permission("allow", "bash(ls)")
    await sandbox.install_skill("pdf", "k1", skill_archive({"SKILL.md": b"# pdf"}))
    assert (await sandbox.execute_command("id -un")).stdout.strip() == "agent"

    # It writes where it works...
    for folder in (layout.workspace, layout.memory, layout.tmp):
        wrote = await sandbox.execute_command(f"mkdir -p {folder}/s1 && echo x > {folder}/s1/a.txt")
        assert wrote.exit_code == 0, wrote.stderr
    # ...and nowhere the harness decides from, nor in the system.
    for command in (
        "sudo -n true",
        f"echo '{{}}' > {layout.settings}",
        f"rm {layout.settings}",
        f"echo x >> {layout.skills}/pdf/SKILL.md",
        f"mv {layout.root} {layout.root}.old",
        "touch /usr/bin/python3",
    ):
        refused = await sandbox.execute_command(command)
        assert refused.exit_code != 0, command
    assert (await sandbox.permissions())["allow"] == ["bash(ls)"]
    assert (await sandbox.execute_command(f"cat {layout.skills}/pdf/SKILL.md")).stdout == "# pdf"


@needs_e2b
@pytest.mark.integration
async def test_a_stopped_command_leaves_nothing_running(e2b):
    sandbox = e2b()
    count = "sleep 1; grep -l sleep /proc/[0-9]*/comm | wc -l"
    # 1. Stopped by its timeout.
    result = await sandbox.execute_command("sleep 300 & sleep 300", timeout=1)
    assert result.timed_out and result.exit_code is None
    assert (await sandbox.execute_command(count)).stdout.strip() == "0"
    # 2. Cancelled from outside.
    running = asyncio.create_task(sandbox.execute_command("sleep 300 & sleep 300", timeout=120))
    await asyncio.sleep(3)
    running.cancel()
    await asyncio.gather(running, return_exceptions=True)
    assert running.cancelled()
    assert (await sandbox.execute_command(count)).stdout.strip() == "0"


@needs_e2b
@pytest.mark.integration
async def test_files_survive_a_close_and_another_process_finds_the_machine(e2b):
    sandbox = e2b()
    await sandbox.prepare()
    await sandbox.upload_file("s1", "a.bin", b"\x00\xff")
    await sandbox.close()
    # A new object, as after a restart: it knows only the identifier.
    again = E2BSandbox().bind(sandbox.name)
    size = await again.execute_command("wc -c < .agent/workspace/s1/a.bin")
    assert size.stdout.strip() == "2"


@needs_e2b
@pytest.mark.integration
async def test_the_machine_pauses_by_itself_and_the_next_command_wakes_it(e2b, monkeypatch):
    from e2b import AsyncSandbox, SandboxQuery, SandboxState

    from max_ai.config import setting

    monkeypatch.setattr(setting, "harness_sandbox_seconds", 15)
    sandbox = e2b()
    await sandbox.prepare()
    await sandbox.upload_file("s1", "a.txt", b"kept")

    async def state() -> list[str]:
        query = SandboxQuery(metadata={"max_ai": sandbox.name}, state=[SandboxState.RUNNING])
        return [box.sandbox_id for box in await AsyncSandbox.list(query=query).next_items()]

    # 1. Nobody uses it: it stops running.
    assert await state()
    await asyncio.sleep(40)
    assert await state() == []
    # 2. The next command wakes it, with the user's files.
    read = await sandbox.execute_command(f"cat {sandbox.layout.workspace}/s1/a.txt")
    assert read.stdout == "kept"


@needs_e2b
@pytest.mark.integration
async def test_without_networks_there_is_no_network_and_with_them_only_those(e2b):
    closed = await e2b().execute_command(FETCH.format("https://pypi.org"), timeout=30)
    assert closed.exit_code != 0
    sandbox = e2b(networks=["pypi.org"])
    assert (await sandbox.execute_command(FETCH.format("https://pypi.org"))).stdout.strip() == "200"
    assert (await sandbox.execute_command(FETCH.format("https://example.com"), timeout=30)).exit_code != 0
