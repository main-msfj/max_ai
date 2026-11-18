"""DockerSandbox: one container per user. The tests that start a container
are skipped without Docker; the ones that reach the internet are
``integration``."""

import asyncio
import shutil
import subprocess

import pytest

from max_ai.base.sandbox import SandboxCore
from max_ai.capabilities.sandbox.docker import DockerSandbox
from max_ai.errors.sandbox import SandboxError


def _docker_runs() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


needs_docker = pytest.mark.skipif(not _docker_runs(), reason="Docker is not running")
FETCH = "python3 -c \"import urllib.request as u; print(u.urlopen('{}', timeout=15).status)\""


@pytest.fixture
async def docker(request):
    """Bound sandboxes that are removed, with their files, after the test."""
    made: list[DockerSandbox] = []

    def build(**options) -> DockerSandbox:
        name = f"test-{request.node.name.replace('_', '-')[:40]}-{len(made)}"
        made.append(DockerSandbox(**options).bind(name, {"MY_VARIABLE": "x"}))
        return made[-1]

    yield build
    for sandbox in made:
        await sandbox.remove()


# -------- without Docker ---------------------------------------------------------
def test_the_container_is_named_after_the_identifier():
    assert DockerSandbox().bind("user-1").name == "docker-sandbox-user-1"


def test_an_identifier_cannot_be_a_path():
    with pytest.raises(ValueError):
        DockerSandbox().bind("../x").name


async def test_it_needs_a_user():
    with pytest.raises(SandboxError, match="bind"):
        await DockerSandbox().execute_command("true")


def test_it_is_stored_with_image_dockerfile_and_networks():
    sandbox = DockerSandbox(dockerfile="Dockerfile", networks=["pypi.org"])
    stored = sandbox.serialize()
    assert stored.config == {
        "dockerfile": "Dockerfile", "networks": ["pypi.org"],
        "memory": "2g", "cpus": 2.0, "processes": 512, "extra_env": [],
    }
    again = SandboxCore.deserialize(stored.model_dump())
    assert isinstance(again, DockerSandbox) and again.networks == ["pypi.org"]


# -------- with Docker ------------------------------------------------------------
@needs_docker
async def test_commands_run_in_the_users_container(docker):
    sandbox = docker()
    await sandbox.prepare()
    result = await sandbox.execute_command("pwd; echo $MY_VARIABLE; ls .agent")
    assert result.stdout.split() == [
        "/home/agent", "x", "memory", "setting.json", "skills", "tmp", "workspace",
    ]
    assert (await sandbox.execute_command("cat", stdin=b"hello")).stdout == "hello"
    assert (await sandbox.execute_command("exit 3")).exit_code == 3


@needs_docker
async def test_only_the_variables_named_in_extra_env_enter_the_container(docker, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "secret-value")
    monkeypatch.setenv("OTHER_KEY", "other-value")
    seen = (await docker(extra_env=["MY_API_KEY"]).execute_command("env")).stdout
    assert "MY_API_KEY=secret-value" in seen and "other-value" not in seen


@needs_docker
async def test_a_command_stops_at_its_timeout_and_leaves_nothing_running(docker):
    sandbox = docker()
    result = await sandbox.execute_command("sleep 300 & sleep 300", timeout=1)
    assert result.timed_out and result.exit_code is None
    left = await sandbox.execute_command(
        "sleep 1; grep -l sleep /proc/[0-9]*/comm | wc -l"
    )
    assert left.stdout.strip() == "1"  # only the container's own


@needs_docker
async def test_a_cancelled_command_leaves_nothing_running(docker):
    sandbox = docker()
    await sandbox.prepare()
    running = asyncio.create_task(sandbox.execute_command("sleep 300 & sleep 300", timeout=120))
    await asyncio.sleep(2)
    running.cancel()
    await asyncio.gather(running, return_exceptions=True)
    left = await sandbox.execute_command("grep -l sleep /proc/[0-9]*/comm | wc -l")
    assert running.cancelled() and left.stdout.strip() == "1"  # only the container's own


@needs_docker
async def test_files_and_skills_survive_the_container(docker, skill_archive):
    sandbox = docker()
    await sandbox.prepare()
    await sandbox.upload_file("s1", "a.bin", b"\x00\xff")
    await sandbox.install_skill("pdf", "k1", skill_archive({"SKILL.md": b"pdf"}))
    await sandbox.close()
    assert await sandbox.outdated_skills({"pdf": "k1"}) == []
    size = await sandbox.execute_command("wc -c < .agent/workspace/s1/a.bin")
    assert size.stdout.strip() == "2"


@needs_docker
async def test_remove_leaves_nothing_of_the_user(docker):
    sandbox = docker(networks=["pypi.org"])
    await sandbox.prepare()
    await sandbox.remove()
    for kind in ("container", "volume", "network"):
        listed = subprocess.run(
            ["docker", kind, "ls", "--format", "{{.Name}}{{.Names}}", *(["-a"] if kind == "container" else [])],
            capture_output=True, text=True,
        )
        assert sandbox.name not in listed.stdout, kind
    # The next command starts from nothing.
    assert (await sandbox.execute_command("ls -A")).stdout.strip() == ""


@needs_docker
async def test_without_networks_there_is_no_network(docker):
    result = await docker().execute_command(
        "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)\""
    )
    assert result.exit_code != 0


@needs_docker
async def test_the_image_is_built_from_a_dockerfile(docker, tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nRUN echo built > /marker\n")
    sandbox = docker(dockerfile=str(dockerfile))
    try:
        assert (await sandbox.execute_command("cat /marker")).stdout.strip() == "built"
    finally:
        await sandbox.close()
        subprocess.run(["docker", "rmi", "-f", await sandbox._image()], capture_output=True)


@needs_docker
@pytest.mark.integration
async def test_only_the_listed_domains_are_reached(docker):
    sandbox = docker(networks=["pypi.org"])
    assert (await sandbox.execute_command(FETCH.format("https://pypi.org"))).stdout.strip() == "200"
    assert (await sandbox.execute_command(FETCH.format("https://example.com"))).exit_code != 0
    direct = await sandbox.execute_command(
        "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)\""
    )
    assert direct.exit_code != 0


# -------- who the commands run as ------------------------------------------------
def _inspect(sandbox: DockerSandbox, what: str) -> str:
    asked = subprocess.run(
        ["docker", "inspect", "--format", what, sandbox.name], capture_output=True, text=True
    )
    return asked.stdout.strip()


@needs_docker
async def test_the_model_s_commands_have_no_privileges(docker):
    sandbox = docker()
    await sandbox.prepare()
    layout = sandbox.layout
    assert (await sandbox.execute_command("id -u")).stdout.strip() == "1000"

    # It writes where it works...
    for folder in (layout.workspace, layout.memory, layout.tmp, "/tmp"):
        wrote = await sandbox.execute_command(f"mkdir -p {folder}/s1 && echo x > {folder}/s1/a.txt")
        assert wrote.exit_code == 0, wrote.stderr
    # ...and nowhere the harness decides from, nor in the system.
    for command in (
        f"echo '{{}}' > {layout.settings}",
        f"rm {layout.settings}",
        f"mkdir {layout.skills}/mine",
        f"mv {layout.root} {layout.root}.old",
        "touch /usr/local/bin/python3",
        "echo x > /etc/own",
    ):
        refused = await sandbox.execute_command(command)
        assert refused.exit_code != 0, command
    assert (await sandbox.execute_command(f"cat {layout.settings}")).exit_code == 0  # read: yes


@needs_docker
async def test_only_the_harness_writes_the_settings_and_the_skills(docker, skill_archive):
    sandbox = docker()
    await sandbox.prepare()
    await sandbox.add_permission("allow", "bash(ls)")
    await sandbox.install_skill("pdf", "k1", skill_archive({"SKILL.md": b"# pdf"}))
    assert (await sandbox.permissions())["allow"] == ["bash(ls)"]
    # The model's user reads and runs the skill; it cannot change it.
    skill = sandbox.layout.skills / "pdf"
    assert (await sandbox.execute_command(f"cat {skill}/SKILL.md")).exit_code == 0
    assert (await sandbox.execute_command(f"echo x >> {skill}/SKILL.md")).exit_code != 0


@needs_docker
async def test_the_container_is_closed_and_has_limits(docker):
    sandbox = docker(memory="256m", cpus=1, processes=64)
    await sandbox.execute_command("true")
    assert _inspect(sandbox, "{{.Config.User}}") == "1000:1000"
    assert _inspect(sandbox, "{{.HostConfig.CapDrop}}") == "[ALL]"
    assert _inspect(sandbox, "{{.HostConfig.ReadonlyRootfs}}") == "true"
    assert "no-new-privileges" in _inspect(sandbox, "{{.HostConfig.SecurityOpt}}")
    assert _inspect(sandbox, "{{.HostConfig.Memory}}") == str(256 * 1024 * 1024)
    assert _inspect(sandbox, "{{.HostConfig.PidsLimit}}") == "64"
    # Too many processes are refused; the container and the next command go on.
    await sandbox.execute_command("for i in $(seq 200); do sleep 30 & done; wait", timeout=5)
    assert (await sandbox.execute_command("echo alive")).stdout.strip() == "alive"


@needs_docker
async def test_the_container_stops_by_itself_when_nobody_uses_it(docker, monkeypatch):
    from max_ai.config import setting

    monkeypatch.setattr(setting, "harness_sandbox_seconds", 10)
    sandbox = docker()
    await sandbox.prepare()
    await sandbox.upload_file("s1", "a.txt", b"kept")

    # 1. A command before the time is up starts the clock again.
    await asyncio.sleep(7)
    await sandbox.execute_command("true")
    await asyncio.sleep(7)
    assert _inspect(sandbox, "{{.State.Running}}") == "true"  # 14s old, 7s idle

    # 2. Nobody uses it: it stops and is removed.
    await asyncio.sleep(16)
    assert _inspect(sandbox, "{{.State.Running}}") == ""

    # 3. The next command creates it again, with the user's files.
    read = await sandbox.execute_command(f"cat {sandbox.layout.workspace}/s1/a.txt")
    assert read.stdout == "kept"
