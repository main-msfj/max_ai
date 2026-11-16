"""DockerSandbox: one container per user. The tests that start a container
are skipped without Docker; the ones that reach the internet are
``integration``."""

import shutil
import subprocess

import pytest

from max_ai.base.sandbox import SandboxCore, SandboxError
from max_ai.capabilities.sandbox.docker import DockerSandbox


def _docker_runs() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


needs_docker = pytest.mark.skipif(not _docker_runs(), reason="Docker is not running")
FETCH = "python3 -c \"import urllib.request as u; print(u.urlopen('{}', timeout=15).status)\""


@pytest.fixture
async def docker(request):
    """Bound sandboxes that are removed, with their volumes, after the test."""
    made: list[DockerSandbox] = []

    def build(**options) -> DockerSandbox:
        name = f"test-{request.node.name.replace('_', '-')[:40]}-{len(made)}"
        made.append(DockerSandbox(**options).bind(name, {"MY_VARIABLE": "x"}))
        return made[-1]

    yield build
    for sandbox in made:
        await sandbox.close()
        subprocess.run(["docker", "volume", "rm", "-f", sandbox.name], capture_output=True)


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
    assert stored.config == {"dockerfile": "Dockerfile", "networks": ["pypi.org"]}
    again = SandboxCore.deserialize(stored.model_dump())
    assert isinstance(again, DockerSandbox) and again.networks == ["pypi.org"]


# -------- with Docker ------------------------------------------------------------
@needs_docker
async def test_commands_run_in_the_users_container(docker):
    sandbox = docker()
    await sandbox.prepare()
    result = await sandbox.execute_command("pwd; echo $MY_VARIABLE; ls .agent")
    assert result.stdout.split() == ["/home/agent", "x", "setting.json", "skills", "tmp", "workspace"]
    assert (await sandbox.execute_command("cat", stdin=b"hello")).stdout == "hello"
    assert (await sandbox.execute_command("exit 3")).exit_code == 3


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
