"""LocalSandbox: this computer is the sandbox."""

import asyncio

from max_ai.base.sandbox import SandboxCore
from max_ai.capabilities.sandbox.local import LocalSandbox


def test_building_it_creates_nothing(tmp_path):
    LocalSandbox(root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_it_is_stored_with_the_root_as_given():
    sandbox = LocalSandbox(root=".")
    assert sandbox.serialize().config == {"root": ".", "extra_env": []}
    again = SandboxCore.deserialize(sandbox.serialize().model_dump())
    assert isinstance(again, LocalSandbox) and again.layout.root == sandbox.layout.root


async def test_a_command_starts_in_the_root_and_reports_its_exit_code(tmp_path):
    sandbox = LocalSandbox(root=str(tmp_path))
    assert (await sandbox.execute_command("pwd")).stdout.strip() == str(tmp_path)
    assert (await sandbox.execute_command("exit 3")).exit_code == 3


async def test_a_command_reads_stdin_and_sees_the_bound_variables(tmp_path):
    sandbox = LocalSandbox(root=str(tmp_path)).bind("user-1", {"MY_VARIABLE": "x"})
    assert (await sandbox.execute_command("cat", stdin=b"hello")).stdout == "hello"
    assert (await sandbox.execute_command("echo $MY_VARIABLE")).stdout.strip() == "x"


async def test_a_command_never_sees_this_machine_s_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "secret-value")
    monkeypatch.setenv("GIT_DIR", "/somewhere/else")
    sandbox = LocalSandbox(root=str(tmp_path)).bind("user-1", {"MY_VARIABLE": "x"})
    seen = (await sandbox.execute_command("env")).stdout
    assert "secret-value" not in seen and "GIT_DIR" not in seen
    # What bash needs is there: programs are found and the home is the user's.
    assert (await sandbox.execute_command("command -v python3 && echo $HOME")).exit_code == 0
    assert "PATH=" in seen and "HOME=" in seen and "MY_VARIABLE=x" in seen


async def test_extra_env_names_what_else_a_command_sees(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "secret-value")
    monkeypatch.delenv("NOT_SET", raising=False)
    sandbox = LocalSandbox(root=str(tmp_path), extra_env=["MY_API_KEY", "NOT_SET"])
    # Stored by name; the value is read when the command runs.
    assert "secret-value" not in sandbox.serialize().model_dump_json()
    again = SandboxCore.deserialize(sandbox.serialize().model_dump())
    seen = (await again.execute_command("echo $MY_API_KEY; env")).stdout
    assert seen.startswith("secret-value\n") and "NOT_SET" not in seen


async def test_git_never_takes_the_repository_above_the_session(tmp_path):
    sandbox = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await sandbox.prepare()
    session = sandbox.session_paths("s1")["workspace"]
    await sandbox.execute_command(f"git init -q . && mkdir -p {session}")  # the project's own
    # 1. From the session's folder, the project's repository is not found.
    above = await sandbox.execute_command(f"cd {session} && git status")
    assert above.exit_code != 0 and "not a git repository" in above.stderr
    # 2. A repository the model makes there works as always.
    own = await sandbox.execute_command(f"cd {session} && git init -q . && git status")
    assert own.exit_code == 0


async def test_a_command_stops_at_its_timeout(tmp_path):
    result = await LocalSandbox(root=str(tmp_path)).execute_command("sleep 5", timeout=0.2)
    assert result.timed_out and result.exit_code is None


async def test_a_cancelled_command_leaves_nothing_running(tmp_path):
    beat = tmp_path / "beat"
    loop = f"while true; do date +%s%N > {beat}; sleep 0.05; done"
    running = asyncio.create_task(
        LocalSandbox(root=str(tmp_path)).execute_command(f"({loop}) & {loop}", timeout=60)
    )
    await asyncio.sleep(0.5)
    running.cancel()
    await asyncio.gather(running, return_exceptions=True)
    await asyncio.sleep(0.3)
    seen = beat.read_text()
    await asyncio.sleep(0.3)
    assert running.cancelled() and beat.read_text() == seen  # nobody writes any more


async def test_output_that_is_not_text_does_not_fail(tmp_path):
    result = await LocalSandbox(root=str(tmp_path)).execute_command(r"printf '\xff\xfe'")
    assert result.exit_code == 0
