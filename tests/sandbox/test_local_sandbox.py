"""LocalSandbox: this computer is the sandbox."""

from max_ai.base.sandbox import SandboxCore
from max_ai.capabilities.sandbox.local import LocalSandbox


def test_building_it_creates_nothing(tmp_path):
    LocalSandbox(root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_it_is_stored_with_the_root_as_given():
    sandbox = LocalSandbox(root=".")
    assert sandbox.serialize().config == {"root": "."}
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


async def test_a_command_stops_at_its_timeout(tmp_path):
    result = await LocalSandbox(root=str(tmp_path)).execute_command("sleep 5", timeout=0.2)
    assert result.timed_out and result.exit_code is None


async def test_output_that_is_not_text_does_not_fail(tmp_path):
    result = await LocalSandbox(root=str(tmp_path)).execute_command(r"printf '\xff\xfe'")
    assert result.exit_code == 0
