"""Background tasks: a command that goes on inside the sandbox after the call that started it."""

import asyncio
import json
import time
from pathlib import Path

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.config import setting


async def box(tmp_path) -> LocalSandbox:
    sandbox = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await sandbox.prepare()
    return sandbox


async def ended(folder: Path, seconds: float = 10) -> dict:
    """How the task of ``folder`` ended, once it says so."""
    limit = time.monotonic() + seconds
    while not (folder / "exit").exists():
        assert time.monotonic() < limit, "the task never said how it ended"
        await asyncio.sleep(0.05)
    return json.loads((folder / "exit").read_text())


async def test_a_task_goes_on_after_the_call_and_says_how_it_ended(tmp_path):
    sandbox = await box(tmp_path)
    start = time.monotonic()
    task_id = await sandbox.start_task("s1", "sleep 0.5; echo out; echo err >&2; exit 3", str(tmp_path))
    # The call does not wait for the command.
    assert task_id == "1" and time.monotonic() - start < 0.4
    folder = Path(str(sandbox.layout.tasks)) / "s1" / "1"
    assert json.loads((folder / "command").read_text()).endswith("exit 3")
    assert await ended(folder) == {"reason": "done", "code": 3}
    assert (folder / "output").read_text() == "out\nerr\n"
    assert time.time() - json.loads((folder / "started").read_text()) < 10


async def test_the_cap_ends_a_task_and_all_it_started(tmp_path, monkeypatch):
    monkeypatch.setattr(setting, "harness_task_seconds", 0.5)
    sandbox = await box(tmp_path)
    marker = tmp_path / "late"
    await sandbox.start_task("s1", f"(sleep 1.5; touch {marker}) & echo started; sleep 30", str(tmp_path))
    folder = Path(str(sandbox.layout.tasks)) / "s1" / "1"
    assert await ended(folder) == {"reason": "cap", "code": None}
    assert (folder / "output").read_text() == "started\n"
    # What the command left running died with it.
    await asyncio.sleep(1.5)
    assert not marker.exists()


async def test_each_session_numbers_its_own_tasks(tmp_path):
    sandbox = await box(tmp_path)
    first = await asyncio.gather(*(sandbox.start_task("s1", "true", str(tmp_path)) for _ in range(3)))
    assert sorted(first) == ["1", "2", "3"]
    assert await sandbox.start_task("s2", "true", str(tmp_path)) == "1"
    # It starts where it is told to.
    await sandbox.start_task("s2", "pwd", str(tmp_path / ".agent"))
    folder = Path(str(sandbox.layout.tasks)) / "s2" / "2"
    await ended(folder)
    assert (folder / "output").read_text().strip() == str(tmp_path / ".agent")
