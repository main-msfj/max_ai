"""Background tasks: a command that goes on inside the sandbox after the call that started it."""

import asyncio
import json
import os
import signal
import time
from pathlib import Path

import pytest

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.config import setting
from max_ai.errors.sandbox import SandboxError


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


async def test_a_task_is_read_while_it_runs_and_after_it_ends(tmp_path):
    sandbox = await box(tmp_path)
    assert await sandbox.tasks("s1") == []
    await sandbox.start_task("s1", "echo one; echo two; sleep 0.6; echo three", str(tmp_path))
    await sandbox.start_task("s1", "exit 4", str(tmp_path))
    await asyncio.sleep(0.3)
    first, second = await sandbox.tasks("s1", lines=1)
    assert (first.id, first.state, first.code, first.output) == ("1", "running", None, "two")
    assert (second.id, second.state, second.code) == ("2", "done", 4)
    await ended(Path(str(sandbox.layout.tasks)) / "s1" / "1")
    # Only the one asked for, with the last lines asked for.
    (only,) = await sandbox.tasks("s1", "1", lines=2)
    assert (only.state, only.code, only.output) == ("done", 0, "two\nthree")
    assert only.command.endswith("echo three") and time.time() - only.started < 10


async def test_stopping_a_task_ends_it_and_all_it_started(tmp_path):
    sandbox = await box(tmp_path)
    marker = tmp_path / "late"
    await sandbox.start_task("s1", f"(sleep 1; touch {marker}) & echo started; sleep 30", str(tmp_path))
    await asyncio.sleep(0.3)
    await sandbox.stop_task("s1", "1")
    (task,) = await sandbox.tasks("s1")
    assert (task.state, task.code, task.output) == ("stopped", None, "started")
    # The watcher does not write over the reason, and what was left running died.
    await asyncio.sleep(1.2)
    assert (await sandbox.tasks("s1"))[0].state == "stopped"
    assert not marker.exists()


async def test_stopping_a_task_that_ended_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(setting, "harness_task_seconds", 0.4)
    sandbox = await box(tmp_path)
    await sandbox.start_task("s1", "exit 2", str(tmp_path))
    await sandbox.start_task("s1", "sleep 30", str(tmp_path))
    for number in ("1", "2"):
        await ended(Path(str(sandbox.layout.tasks)) / "s1" / number)
        await sandbox.stop_task("s1", number)
    done, cap = await sandbox.tasks("s1")
    assert (done.state, done.code, cap.state, cap.code) == ("done", 2, "cap", None)


async def test_a_task_whose_watcher_died_is_lost(tmp_path):
    sandbox = await box(tmp_path)
    await sandbox.start_task("s1", "sleep 30", str(tmp_path))
    folder = Path(str(sandbox.layout.tasks)) / "s1" / "1"
    await asyncio.sleep(0.3)
    command = json.loads((folder / "pid").read_text())
    os.kill(json.loads((folder / "watcher").read_text()), signal.SIGKILL)
    await asyncio.sleep(0.3)
    (task,) = await sandbox.tasks("s1")
    assert task.state == "lost"
    # A lost task is not stopped: its numbers may be someone else's by now.
    await sandbox.stop_task("s1", "1")
    os.kill(command, signal.SIGKILL)
    assert (await sandbox.tasks("s1"))[0].state == "lost"


async def test_a_task_past_its_cap_with_no_end_is_lost(tmp_path):
    sandbox = await box(tmp_path)
    await sandbox.start_task("s1", "sleep 2", str(tmp_path))
    folder = Path(str(sandbox.layout.tasks)) / "s1" / "1"
    # As after a restart: the watcher's number exists, but the cap passed long ago.
    (folder / "started").write_text(json.dumps(time.time() - 10_000))
    assert (await sandbox.tasks("s1"))[0].state == "lost"


async def test_a_task_is_only_reached_from_its_own_session(tmp_path):
    sandbox = await box(tmp_path)
    await sandbox.start_task("s1", "sleep 2", str(tmp_path))
    for task_id in ("7", "../s1/1", "x"):
        with pytest.raises(SandboxError, match="there is no task|leaves"):
            await sandbox.tasks("s2", task_id)
        with pytest.raises(SandboxError, match="there is no task|leaves"):
            await sandbox.stop_task("s2", task_id)
    assert (await sandbox.tasks("s1"))[0].state == "running"
    await sandbox.stop_task("s1", "1")


async def test_a_session_runs_only_so_many_tasks_at_once(tmp_path):
    sandbox = await box(tmp_path)
    for _ in range(setting.harness_task_max):
        await sandbox.start_task("s1", "sleep 30", str(tmp_path))
    with pytest.raises(SandboxError, match="already has 3 background tasks running"):
        await sandbox.start_task("s1", "true", str(tmp_path))
    # Another session has its own count, and room comes back when one ends.
    assert await sandbox.start_task("s2", "true", str(tmp_path)) == "1"
    await sandbox.stop_task("s1", "2")
    assert await sandbox.start_task("s1", "true", str(tmp_path)) == "4"
    for task in await sandbox.tasks("s1"):
        await sandbox.stop_task("s1", task.id)


async def test_only_the_last_tasks_that_ended_are_kept(tmp_path):
    sandbox = await box(tmp_path)
    root = Path(str(sandbox.layout.tasks)) / "s1"
    await sandbox.start_task("s1", "sleep 30", str(tmp_path))
    for number in range(2, 14):
        await sandbox.start_task("s1", "true", str(tmp_path))
        await ended(root / str(number))
    # The one still running stays, however old; of the rest, the last ten at each launch.
    kept = [task.id for task in await sandbox.tasks("s1")]
    assert kept == ["1", *(str(number) for number in range(3, 14))]
    await sandbox.stop_task("s1", "1")


async def test_a_long_output_keeps_its_beginning_and_its_end(tmp_path, monkeypatch):
    monkeypatch.setattr(setting, "harness_task_output_chars", 1000)
    sandbox = await box(tmp_path)
    await sandbox.start_task("s1", "seq 1 2000; sleep 0.8; echo last", str(tmp_path))
    await sandbox.start_task("s1", "seq 1 200", str(tmp_path))
    folder = Path(str(sandbox.layout.tasks)) / "s1"
    # While it runs, the last lines are already there.
    await asyncio.sleep(0.4)
    (running,) = await sandbox.tasks("s1", "1", lines=1)
    assert (running.state, running.output) == ("running", "2000")
    assert await ended(folder / "1") == {"reason": "done", "code": 0}
    text = (folder / "1" / "output").read_text()
    left = len("".join(f"{n}\n" for n in range(1, 2001)) + "last\n") - 1000
    assert text.startswith("1\n2\n3\n") and text.endswith("1999\n2000\nlast\n")
    assert f"[output cut to make room: {left} characters left out here]" in text
    assert len(text) < 1100
    # One that fits is kept whole.
    await ended(folder / "2")
    assert (folder / "2" / "output").read_text() == "".join(f"{n}\n" for n in range(1, 201))
