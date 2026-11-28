"""Everything the harness does to ``.agent/`` from outside, run inside the
environment: create it, read its settings, put a file in, install a skill.

Standard library only. The request is one JSON argument (``op`` says what
to do); a file or a skill arrives as raw bytes on stdin; the answer is one
JSON on stdout.

Nothing is ever left half done: every file and folder is built under a
temporary name and renamed into place, and what arrives is checked against
its sha256 first, so a connection cut in the middle changes nothing.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import typing as t

# A request and its answer: what JSON carries.
Json = t.Dict[str, t.Any]

# Temporary names start like this; what a killed run left behind is removed
# by a later install, once it is old enough.
INCOMING = ".incoming-"
STALE_SECONDS = 3600
# The states of a task; the same words as TaskStatus on the host.
RUNNING, DONE, CAP, STOPPED, LOST = "running", "done", "cap", "stopped", "lost"
# A task with no end written still counts as running this long after its cap.
GRACE_SECONDS = 5
# How much of the end of an output is read to show its last lines.
TAIL_BYTES = 1 << 16
# How many tasks that ended a session keeps; older ones are removed.
KEPT_TASKS = 10
# How often a watcher looks whether its command ended without printing.
POLL_SECONDS = 0.2
# What takes the place of the middle of an output too long to keep.
CUT = "\n[output cut to make room: {left} characters left out here]\n"


def inside(folder: str, path: str) -> str:
    """``path`` under ``folder`` with symlinks followed, or an error."""
    folder = os.path.realpath(folder)
    full = os.path.realpath(os.path.join(folder, path))
    if not full.startswith(folder + os.sep):
        raise ValueError(f"{path} leaves {folder}")
    return full


def receive(folder: str, sha256: str) -> str:
    """Stdin saved under a temporary name in ``folder``, only if all of it
    arrived. The caller renames it into place."""
    handle, temp = tempfile.mkstemp(dir=folder, prefix=INCOMING)
    digest = hashlib.sha256()
    with os.fdopen(handle, "wb") as file:
        while True:
            chunk = sys.stdin.buffer.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            file.write(chunk)
        file.flush()
        os.fsync(file.fileno())
    if digest.hexdigest() != sha256:
        os.unlink(temp)
        raise ValueError("the content arrived incomplete; nothing was changed")
    return temp


def write_settings(path: str, content: Json) -> None:
    """Replace ``setting.json`` in one step: a reader sees the old one or
    the new one, never half."""
    handle, temp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=INCOMING)
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        file.write(json.dumps(content, indent=2) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.chmod(temp, 0o644)
    os.replace(temp, path)


def read_settings(path: str) -> Json:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


@contextlib.contextmanager
def locked(path: str) -> t.Generator[None, None, None]:
    """One install at a time in this ``.agent/``. The system frees the lock
    when the process ends, also when it is killed."""
    with open(path, "a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield


def create(request: Json) -> Json:
    for folder in request["folders"]:
        os.makedirs(folder, exist_ok=True)
    # Where the harness is another user: the agent's one writes only here.
    for folder in request.get("open", []):
        os.chmod(folder, 0o777)
    settings = request["settings"]
    handle, temp = tempfile.mkstemp(dir=os.path.dirname(settings), prefix=INCOMING)
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        file.write(json.dumps(request["content"], indent=2) + "\n")
    os.chmod(temp, 0o644)
    # link fails if the file exists and never shows it empty: two creations
    # at once leave one complete setting.json, and an existing one stays.
    try:
        os.link(temp, settings)
    except FileExistsError:
        pass
    os.unlink(temp)
    return {}


def settings(request: Json) -> Json:
    return {"settings": read_settings(request["settings"])}


def permit(request: Json) -> Json:
    """Add one rule to a list of ``permissions``; the same rule is kept once."""
    with locked(request["lock"]):
        current = read_settings(request["settings"])
        rules = current.setdefault("permissions", {}).setdefault(request["kind"], [])
        if request["rule"] not in rules:
            rules.append(request["rule"])
            write_settings(request["settings"], current)
    return {}


def unpermit(request: Json) -> Json:
    """Take one rule out of a list of ``permissions``."""
    with locked(request["lock"]):
        current = read_settings(request["settings"])
        rules = current.get("permissions", {}).get(request["kind"], [])
        if request["rule"] in rules:
            rules.remove(request["rule"])
            write_settings(request["settings"], current)
    return {}


def upload(request: Json) -> Json:
    full = inside(request["folder"], request["path"])
    os.makedirs(os.path.dirname(full), exist_ok=True)
    temp = receive(os.path.dirname(full), request["sha256"])
    os.chmod(temp, 0o644)
    os.replace(temp, full)
    sweep(os.path.dirname(full))
    return {"path": full}


def unpack(packed: str, folder: str) -> None:
    """Only plain files and folders: no links, nothing outside ``folder``."""
    with tarfile.open(packed) as archive:
        for member in archive:
            if member.isdir():
                os.makedirs(inside(folder, member.name), exist_ok=True)
            elif member.isfile():
                full = inside(folder, member.name)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "wb") as file:
                    # A plain file always has its content in the archive.
                    shutil.copyfileobj(t.cast("t.IO[bytes]", archive.extractfile(member)), file)
                os.chmod(full, 0o755 if member.mode & 0o100 else 0o644)


def digest(folder: str) -> str | None:
    """One hash of everything in ``folder``: each file's path, whether it
    runs, and its content. None when the folder is not there."""
    if not os.path.isdir(folder):
        return None
    total = hashlib.sha256()
    for root, folders, files in os.walk(folder):
        folders.sort()
        for name in sorted(files + [f for f in folders if os.path.islink(os.path.join(root, f))]):
            full = os.path.join(root, name)
            total.update(os.path.relpath(full, folder).encode() + b"\0")
            # A link counts as where it points, never as what is there.
            if os.path.islink(full):
                total.update(b"link\0" + os.readlink(full).encode() + b"\0")
                continue
            total.update(b"x\0" if os.lstat(full).st_mode & 0o100 else b"-\0")
            with open(full, "rb") as file:
                for chunk in iter(lambda: file.read(1 << 20), b""):
                    total.update(chunk)
            total.update(b"\0")
    return total.hexdigest()


def hashes(request: Json) -> Json:
    """The hash of each skill folder asked for, as it is on disk now."""
    folder: str = request["folder"]
    names: list[str] = request["names"]
    return {"hashes": {name: digest(os.path.join(folder, name)) for name in names}}


def install(request: Json) -> Json:
    name, key, target = request["name"], request["key"], request["folder"]
    parent = os.path.dirname(target)
    # Receive everything BEFORE taking the lock: a slow connection never
    # keeps the others waiting, and a cut one stops here, nothing touched.
    packed = receive(parent, request["sha256"])
    try:
        with locked(request["lock"]):
            sweep(parent)
            sweep(os.path.dirname(request["settings"]))
            current = read_settings(request["settings"])
            skills = current.setdefault("skills", {})
            # Someone installed it while this call waited for the lock, and
            # what is on disk is still what was installed.
            known: Json = skills.get(name) or {}
            found = digest(target)
            if known.get("key") == key and found is not None and known.get("hash") == found:
                return {"installed": False}

            fresh = tempfile.mkdtemp(dir=parent, prefix=INCOMING)
            try:
                unpack(packed, fresh)
            except BaseException:
                shutil.rmtree(fresh, ignore_errors=True)
                raise
            os.chmod(fresh, 0o755)

            # 1. Forget the old key first. If anything stops between here and
            #    step 3, the settings say "not installed" and the next run
            #    installs again: they never name a version that is not there.
            if skills.pop(name, None) is not None:
                write_settings(request["settings"], current)
            # 2. Swap the folders: two renames, the old one deleted after.
            old = fresh + ".old"
            if os.path.lexists(target):
                os.replace(target, old)
            os.replace(fresh, target)
            shutil.rmtree(old, ignore_errors=True)
            # 3. Record the new key, what the prompt says of the skill, and the
            #    hash of its files: a later change to them is noticed.
            skills[name] = {
                "key": key,
                "description": request.get("description", ""),
                "hash": digest(target),
            }
            write_settings(request["settings"], current)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(packed)
    return {"installed": True}


def sweep(folder: str) -> None:
    """Remove what a killed install left behind long ago. Recent temporary
    names may belong to an install still receiving its content."""
    for entry in os.listdir(folder):
        if not entry.startswith(INCOMING):
            continue
        leftover = os.path.join(folder, entry)
        # Another install may remove its own temporary file meanwhile.
        with contextlib.suppress(OSError):
            if time.time() - os.lstat(leftover).st_mtime > STALE_SECONDS:
                shutil.rmtree(leftover) if os.path.isdir(leftover) else os.unlink(leftover)


def put(folder: str, name: str, value: t.Any, once: bool = False) -> None:
    """Write one small file of a task whole. With ``once`` the first to
    write it wins, and a later one changes nothing."""
    handle, temp = tempfile.mkstemp(dir=folder, prefix=INCOMING)
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        file.write(json.dumps(value))
    os.chmod(temp, 0o644)
    target = os.path.join(folder, name)
    if not once:
        os.replace(temp, target)
        return
    # link fails if the file exists: nobody writes over an end already told.
    with contextlib.suppress(FileExistsError):
        os.link(temp, target)
    os.unlink(temp)


def get(folder: str, name: str) -> t.Any:
    """One small file of a task, or None when it is not there."""
    try:
        with open(os.path.join(folder, name), encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None


def alive(pid: t.Any) -> bool:
    """Whether a process with this number exists."""
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def tail(path: str, lines: int) -> str:
    """The last ``lines`` lines of a file, reading only its end."""
    if lines <= 0 or not os.path.isfile(path):
        return ""
    with open(path, "rb") as file:
        file.seek(0, os.SEEK_END)
        file.seek(max(0, file.tell() - TAIL_BYTES))
        text = file.read().decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-lines:])


def task_folder(root: str, name: str) -> str:
    """The folder of one task of a session, or an error when there is none."""
    folder = inside(root, name)
    if not name.isdigit() or not os.path.isdir(folder):
        raise ValueError(f"there is no task {name}")
    return folder


class Output:
    """What a task prints, on disk: all of it while it fits in ``limit``,
    then only its beginning and its end."""

    def __init__(self, file: t.IO[bytes], limit: int) -> None:
        self.file = file
        self.half = limit // 2
        self.total = 0  # everything printed so far
        self.end = b""  # the last half, once the first one is written

    def add(self, chunk: bytes) -> None:
        # 1. The first half is written as it comes; of the rest, only the end.
        first = chunk[: max(0, self.half - self.total)]
        self.file.write(first)
        self.end = (self.end + chunk[len(first) :])[-self.half :]
        self.total += len(chunk)
        # 2. The end is written again behind the first half, with what was left out.
        if self.total > self.half:
            left = self.total - 2 * self.half
            self.file.seek(self.half)
            self.file.truncate()
            self.file.write((CUT.format(left=left).encode() if left > 0 else b"") + self.end)
        self.file.flush()


def watch(folder: str, command: str, cwd: str, seconds: float, limit: int) -> None:
    """Stays beside one background task: runs it, keeps what it prints, ends
    it at the cap and writes how it ended."""
    deadline = time.monotonic() + seconds
    with open(os.path.join(folder, "output"), "wb") as file:
        # 1. The command in its own group: killing the group kills all it started.
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        put(folder, "pid", process.pid)
        # A stop that came before the pid was written.
        if get(folder, "exit") is not None:
            os.killpg(process.pid, signal.SIGKILL)
        # 2. What it prints is kept until it ends by itself, or the cap ends it.
        output, pipe = Output(file, limit), t.cast("t.IO[bytes]", process.stdout).fileno()
        code, reason = None, CAP
        while time.monotonic() < deadline:
            over = process.poll() is not None
            ready = select.select([pipe], [], [], 0 if over else POLL_SECONDS)[0]
            chunk = os.read(pipe, 1 << 16) if ready else b""
            if chunk:
                output.add(chunk)
            elif ready or over:
                # Nothing more will come: it ended, or it closed its output.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    code = process.wait(max(0.0, deadline - time.monotonic()))
                    reason = DONE
                break
        if reason == CAP:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    # 3. How it ended.
    put(folder, "exit", {"reason": reason, "code": code}, once=True)


def numbered(root: str) -> list[str]:
    """The tasks of a session, in the order they started."""
    if not os.path.isdir(root):
        return []
    return sorted((name for name in os.listdir(root) if name.isdigit()), key=int)


def task_start(request: Json) -> Json:
    root: str = request["folder"]
    os.makedirs(root, exist_ok=True)
    # Where it starts is there, also for the first command of a session.
    os.makedirs(request["cwd"], exist_ok=True)
    with locked(os.path.join(root, ".lock")):
        sweep(root)
        # 1. No more than the session may have running at once.
        states = [task_state(os.path.join(root, name), 0) for name in numbered(root)]
        if sum(task["state"] == RUNNING for task in states) >= request["max"]:
            raise ValueError(
                f"this session already has {request['max']} background tasks running; "
                "wait for one to end or stop one"
            )
        # 2. Of the ones that ended, only the last ones are kept.
        ended = [task["id"] for task in states if task["state"] != RUNNING]
        for name in ended[:-KEPT_TASKS]:
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
        # 3. The folder of the task, whole from the start: the next number.
        number = 1 + max((int(task["id"]) for task in states), default=0)
        fresh = tempfile.mkdtemp(dir=root, prefix=INCOMING)
        os.chmod(fresh, 0o755)
        put(fresh, "command", request["command"])
        put(fresh, "started", time.time())
        put(fresh, "seconds", request["seconds"])
        folder = os.path.join(root, str(number))
        os.rename(fresh, folder)

    # 4. A copy of this process stays to watch; the call itself answers and ends.
    watcher = os.fork()
    if watcher == 0:
        try:
            os.setsid()
            # Nothing of the call stays open, or whoever called would wait for it.
            null = os.open(os.devnull, os.O_RDWR)
            for stream in (0, 1, 2):
                os.dup2(null, stream)
            watch(folder, request["command"], request["cwd"], request["seconds"], request["limit"])
        finally:
            os._exit(0)
    put(folder, "watcher", watcher)
    return {"id": str(number)}


def task_state(folder: str, lines: int) -> Json:
    """What is known of one task: how it ended, or whether it still runs."""
    ended = get(folder, "exit")
    if ended is None:
        # No end written: it runs while its watcher lives (or is about to be
        # written down) and its cap has not passed.
        limit = (get(folder, "started") or 0) + (get(folder, "seconds") or 0) + GRACE_SECONDS
        watcher = get(folder, "watcher")
        running = time.time() < limit and (watcher is None or alive(watcher))
        ended = {"reason": RUNNING if running else LOST, "code": None}
    return {
        "id": os.path.basename(folder),
        "command": get(folder, "command"),
        "started": get(folder, "started"),
        "state": ended["reason"],
        "code": ended["code"],
        "output": tail(os.path.join(folder, "output"), lines),
    }


def task_list(request: Json) -> Json:
    """The tasks of one session, or only the one asked for."""
    root: str = request["folder"]
    wanted: str | None = request.get("id")
    # 1. The one asked for, or all of the session in the order they started.
    names = [wanted] if wanted is not None else numbered(root)
    # 2. What is known of each.
    lines: int = request["lines"]
    return {"tasks": [task_state(task_folder(root, name), lines) for name in names]}


def task_stop(request: Json) -> Json:
    folder = task_folder(request["folder"], request["id"])
    # Only a task that still runs is stopped.
    if task_state(folder, 0)["state"] == RUNNING:
        # 1. Say why first: the watcher will not write over it.
        put(folder, "exit", {"reason": STOPPED, "code": None}, once=True)
        # 2. Then kill the command and all it started.
        with contextlib.suppress(ProcessLookupError, TypeError):
            os.killpg(get(folder, "pid"), signal.SIGKILL)
    return {}


OPERATIONS: dict[str, t.Callable[[Json], Json]] = {
    "create": create,
    "settings": settings,
    "upload": upload,
    "install": install,
    "hashes": hashes,
    "permit": permit,
    "unpermit": unpermit,
    "task_start": task_start,
    "task_list": task_list,
    "task_stop": task_stop,
}


def main() -> None:
    request = json.loads(sys.argv[1])
    try:
        answer: dict[str, t.Any] = {"ok": True, **OPERATIONS[request["op"]](request)}
    except (ValueError, OSError, tarfile.TarError) as error:
        answer = {"ok": False, "error": str(error)}
    print(json.dumps(answer))


if __name__ == "__main__":
    main()
