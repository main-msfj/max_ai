"""Everything the harness does to ``.agent/`` from outside, run inside the
environment: create it, read its settings, put a file in, install a skill.

Standard library only. The request is one JSON argument (``op`` says what
to do); a file or a skill arrives as raw bytes on stdin; the answer is one
JSON on stdout.

Nothing is ever left half done: every file and folder is built under a
temporary name and renamed into place, and what arrives is checked against
its sha256 first, so a connection cut in the middle changes nothing.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time

# Temporary names start like this; what a killed run left behind is removed
# by a later install, once it is old enough.
INCOMING = ".incoming-"
STALE_SECONDS = 3600


def inside(folder, path):
    """``path`` under ``folder`` with symlinks followed, or an error."""
    folder = os.path.realpath(folder)
    full = os.path.realpath(os.path.join(folder, path))
    if not full.startswith(folder + os.sep):
        raise ValueError(f"{path} leaves {folder}")
    return full


def receive(folder, sha256):
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


def write_settings(path, content):
    """Replace ``setting.json`` in one step: a reader sees the old one or
    the new one, never half."""
    handle, temp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=INCOMING)
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        file.write(json.dumps(content, indent=2) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.chmod(temp, 0o644)
    os.replace(temp, path)


def read_settings(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


@contextlib.contextmanager
def locked(path):
    """One install at a time in this ``.agent/``. The system frees the lock
    when the process ends, also when it is killed."""
    with open(path, "a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield


def create(request):
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


def settings(request):
    return {"settings": read_settings(request["settings"])}


def permit(request):
    """Add one rule to a list of ``permissions``; the same rule is kept once."""
    with locked(request["lock"]):
        current = read_settings(request["settings"])
        rules = current.setdefault("permissions", {}).setdefault(request["kind"], [])
        if request["rule"] not in rules:
            rules.append(request["rule"])
            write_settings(request["settings"], current)
    return {}


def unpermit(request):
    """Take one rule out of a list of ``permissions``."""
    with locked(request["lock"]):
        current = read_settings(request["settings"])
        rules = current.get("permissions", {}).get(request["kind"], [])
        if request["rule"] in rules:
            rules.remove(request["rule"])
            write_settings(request["settings"], current)
    return {}


def upload(request):
    full = inside(request["folder"], request["path"])
    os.makedirs(os.path.dirname(full), exist_ok=True)
    temp = receive(os.path.dirname(full), request["sha256"])
    os.chmod(temp, 0o644)
    os.replace(temp, full)
    sweep(os.path.dirname(full))
    return {"path": full}


def unpack(packed, folder):
    """Only plain files and folders: no links, nothing outside ``folder``."""
    with tarfile.open(packed) as archive:
        for member in archive:
            if member.isdir():
                os.makedirs(inside(folder, member.name), exist_ok=True)
            elif member.isfile():
                full = inside(folder, member.name)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "wb") as file:
                    shutil.copyfileobj(archive.extractfile(member), file)
                os.chmod(full, 0o755 if member.mode & 0o100 else 0o644)


def digest(folder):
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


def hashes(request):
    """The hash of each skill folder asked for, as it is on disk now."""
    return {"hashes": {name: digest(os.path.join(request["folder"], name)) for name in request["names"]}}


def install(request):
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
            known = skills.get(name) or {}
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
                "key": key, "description": request.get("description", ""), "hash": digest(target),
            }
            write_settings(request["settings"], current)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(packed)
    return {"installed": True}


def sweep(folder):
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


OPERATIONS = {"create": create, "settings": settings, "upload": upload, "install": install,
              "hashes": hashes, "permit": permit, "unpermit": unpermit}


def main():
    request = json.loads(sys.argv[1])
    try:
        answer = {"ok": True, **OPERATIONS[request["op"]](request)}
    except (ValueError, OSError, tarfile.TarError) as error:
        answer = {"ok": False, "error": str(error)}
    print(json.dumps(answer))


if __name__ == "__main__":
    main()
