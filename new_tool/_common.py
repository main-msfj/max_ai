"""The start of every file script: it goes in front of read/script.py,
write/script.py and edit/script.py before they are sent to the environment.

Standard library only. Each script defines ``run(full, request)`` and ends
with ``main(run)``: one JSON request comes in on stdin, one JSON answer goes
out on stdout.
"""

import hashlib
import json
import os
import stat
import sys
import tempfile


class Refused(Exception):
    """An answer for the model (the file changed, the text was not found...),
    not a crash."""


def resolve(path, roots):
    """``path`` with its symlinks followed, if it is still inside one of
    ``roots``. The harness already checked the text of the path; only here,
    inside the environment, can a symlink that points outside be seen."""
    full = os.path.realpath(path)
    for root in roots:
        root = os.path.realpath(root)
        if full == root or full.startswith(root + os.sep):
            return full
    raise Refused("it is outside the workspace")


def stamp_of(data):
    """The fingerprint of a file's content. Same bytes, same stamp."""
    return hashlib.sha256(data).hexdigest()


def check(data, stamp):
    """Stop unless ``data`` is exactly what the model last read. ``stamp`` is
    what the harness kept from that read; None means it never read the file."""
    if stamp is None:
        raise Refused("read it before changing it")
    if stamp_of(data) != stamp:
        raise Refused("it changed since you read it; read it again before changing it")


def save(full, data):
    """Write ``data`` through a temporary file in the same folder, so nobody
    ever reads half a file. An existing file keeps its permissions."""
    folder = os.path.dirname(full)
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(full):
        mode = stat.S_IMODE(os.stat(full).st_mode)
    else:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    handle, temp = tempfile.mkstemp(dir=folder)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
        os.chmod(temp, mode)
        os.replace(temp, full)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def main(run):
    request = json.load(sys.stdin)
    try:
        full = resolve(request["path"], request["roots"])
        answer = {"ok": True, **run(full, request)}
    except Refused as error:
        answer = {"ok": False, "error": str(error)}
    except UnicodeDecodeError:
        answer = {"ok": False, "error": "it is not a UTF-8 text file"}
    except OSError as error:
        answer = {"ok": False, "error": (error.strerror or str(error)).lower()}
    print(json.dumps(answer))
