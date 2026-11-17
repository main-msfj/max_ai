"""Runs inside the sandbox: writes one file for write_file. Standard library only."""

import hashlib
import json
import os
import shutil
import sys


def inside(file, folders):
    """``file`` (links resolved) is in one of ``folders``."""
    return any(file.startswith(os.path.realpath(folder) + os.sep) for folder in folders)


def write(request, content):
    # 1. All of the content arrived.
    if hashlib.sha256(content).hexdigest() != request["sha256"]:
        raise ValueError("its content arrived incomplete")

    # 2. Resolve the path from the working directory, links included.
    file = os.path.realpath(os.path.join(request["cwd"], request["path"]))
    if not inside(file, request["allowed"]):
        raise ValueError("it is outside the folders you can write")
    if os.path.isdir(file):
        raise ValueError("it is a folder, not a file")

    # 3. Write beside the file and swap, so a cut write leaves no half file.
    existed = os.path.exists(file)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    partial = file + ".partial"
    with open(partial, "wb") as handle:
        handle.write(content)
    if existed:
        shutil.copymode(file, partial)  # a script stays executable
    os.replace(partial, file)
    return {"existed": existed}


def main():
    try:
        answer = {"ok": True, **write(json.loads(sys.argv[1]), sys.stdin.buffer.read())}
    except (ValueError, OSError) as error:
        answer = {"ok": False, "error": str(error)}
    print(json.dumps(answer))


if __name__ == "__main__":
    main()
