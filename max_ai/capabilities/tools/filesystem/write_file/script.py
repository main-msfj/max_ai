"""Runs inside the sandbox: writes one file for write_file. Standard library only."""

import hashlib
import json
import os
import shutil
import sys
from typing import Any


def inside(file: str, folders: list[str]) -> bool:
    """``file`` (links resolved) is in one of ``folders``."""
    return any(file.startswith(os.path.realpath(folder) + os.sep) for folder in folders)


TYPES = ("user", "feedback", "project", "reference")
HEADER = (
    "---\nname: short-name\ndescription: one line saying what it holds\n"
    "type: user | feedback | project | reference\n---"
)


def memory_file(content: bytes, most: int) -> None:
    """A memory file is small and starts by saying what it holds: the
    index the model sees is made of those lines."""
    if len(content) > most:
        raise ValueError(f"a memory file holds at most {most // 1000} kB; split it into smaller files")
    # The fields between the two ``---`` lines at the top.
    lines = content.decode("utf-8", errors="replace").splitlines()
    fields: dict[str, str] = {}
    closed = False
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                closed = True
                break
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip().strip("\"'")
    complete = closed and fields.get("name") and fields.get("description")
    if not complete or fields.get("type") not in TYPES:
        raise ValueError(f"a memory file starts with these lines, all filled in:\n{HEADER}\n")


def write(request: dict[str, Any], content: bytes) -> dict[str, bool]:
    # 1. All of the content arrived.
    if hashlib.sha256(content).hexdigest() != request["sha256"]:
        raise ValueError("its content arrived incomplete")

    # 2. Resolve the path from the working directory, links included.
    cwd: str = request["cwd"]
    given: str = request["path"]
    file = os.path.realpath(os.path.join(cwd, given))
    if not inside(file, request["allowed"]):
        folders = ", ".join(request["allowed"])
        raise ValueError(
            f"it resolves to {file}, which is outside the folders you can write "
            f"({folders}); use a full path inside one of them"
        )
    if os.path.isdir(file):
        raise ValueError("it is a folder, not a file")
    # 3. The memory is one folder: a path that only looks like it would be
    #    written where nobody reads it.
    memory: str = request["memory"]
    if inside(file, [memory]):
        memory_file(content, request["memory_max"])
    elif os.sep + os.sep.join(memory.split(os.sep)[-2:]) + os.sep in file:
        raise ValueError(
            f"it resolves to {file}, which is not the memory; the memory is in {memory}: "
            f"write to {os.path.join(memory, os.path.basename(file))}"
        )

    # 4. An existing file is replaced only if the model saw it as it is now.
    existed = os.path.exists(file)
    if existed:
        with open(file, "rb") as handle:
            now = hashlib.sha256(handle.read()).hexdigest()
        if request["seen"] is None:
            raise ValueError("it already exists and you have not read it; read it first")
        if request["seen"] != now:
            raise ValueError("it changed since you read it; read it again")

    # 5. Write beside the file and swap, so a cut write leaves no half file.
    os.makedirs(os.path.dirname(file), exist_ok=True)
    partial = file + ".partial"
    with open(partial, "wb") as handle:
        handle.write(content)
    if existed:
        shutil.copymode(file, partial)  # a script stays executable
    os.replace(partial, file)
    return {"existed": existed}


def main() -> None:
    try:
        answer: dict[str, Any] = {"ok": True, **write(json.loads(sys.argv[1]), sys.stdin.buffer.read())}
    except (ValueError, OSError) as error:
        answer = {"ok": False, "error": str(error)}
    print(json.dumps(answer))


if __name__ == "__main__":
    main()
