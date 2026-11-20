"""Runs inside the sandbox: edits one file for edit_file. Standard library only."""

import hashlib
import json
import os
import shutil
import sys
from typing import Any


def inside(file: str, folders: list[str]) -> bool:
    """``file`` (links resolved) is in one of ``folders``."""
    return any(file.startswith(os.path.realpath(folder) + os.sep) for folder in folders)


def windows(text: str) -> str:
    """``text`` with the line breaks of a Windows file."""
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


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


def edit(request: dict[str, Any]) -> dict[str, Any]:
    # 1. Resolve the path from the working directory, links included.
    cwd: str = request["cwd"]
    given: str = request["path"]
    file = os.path.realpath(os.path.join(cwd, given))
    if not inside(file, request["allowed"]):
        folders = ", ".join(request["allowed"])
        raise ValueError(
            f"it resolves to {file}, which is outside the folders you can write "
            f"({folders}); use a full path inside one of them"
        )
    if not os.path.exists(file):
        raise ValueError("it does not exist; create it with the tool that writes files")
    if os.path.isdir(file):
        raise ValueError("it is a folder, not a file")

    # 2. Only UTF-8 text is edited: other bytes would be damaged when saved.
    with open(file, "rb") as handle:
        try:
            content = handle.read().decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                "it is not UTF-8 text, so it cannot be edited in parts; "
                "to change it, write the whole file again"
            )

    # 3. A Windows file is shown without its \r: put them back in both texts.
    old: str = request["old"]
    new: str = request["new"]
    if "\r\n" in content:
        old, new = windows(old), windows(new)

    # 4. The old text must be there, and once unless all are wanted.
    found = content.count(old)
    if found == 0:
        raise ValueError("the old text is not in the file; read it again and copy it exactly")
    if found > 1 and not request["all"]:
        raise ValueError(
            f"the old text appears {found} times; add lines around it to make it unique, "
            "or set replace_all"
        )

    # 5. Write beside the file and swap, so a cut write leaves no half file.
    edited = content.replace(old, new).encode("utf-8")
    if inside(file, [request["memory"]]):
        memory_file(edited, request["memory_max"])
    partial = file + ".partial"
    with open(partial, "wb") as handle:
        handle.write(edited)
    shutil.copymode(file, partial)  # a script stays executable
    os.replace(partial, file)
    # 6. Where the first change begins, counted in lines.
    line = content[: content.index(old)].count("\n") + 1
    return {"replaced": found, "line": line, "sha256": hashlib.sha256(edited).hexdigest()}


def main() -> None:
    try:
        answer: dict[str, Any] = {"ok": True, **edit(json.loads(sys.stdin.read()))}
    except (ValueError, OSError) as error:
        answer = {"ok": False, "error": str(error)}
    print(json.dumps(answer))


if __name__ == "__main__":
    main()
