"""Runs inside the sandbox: edits one file for edit_file. Standard library only."""

import hashlib
import json
import os
import shutil
import sys


def inside(file, folders):
    """``file`` (links resolved) is in one of ``folders``."""
    return any(file.startswith(os.path.realpath(folder) + os.sep) for folder in folders)


def windows(text):
    """``text`` with the line breaks of a Windows file."""
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def edit(request):
    # 1. Resolve the path from the working directory, links included.
    file = os.path.realpath(os.path.join(request["cwd"], request["path"]))
    if not inside(file, request["allowed"]):
        raise ValueError("it is outside the folders you can write")
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
    old, new = request["old"], request["new"]
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
    partial = file + ".partial"
    with open(partial, "wb") as handle:
        handle.write(edited)
    shutil.copymode(file, partial)  # a script stays executable
    os.replace(partial, file)
    return {"replaced": found, "sha256": hashlib.sha256(edited).hexdigest()}


def main():
    try:
        answer = {"ok": True, **edit(json.loads(sys.stdin.read()))}
    except (ValueError, OSError) as error:
        answer = {"ok": False, "error": str(error)}
    print(json.dumps(answer))


if __name__ == "__main__":
    main()
