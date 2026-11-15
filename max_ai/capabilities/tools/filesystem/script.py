"""File operations for read_file, write_file and edit_file.

Runs inside any sandbox with plain python3 (standard library only): reads
one JSON request on stdin and prints one JSON answer. Never touches a path
outside ``root``, and writes through a temporary file, so a reader never
sees half a file.
"""

import json
import os
import sys
import tempfile


def resolve(root, path):
    root = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError(f"{path} is outside the workspace")
    return full


def save(full, text):
    os.makedirs(os.path.dirname(full), exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=os.path.dirname(full))
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        file.write(text)
    os.replace(temp, full)


def read(full, offset=0, limit=None):
    with open(full, encoding="utf-8") as file:
        lines = file.read().splitlines(keepends=True)
    end = len(lines) if limit is None else offset + limit
    return {"content": "".join(lines[offset:end]), "total_lines": len(lines)}


def write(full, content):
    save(full, content)
    return {"written": len(content)}


def edit(full, old_string, new_string, replace_all=False):
    with open(full, encoding="utf-8") as file:
        text = file.read()
    count = text.count(old_string)
    if count == 0:
        raise ValueError("old_string was not found in the file")
    if count > 1 and not replace_all:
        raise ValueError(f"old_string appears {count} times; add context or set replace_all")
    save(full, text.replace(old_string, new_string))
    return {"replaced": count if replace_all else 1}


OPERATIONS = {"read": read, "write": write, "edit": edit}


def main():
    request = json.load(sys.stdin)
    try:
        full = resolve(request.pop("root"), request.pop("path"))
        answer = OPERATIONS[request.pop("op")](full, **request)
        print(json.dumps({"ok": True, **answer}))
    except (ValueError, OSError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))


if __name__ == "__main__":
    main()
