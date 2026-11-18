"""Lists the memory files with what each one holds. Runs inside the sandbox."""

import json
import os
import sys
from typing import Any


def header(path: str) -> dict[str, str]:
    """The fields between the two ``---`` lines at the top of a file."""
    fields: dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as file:
        if file.readline().strip() != "---":
            return fields
        for line in file:
            if line.strip() == "---":
                break
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip().strip("\"'")
    return fields


def index(request: dict[str, Any]) -> dict[str, Any]:
    # 1. The Markdown files of the folder, in order.
    folder: str = request["folder"]
    limit: int = request["max_entries"]
    names: list[str] = []
    if os.path.isdir(folder):
        names = sorted(name for name in os.listdir(folder) if name.endswith(".md"))
    # 2. What each one says about itself.
    entries: list[dict[str, str]] = []
    for name in names[:limit]:
        fields = header(os.path.join(folder, name))
        description = fields.get("description", "")[: request["max_description"]]
        entries.append({"file": name, "type": fields.get("type", ""), "description": description})
    return {"ok": True, "entries": entries, "left_out": max(0, len(names) - limit)}


if __name__ == "__main__":
    print(json.dumps(index(json.loads(sys.argv[1]))))
