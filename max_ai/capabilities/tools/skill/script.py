"""Runs inside the sandbox: reads one skill for read_skill. Standard library only."""

import json
import os
import sys
from typing import Any


def read(request: dict[str, Any]) -> dict[str, Any]:
    folder: str = request["folder"]
    name: str = request["file"]
    # 1. The skill is a folder; without it, it is not installed.
    if not os.path.isdir(folder):
        return {"ok": False, "error": "missing"}

    # 2. Its instructions, never through a link that leads out of the skill.
    instructions = os.path.realpath(os.path.join(folder, name))
    if not instructions.startswith(os.path.realpath(folder) + os.sep):
        return {"ok": False, "error": "no_instructions"}
    try:
        with open(instructions, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return {"ok": False, "error": "no_instructions"}

    # 3. Its files, so the model can open them.
    files: list[str] = []
    for path, _, names in os.walk(folder):
        files += [os.path.relpath(os.path.join(path, each), folder) for each in names]
    return {"ok": True, "files": sorted(files)[: request["max_files"]], "instructions": text}


print(json.dumps(read(json.loads(sys.argv[1]))))
