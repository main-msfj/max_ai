"""Runs inside the sandbox: writes one memory file for save_memory. Standard library only."""

import hashlib
import json
import os
import sys
from typing import Any


def save(request: dict[str, Any], content: bytes) -> dict[str, bool]:
    # 1. All of the content arrived.
    if hashlib.sha256(content).hexdigest() != request["sha256"]:
        raise ValueError("its content arrived incomplete")

    # 2. The file is in the memory folder itself, never through a link out of it.
    folder: str = os.path.realpath(request["folder"])
    file = os.path.realpath(os.path.join(folder, request["file"]))
    if os.path.dirname(file) != folder:
        raise ValueError("it would be written outside the memory")

    # 3. Write beside the file and swap, so a cut write leaves no half file.
    existed = os.path.exists(file)
    os.makedirs(folder, exist_ok=True)
    partial = file + ".partial"
    with open(partial, "wb") as handle:
        handle.write(content)
    os.replace(partial, file)
    return {"existed": existed}


def main() -> None:
    try:
        answer: dict[str, Any] = {"ok": True, **save(json.loads(sys.argv[1]), sys.stdin.buffer.read())}
    except (ValueError, OSError) as error:
        answer = {"ok": False, "error": str(error)}
    print(json.dumps(answer))


if __name__ == "__main__":
    main()
