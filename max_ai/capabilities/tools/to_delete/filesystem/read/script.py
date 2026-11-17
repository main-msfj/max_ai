"""read_file inside the environment.

Standard library only: reads one JSON request on stdin and prints one JSON
answer. Goes through the file line by line, so a big file is never loaded
whole, and returns a stamp so a later edit can tell whether the file changed.
"""

import hashlib
import json
import os
import sys

LIMIT = 2000


def resolve(root, path):
    root = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError(f"{path} is outside the workspace")
    return full


def read(full, offset, limit):
    digest = hashlib.sha256()
    lines = []
    total = 0
    with open(full, "rb") as file:
        for total, raw in enumerate(file, start=1):
            digest.update(raw)
            if offset <= total < offset + limit:
                text = raw.decode("utf-8").rstrip("\r\n")
                lines.append(f"{total:>6}\t{text}")
    return {"content": "\n".join(lines), "total_lines": total, "stamp": digest.hexdigest()}


def main():
    request = json.load(sys.stdin)
    try:
        full = resolve(request["root"], request["path"])
        answer = read(full, request.get("offset", 1), request.get("limit", LIMIT))
        print(json.dumps({"ok": True, **answer}))
    except (ValueError, OSError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))


if __name__ == "__main__":
    main()
