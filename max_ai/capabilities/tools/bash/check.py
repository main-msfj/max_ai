"""Runs inside the sandbox after a bash command: which of the files the
model had read are no longer the same. Standard library only.

The request is one JSON argument, path -> sha256 the model saw; the answer
is one JSON on stdout with the paths that changed and the ones that are gone.
"""

import hashlib
import json
import os
import sys


def main() -> None:
    seen: dict[str, str] = json.loads(sys.argv[1])
    changed: list[str] = []
    deleted: list[str] = []
    for path, digest in seen.items():
        # 1. Gone, or no longer a file.
        if not os.path.isfile(path):
            deleted.append(path)
            continue
        # 2. Still there: the same content or not. One that cannot be read
        #    is left alone; the file tools say why when the model opens it.
        try:
            with open(path, "rb") as handle:
                now = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            continue
        if now != digest:
            changed.append(path)
    print(json.dumps({"changed": changed, "deleted": deleted}))


main()
